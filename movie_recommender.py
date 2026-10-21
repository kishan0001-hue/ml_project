#!/usr/bin/env python3
"""
Bollywood + Hollywood Movie Recommendation System
=================================================
Three recommenders built with pandas + scikit-learn:

1. Content-based   TF-IDF over genres / director / cast / keywords / overview,
                   compared with cosine similarity.
2. Collaborative   Item-based CF: cosine similarity between mean-centred
                   item rating vectors (users x movies matrix).
3. Hybrid          Weighted blend of the two (alpha controls the mix) which
                   also helps with cold-start: unknown users fall back to
                   content-based on the titles they say they like.

Usage
-----
    python movie_recommender.py                         # demo
    python movie_recommender.py --title "3 Idiots"
    python movie_recommender.py --title Inception --industry Bollywood
    python movie_recommender.py --user 7 --n 8
    python movie_recommender.py --liked "Dangal,Rocky,Lagaan"
    python movie_recommender.py --evaluate

Data
----
movies.csv is bundled. If ratings.csv (columns: user_id, movie_id, rating) is
missing, a SYNTHETIC ratings file is generated so collaborative filtering has
something to work with. To use real data, drop in your own ratings.csv
(e.g. a MovieLens / Kaggle export mapped to the movie_ids in movies.csv).
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _tok(text: str) -> str:
    """'Leonardo DiCaprio' -> 'leonardodicaprio' so names stay single tokens."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def load_movies(path: Path = DATA_DIR / "movies.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("genres", "cast", "keywords"):
        df[col] = df[col].fillna("").str.split("|")
    df.insert(0, "movie_id", range(1, len(df) + 1))
    return df


def generate_synthetic_ratings(movies: pd.DataFrame, n_users: int = 150,
                               seed: int = 42) -> pd.DataFrame:
    """Create taste-driven fake ratings: each user has a genre persona and an
    optional industry preference, rates ~12-25 movies, mostly ones they'd like."""
    rng = np.random.default_rng(seed)
    personas = {
        "action_fan":  {"Action", "Sci-Fi", "Thriller", "Adventure"},
        "romance_fan": {"Romance", "Musical", "Drama"},
        "comedy_fan":  {"Comedy", "Family", "Animation"},
        "drama_buff":  {"Drama", "Biography", "Crime"},
        "sports_fan":  {"Sport", "Biography", "Drama"},
        "mystery_fan": {"Mystery", "Thriller", "Crime"},
    }
    names = list(personas)
    affinity_cols = np.array([[len(set(g) & fav) / len(g) for g in movies.genres]
                              for fav in personas.values()])
    rows = []
    for uid in range(1, n_users + 1):
        p = rng.integers(len(names))
        affin = affinity_cols[p]
        pref = rng.choice(["Bollywood", "Hollywood", "none"], p=[0.3, 0.3, 0.4])
        bonus = (movies.industry.values == pref).astype(float) * 0.6
        prob = 0.3 + affin + bonus
        prob /= prob.sum()
        k = rng.integers(12, 26)
        picks = rng.choice(len(movies), size=k, replace=False, p=prob)
        for i in picks:
            r = 2.4 + 2.2 * affin[i] + bonus[i] + rng.normal(0, 0.5)
            rows.append((uid, int(movies.movie_id.iloc[i]),
                         int(np.clip(round(r), 1, 5))))
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"])


def load_ratings(movies: pd.DataFrame,
                 path: Path = DATA_DIR / "ratings.csv") -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    ratings = generate_synthetic_ratings(movies)
    ratings.to_csv(path, index=False)
    print(f"[info] No ratings.csv found - wrote synthetic ratings to {path.name}")
    return ratings


# --------------------------------------------------------------------------- #
# Recommender
# --------------------------------------------------------------------------- #
def _minmax(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Scale to [0, 1], computing min/max only over `mask` if given."""
    vals = x[mask] if mask is not None else x
    lo, hi = vals.min(), vals.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x, dtype=float)


class MovieRecommender:
    def __init__(self, movies: pd.DataFrame, ratings: pd.DataFrame):
        self.movies = movies.reset_index(drop=True)
        self.ratings = ratings
        self.id_to_idx = {mid: i for i, mid in enumerate(self.movies.movie_id)}
        self._build_content_model()
        self._build_cf_model()

    # ----- content-based ---------------------------------------------------
    @staticmethod
    def _soup(row) -> str:
        # Repetition acts as a cheap field weight inside TF-IDF.
        genres = [_tok(g) for g in row.genres] * 3
        director = [_tok(d) for d in row.director.split("|")] * 2
        cast = [_tok(c) for c in row.cast] * 2
        keywords = (" ".join(row.keywords) + " ") * 2
        return " ".join(genres + director + cast) + " " + keywords + row.overview

    def _build_content_model(self):
        soups = self.movies.apply(self._soup, axis=1)
        self.tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True,
                                     ngram_range=(1, 1), min_df=1)
        self.tfidf_matrix = self.tfidf.fit_transform(soups)
        self.content_sim = cosine_similarity(self.tfidf_matrix)

    # ----- collaborative filtering ----------------------------------------
    def _build_cf_model(self):
        R = (self.ratings
             .pivot_table(index="user_id", columns="movie_id", values="rating")
             .reindex(columns=self.movies.movie_id))
        self.R = R
        self.user_mean = R.mean(axis=1)
        centred = R.sub(self.user_mean, axis=0).fillna(0.0).values
        sim = cosine_similarity(centred.T)           # item x item
        np.fill_diagonal(sim, 0.0)
        self.cf_sim = np.clip(sim, 0.0, None)        # keep positive evidence only
        self.item_popularity = R.count(axis=0).values

    # ----- helpers ---------------------------------------------------------
    def resolve_title(self, title: str) -> int:
        lowered = self.movies.title.str.lower()
        exact = np.where(lowered == title.strip().lower())[0]
        if len(exact):
            return int(exact[0])
        close = difflib.get_close_matches(title.lower(), lowered.tolist(), n=3, cutoff=0.5)
        hint = f" Did you mean: {', '.join(self.movies.title[lowered == c].iloc[0] for c in close)}?" if close else ""
        raise ValueError(f"Movie '{title}' not found.{hint}")

    def _format(self, idxs, scores) -> pd.DataFrame:
        out = self.movies.iloc[idxs][["title", "year", "industry", "genres"]].copy()
        out["genres"] = out["genres"].apply(lambda g: ", ".join(g))
        out["score"] = np.round(scores, 3)
        return out.reset_index(drop=True)

    def _industry_mask(self, industry: str | None) -> np.ndarray:
        if industry is None:
            return np.ones(len(self.movies), dtype=bool)
        return (self.movies.industry.str.lower() == industry.lower()).to_numpy(copy=True)

    # ----- 1) similar-movie recommendations -------------------------------
    def recommend_similar(self, title: str, n: int = 10, mode: str = "hybrid",
                          alpha: float = 0.6, industry: str | None = None) -> pd.DataFrame:
        """alpha = weight on content similarity in hybrid mode."""
        i = self.resolve_title(title)
        valid = self._industry_mask(industry)
        valid[i] = False

        content = _minmax(self.content_sim[i], valid)
        cf = _minmax(self.cf_sim[i], valid)
        if mode == "content":
            score = content
        elif mode == "cf":
            score = cf
        elif mode == "hybrid":
            score = alpha * content + (1 - alpha) * cf
        else:
            raise ValueError("mode must be 'content', 'cf' or 'hybrid'")

        score = np.where(valid, score, -np.inf)
        top = np.argsort(-score)[:n]
        return self._format(top, score[top])

    # ----- 2) personalised recommendations --------------------------------
    def _user_scores(self, rated: pd.Series):
        """rated: Series indexed by movie_id. Returns (content_score, cf_score) arrays."""
        idx = np.array([self.id_to_idx[m] for m in rated.index])
        r = rated.values.astype(float)
        mean = r.mean()

        # content: taste profile = weighted mean of TF-IDF vectors of liked movies
        w = np.clip(r - mean, 0, None)
        if w.sum() == 0:
            w = np.ones_like(r)
        profile = np.asarray(self.tfidf_matrix[idx].T @ w).reshape(1, -1) / w.sum()
        content = cosine_similarity(profile, self.tfidf_matrix).ravel()

        # CF: weighted average of the user's centred ratings over similar items
        S = self.cf_sim[:, idx]
        num = S @ (r - mean)
        den = S.sum(axis=1)
        pred = mean + np.divide(num, den, out=np.zeros_like(num), where=den > 1e-9)
        return content, pred

    def _rank(self, rated: pd.Series, n, mode, alpha, industry):
        content, cf = self._user_scores(rated)
        seen = np.array([self.id_to_idx[m] for m in rated.index])
        valid = self._industry_mask(industry)
        valid[seen] = False

        if mode == "popular":
            score = _minmax(self.item_popularity.astype(float), valid)
        elif mode == "content":
            score = _minmax(content, valid)
        elif mode == "cf":
            score = _minmax(cf, valid)
        else:
            score = alpha * _minmax(content, valid) + (1 - alpha) * _minmax(cf, valid)
        score = np.where(valid, score, -np.inf)
        top = np.argsort(-score)[:n]
        return top, score[top]

    def recommend_for_user(self, user_id: int, n: int = 10, mode: str = "hybrid",
                           alpha: float = 0.4, industry: str | None = None) -> pd.DataFrame:
        if user_id not in self.R.index:
            raise ValueError(f"Unknown user_id {user_id}. Use --liked for new users.")
        rated = self.R.loc[user_id].dropna()
        top, sc = self._rank(rated, n, mode, alpha, industry)
        return self._format(top, sc)

    # ----- 3) cold start: user supplies a few liked titles ----------------
    def recommend_for_liked(self, titles: list[str], n: int = 10,
                            industry: str | None = None) -> pd.DataFrame:
        ids = [int(self.movies.movie_id.iloc[self.resolve_title(t)]) for t in titles]
        rated = pd.Series(5.0, index=ids)
        top, sc = self._rank(rated, n, "content", 1.0, industry)
        return self._format(top, sc)


# --------------------------------------------------------------------------- #
# Offline evaluation (hold-out ranking metrics)
# --------------------------------------------------------------------------- #
def evaluate(movies: pd.DataFrame, ratings: pd.DataFrame, k: int = 10,
             test_frac: float = 0.25, seed: int = 0) -> pd.DataFrame:
    """Hide a fraction of each user's ratings, recommend from the rest, and
    check how many hidden *liked* (rating >= 4) movies appear in the top-k."""
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, grp in ratings.groupby("user_id"):
        if len(grp) >= 8:
            test_idx += list(rng.choice(grp.index, size=max(1, int(len(grp) * test_frac)),
                                        replace=False))
    test = ratings.loc[test_idx]
    train = ratings.drop(test_idx)
    model = MovieRecommender(movies, train)

    configs = {"popularity": ("popular", 0), "content-based": ("content", 1),
               "collaborative": ("cf", 0), "hybrid (a=0.4)": ("hybrid", 0.4)}
    results = {name: {"precision": [], "recall": [], "hit": []} for name in configs}

    for uid, grp in test[test.rating >= 4].groupby("user_id"):
        if uid not in model.R.index:
            continue
        relevant = set(grp.movie_id)
        rated = model.R.loc[uid].dropna()
        for name, (mode, alpha) in configs.items():
            top, _ = model._rank(rated, k, mode, alpha, None)
            recs = set(model.movies.movie_id.iloc[top])
            hits = len(recs & relevant)
            results[name]["precision"].append(hits / k)
            results[name]["recall"].append(hits / len(relevant))
            results[name]["hit"].append(hits > 0)

    rows = [{"model": name,
             f"precision@{k}": np.mean(m["precision"]),
             f"recall@{k}": np.mean(m["recall"]),
             f"hit_rate@{k}": np.mean(m["hit"])} for name, m in results.items()]
    return pd.DataFrame(rows).round(3)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _show(title: str, df: pd.DataFrame):
    print(f"\n=== {title} ===")
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="Bollywood + Hollywood recommender")
    ap.add_argument("--title", help="Recommend movies similar to this title")
    ap.add_argument("--user", type=int, help="Recommend for an existing user_id")
    ap.add_argument("--liked", help="Comma-separated liked titles (new user / cold start)")
    ap.add_argument("--industry", choices=["Bollywood", "Hollywood"],
                    help="Restrict recommendations to one industry")
    ap.add_argument("--mode", default="hybrid", choices=["content", "cf", "hybrid"])
    ap.add_argument("--alpha", type=float, default=0.6, help="Content weight in hybrid mode")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--evaluate", action="store_true", help="Run hold-out evaluation")
    args = ap.parse_args()

    movies = load_movies()
    ratings = load_ratings(movies)
    rec = MovieRecommender(movies, ratings)
    print(f"Loaded {len(movies)} movies "
          f"({(movies.industry == 'Bollywood').sum()} Bollywood / "
          f"{(movies.industry == 'Hollywood').sum()} Hollywood), "
          f"{ratings.user_id.nunique()} users, {len(ratings)} ratings")

    try:
        if args.evaluate:
            _show("Hold-out evaluation", evaluate(movies, ratings))
        elif args.title:
            _show(f"Because you liked '{args.title}' [{args.mode}"
                  f"{', ' + args.industry if args.industry else ''}]",
                  rec.recommend_similar(args.title, args.n, args.mode, args.alpha, args.industry))
        elif args.user is not None:
            hist = rec.R.loc[args.user].dropna().sort_values(ascending=False).head(5)
            liked = movies.set_index("movie_id").loc[hist.index, "title"].tolist()
            print(f"\nUser {args.user}'s top-rated: {', '.join(liked)}")
            _show(f"Recommendations for user {args.user}",
                  rec.recommend_for_user(args.user, args.n, args.mode, industry=args.industry))
        elif args.liked:
            titles = [t.strip() for t in args.liked.split(",")]
            _show(f"Cold-start picks for someone who likes {titles}",
                  rec.recommend_for_liked(titles, args.n, args.industry))
        else:  # demo
            _show("Similar to 'Inception' (hybrid, all)", rec.recommend_similar("Inception", 5))
            _show("Similar to 'Inception' -> Bollywood only (cross-industry)",
                  rec.recommend_similar("Inception", 5, industry="Bollywood"))
            _show("Similar to '3 Idiots' -> Hollywood only (cross-industry)",
                  rec.recommend_similar("3 Idiots", 5, industry="Hollywood"))
            _show("Similar to 'Dangal' (content-based)",
                  rec.recommend_similar("Dangal", 5, mode="content"))
            _show("Cold start: likes Titanic + La La Land",
                  rec.recommend_for_liked(["Titanic", "La La Land"], 5))
            _show("Personalised for user 7 (hybrid)", rec.recommend_for_user(7, 5))
            _show("Hold-out evaluation", evaluate(movies, ratings))
    except ValueError as e:
        raise SystemExit(f"Error: {e}")


if __name__ == "__main__":
    main()
