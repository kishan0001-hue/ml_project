from datetime import datetime, timezone, date, time, timedelta
from decimal import Decimal
from enum import Enum

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    Time,
    Numeric,
    UniqueConstraint
)

from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Department(Base):
    __tablename__ = "departments"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True
    )

    name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False
    )

    code: Mapped[str] = mapped_column(
        String(20),
        unique=True,
        nullable=False,
        index=True
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    status: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
    )

    employees: Mapped[list["Employee"]] = relationship(
        back_populates="department"
    )

class Employee(Base):
    __tablename__ = "employees"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True
    )

    employee_code: Mapped[str] = mapped_column(
        String(30),
        unique=True,
        nullable=False,
        index=True
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    department_id: Mapped[int] = mapped_column(
        ForeignKey("departments.id"),
        nullable=False
    )

    designation: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    joining_date: Mapped[date] = mapped_column(
        Date,
        nullable=False
    )

    status: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
    )

    department: Mapped["Department"] = relationship(
        back_populates="employees"
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True
    )

    login_id: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )

    name: Mapped[str] = mapped_column(
        String(100), nullable=False
    )

    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )

    password_hash: Mapped[str] = mapped_column(
        String(255), nullable=False
    )

    role: Mapped[str] = mapped_column(
        String(20), nullable=False
    )

    department_id: Mapped[int | None] = mapped_column(
        ForeignKey("departments.id"),
        nullable=True,
        index=True,
    )

    department: Mapped["Department | None"] = relationship()

    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    salary: Mapped["Salary | None"] = relationship(
        "Salary",
        back_populates="employee",
        uselist=False,
        cascade="all, delete-orphan",
    )

    employee_profile: Mapped["EmployeeProfile | None"] = relationship(
        "EmployeeProfile",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )

    refresh_tokens: Mapped[list["RefreshToken"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False
    )

    jti: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        nullable=False,
        index=True
    )

    token_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False
    )

    expires_at: Mapped[datetime] = mapped_column(
        nullable=False
    )

    revoked: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    user: Mapped["User"] = relationship(
        back_populates="refresh_tokens"
    )

class Attendance(Base):
    __tablename__ = "attendance"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True
    )

    attendance_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        index=True
    )

    check_in: Mapped[time | None] = mapped_column(
        Time,
        nullable=True
    )

    check_out: Mapped[time | None] = mapped_column(
        Time,
        nullable=True
    )

    status: Mapped[str] = mapped_column(
        String(20),
        default="PRESENT",
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    user: Mapped["User"] = relationship()

    breaks: Mapped[list["BreakRecord"]] = relationship(
        back_populates="attendance",
        cascade="all, delete-orphan"
    )

class BreakRecord(Base):
    __tablename__ = "break_records"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True
    )

    attendance_id: Mapped[int] = mapped_column(
        ForeignKey("attendance.id"),
        nullable=False,
        index=True
    )

    break_in: Mapped[time] = mapped_column(
        Time,
        nullable=False
    )

    break_out: Mapped[time | None] = mapped_column(
        Time,
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    attendance: Mapped["Attendance"] = relationship(
        back_populates="breaks"
    )

class LeaveStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class Leave(Base):
    __tablename__ = "leaves"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    department_id: Mapped[int] = mapped_column(
        ForeignKey("departments.id"),
        nullable=False,
        index=True,
    )

    leave_type: Mapped[str] = mapped_column(
        nullable=False,
    )

    start_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    end_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    status: Mapped[LeaveStatus] = mapped_column(
        default=LeaveStatus.PENDING,
        nullable=False,
    )

    # Approval status
    hr_status: Mapped[ApprovalStatus] = mapped_column(
        default=ApprovalStatus.PENDING,
        nullable=False,
    )

    manager_status: Mapped[ApprovalStatus] = mapped_column(
        default=ApprovalStatus.PENDING,
        nullable=False,
    )

    admin_status: Mapped[ApprovalStatus] = mapped_column(
        default=ApprovalStatus.PENDING,
        nullable=False,
    )

    # Who approved/rejected
    hr_approved_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
    )

    manager_approved_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
    )

    admin_approved_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
    )

    # Approval timestamps
    hr_approved_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    manager_approved_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    admin_approved_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    is_paid: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )

    duration: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="FULL_DAY",
        server_default="FULL_DAY",
    )

    # Relationships
    user = relationship(
        "User",
        foreign_keys=[user_id],
    )

    department = relationship(
        "Department",
        foreign_keys=[department_id],
    )

    hr_approver = relationship(
        "User",
        foreign_keys=[hr_approved_by],
    )

    manager_approver = relationship(
        "User",
        foreign_keys=[manager_approved_by],
    )

    admin_approver = relationship(
        "User",
        foreign_keys=[admin_approved_by],
    )

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Date, DateTime, ForeignKey, Numeric
from sqlalchemy.orm import Mapped, mapped_column, relationship


class Salary(Base):
    __tablename__ = "salaries"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    employee_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False
    )

    basic_salary: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False
    )

    hra: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        default=0,
        nullable=False
    )

    allowances: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        default=0,
        nullable=False
    )

    deductions: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        default=0,
        nullable=False
    )

    effective_from: Mapped[date] = mapped_column(
        Date,
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

    employee = relationship(
        "User",
        back_populates="salary"
    )

class PayrollStatus(str, Enum):
    DRAFT = "DRAFT"
    GENERATED = "GENERATED"
    APPROVED = "APPROVED"
    PAID = "PAID"


class Payroll(Base):
    __tablename__ = "payrolls"

    __table_args__ = (
        UniqueConstraint(
            "employee_id",
            "year",
            "month",
            name="unique_employee_payroll_month"
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True
    )

    employee_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    year: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    month: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    basic_salary: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False
    )

    hra: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
        default=0
    )

    allowances: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
        default=0
    )

    deductions: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
        default=0
    )

    gross_salary: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False
    )

    unpaid_leave_days: Mapped[Decimal] = mapped_column(
        Numeric(6, 2),
        nullable=False,
        default=0,
        server_default="0",
    )

    unpaid_leave_deduction: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
        default=0,
        server_default="0",
    )

    net_salary: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False
    )

    status: Mapped[PayrollStatus] = mapped_column(
        String(20),
        nullable=False,
        default=PayrollStatus.DRAFT
    )

    generated_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True
    )

    approved_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True
    )

    paid_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

    employee = relationship(
        "User",
        foreign_keys=[employee_id]
    )

    generator = relationship(
        "User",
        foreign_keys=[generated_by]
    )

    approver = relationship(
        "User",
        foreign_keys=[approved_by]
    )

class EmployeeProfile(Base):
    __tablename__ = "employee_profiles"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True
    )

    employee_code: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
        index=True
    )

    phone: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True
    )

    address: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    date_of_birth: Mapped[date | None] = mapped_column(
        Date,
        nullable=True
    )

    joining_date: Mapped[date] = mapped_column(
        Date,
        nullable=False
    )

    designation: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True
    )

    employment_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="ACTIVE"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

    user = relationship(
        "User",
        back_populates="employee_profile"
    )