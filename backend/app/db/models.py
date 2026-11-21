"""
Step 1: SQLAlchemy models for the "AppSec Testing & Vulnerability Tracker"
synthetic dataset.

These models define the *sample* database only. The agent itself never
imports them: Step 0's introspector reads the live schema, so the agent
works the same against any database. These models exist to create and
seed the demo data.

Tables (8):
    users, assets, test_cases, test_runs, defects,
    scans, vulnerabilities, scan_findings
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Date,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(150), nullable=False, unique=True)
    role: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # tester | analyst | developer | security_engineer
    team: Mapped[str] = mapped_column(String(50), nullable=False)


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    environment: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # dev | staging | production
    owner_team: Mapped[str] = mapped_column(String(50), nullable=False)
    criticality: Mapped[str] = mapped_column(
        String(10), nullable=False
    )  # low | medium | high | critical

    vulnerabilities: Mapped[list[Vulnerability]] = relationship(back_populates="asset")


class TestCase(Base):
    __tablename__ = "test_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    test_type: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # functional | security | regression
    priority: Mapped[str] = mapped_column(String(10), nullable=False)  # P1 | P2 | P3

    runs: Mapped[list[TestRun]] = relationship(back_populates="test_case")


class TestRun(Base):
    __tablename__ = "test_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    test_case_id: Mapped[int] = mapped_column(ForeignKey("test_cases.id"), nullable=False)
    tester_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    run_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    status: Mapped[str] = mapped_column(String(10), nullable=False)  # pass | fail | blocked
    build_version: Mapped[str] = mapped_column(String(20), nullable=False)

    test_case: Mapped[TestCase] = relationship(back_populates="runs")
    defects: Mapped[list[Defect]] = relationship(back_populates="test_run")


class Defect(Base):
    __tablename__ = "defects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    test_run_id: Mapped[int] = mapped_column(ForeignKey("test_runs.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    severity: Mapped[str] = mapped_column(
        String(10), nullable=False
    )  # low | medium | high | critical
    status: Mapped[str] = mapped_column(
        String(15), nullable=False
    )  # open | in_progress | resolved | closed
    assigned_to: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    reported_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    resolved_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)

    test_run: Mapped[TestRun] = relationship(back_populates="defects")


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id"), nullable=False)
    scan_type: Mapped[str] = mapped_column(String(20), nullable=False)  # SAST | DAST | dependency
    tool_name: Mapped[str] = mapped_column(String(50), nullable=False)
    scan_date: Mapped[dt.date] = mapped_column(Date, nullable=False)

    findings: Mapped[list[ScanFinding]] = relationship(back_populates="scan")


class Vulnerability(Base):
    __tablename__ = "vulnerabilities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id"), nullable=False)
    cve_id: Mapped[str | None] = mapped_column(String(20), nullable=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    severity: Mapped[str] = mapped_column(
        String(10), nullable=False
    )  # low | medium | high | critical
    cvss_score: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(15), nullable=False)  # open | fixed | accepted
    discovered_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    fixed_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)

    asset: Mapped[Asset] = relationship(back_populates="vulnerabilities")
    scan_findings: Mapped[list[ScanFinding]] = relationship(back_populates="vulnerability")


class ScanFinding(Base):
    """Join table: which vulnerabilities a given scan reported."""

    __tablename__ = "scan_findings"

    scan_id: Mapped[int] = mapped_column(ForeignKey("scans.id"), primary_key=True)
    vulnerability_id: Mapped[int] = mapped_column(
        ForeignKey("vulnerabilities.id"), primary_key=True
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    scan: Mapped[Scan] = relationship(back_populates="findings")
    vulnerability: Mapped[Vulnerability] = relationship(back_populates="scan_findings")
