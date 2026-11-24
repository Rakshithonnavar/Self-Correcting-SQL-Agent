"""
Step 1: deterministic synthetic data generator for the AppSec Testing &
Vulnerability Tracker sample database.

Usage (from backend/):
    python -m app.db.seed_data --reset                         #uses DATABASE_URL from .env
    python -m app.db.seed_data --url sqlite:///./dev.db
    python -m app.db.seed_data --url sqlite:///./dev.db --seed 42 --reset

Same seed always produces the same data, so eval answers and the README
examples stay stable across machines.

Default volumes (meets the plan: ~20 assets, 500+ test runs, 200+ vulns):
    20 assets, 12 users, 60 test cases, 600 test runs, ~120 defects,
    80 scans, 240 vulnerabilities, ~400 scan findings.
"""

from __future__ import annotations

import argparse
import datetime as dt
import random
from dataclasses import dataclass

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import (
    Asset,
    Base,
    Defect,
    Scan,
    ScanFinding,
    TestCase,
    TestRun,
    User,
    Vulnerability,
)

ASSET_NAMES = [
    "customer-portal",
    "payments-api",
    "admin-console",
    "mobile-gateway",
    "search-service",
    "auth-service",
    "reporting-dashboard",
    "notifications-worker",
    "inventory-api",
    "order-service",
    "data-ingest-pipeline",
    "partner-webhooks",
    "marketing-site",
    "help-center",
    "identity-provider",
    "billing-engine",
    "analytics-api",
    "file-storage-service",
    "chat-support-widget",
    "internal-wiki",
]
ENVIRONMENTS = ["dev", "staging", "production"]
TEAMS = ["platform", "payments", "growth", "security", "data", "mobile"]
CRITICALITY = ["low", "medium", "high", "critical"]

FIRST_NAMES = [
    "Asha",
    "Ben",
    "Chen",
    "Dana",
    "Elif",
    "Farid",
    "Grace",
    "Hiro",
    "Ines",
    "Jonas",
    "Kiran",
    "Lena",
]
LAST_NAMES = [
    "Iyer",
    "Okafor",
    "Lindqvist",
    "Moreau",
    "Patel",
    "Rossi",
    "Tanaka",
    "Novak",
    "Garcia",
    "Haddad",
    "Kim",
    "Silva",
]
USER_ROLES = ["tester", "analyst", "developer", "security_engineer"]

TEST_TYPES = ["functional", "security", "regression"]
PRIORITIES = ["P1", "P2", "P3"]
RUN_STATUSES = ["pass", "fail", "blocked"]
RUN_STATUS_WEIGHTS = [0.68, 0.22, 0.10]

DEFECT_SEVERITIES = ["low", "medium", "high", "critical"]
DEFECT_STATUSES = ["open", "in_progress", "resolved", "closed"]

SCAN_TYPES = ["SAST", "DAST", "dependency"]
SCAN_TOOLS = {
    "SAST": ["semgrep", "bandit"],
    "DAST": ["OWASP ZAP", "nuclei"],
    "dependency": ["pip-audit", "npm audit"],
}

VULN_TITLES = [
    "SQL injection in search parameter",
    "Stored XSS in profile bio",
    "Missing rate limiting on login",
    "Outdated TLS configuration",
    "Insecure direct object reference on invoices",
    "Verbose error messages leak stack traces",
    "Hardcoded API key in config",
    "Vulnerable dependency: outdated JWT library",
    "Missing CSRF token on password change",
    "Open redirect after login",
    "Sensitive data in application logs",
    "Weak password hashing parameters",
]
VULN_SEVERITY_CVSS = {
    "critical": (9.0, 10.0),
    "high": (7.0, 8.9),
    "medium": (4.0, 6.9),
    "low": (0.1, 3.9),
}
VULN_SEVERITY_WEIGHTS = {"critical": 0.08, "high": 0.25, "medium": 0.40, "low": 0.27}

TEST_CASE_TEMPLATES = [
    "Login with valid credentials",
    "Reject expired session token",
    "Password reset email delivered",
    "Checkout total matches cart",
    "Search returns paginated results",
    "Upload rejects oversized file",
    "Role-based access on admin pages",
    "API returns 401 without token",
    "Input validation on signup form",
    "Export report as CSV",
]


@dataclass
class SeedVolumes:
    users: int = 12
    assets: int = 20
    test_cases_per_asset: int = 3
    test_runs: int = 600
    scans: int = 80
    vulnerabilities: int = 240
    max_findings_per_scan: int = 8


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights.keys())
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def _random_date(rng: random.Random, start: dt.date, end: dt.date) -> dt.date:
    span = (end - start).days
    return start + dt.timedelta(days=rng.randint(0, span))


def generate(seed: int = 42, volumes: SeedVolumes | None = None) -> dict[str, list]:
    """
    Build all rows in memory, with no DB access. Returned as plain model
    instances so the caller decides how to persist them. Pure function of
    `seed` and `volumes`: no global state, no clock reads.
    """
    v = volumes or SeedVolumes()
    rng = random.Random(seed)
    start = dt.date(2025, 1, 1)
    end = dt.date(2026, 10, 31)

    users = []
    for i in range(v.users):
        first, last = FIRST_NAMES[i % len(FIRST_NAMES)], LAST_NAMES[i % len(LAST_NAMES)]
        users.append(
            User(
                id=i + 1,
                name=f"{first} {last}",
                email=f"{first.lower()}.{last.lower()}@example.com",
                role=USER_ROLES[i % len(USER_ROLES)],
                team=rng.choice(TEAMS),
            )
        )

    assets = []
    for i in range(v.assets):
        assets.append(
            Asset(
                id=i + 1,
                name=ASSET_NAMES[i % len(ASSET_NAMES)],
                environment=rng.choices(ENVIRONMENTS, weights=[0.3, 0.2, 0.5], k=1)[0],
                owner_team=rng.choice(TEAMS),
                criticality=rng.choices(CRITICALITY, weights=[0.2, 0.4, 0.28, 0.12], k=1)[0],
            )
        )

    test_cases = []
    tc_id = 1
    for asset in assets:
        for _ in range(v.test_cases_per_asset):
            test_cases.append(
                TestCase(
                    id=tc_id,
                    asset_id=asset.id,
                    title=f"{rng.choice(TEST_CASE_TEMPLATES)} ({asset.name})",
                    test_type=rng.choice(TEST_TYPES),
                    priority=rng.choices(PRIORITIES, weights=[0.25, 0.5, 0.25], k=1)[0],
                )
            )
            tc_id += 1

    test_runs = []
    for i in range(v.test_runs):
        tc = rng.choice(test_cases)
        test_runs.append(
            TestRun(
                id=i + 1,
                test_case_id=tc.id,
                tester_id=rng.choice(users).id,
                run_date=_random_date(rng, start, end),
                status=rng.choices(RUN_STATUSES, weights=RUN_STATUS_WEIGHTS, k=1)[0],
                build_version=f"v{rng.randint(1, 4)}.{rng.randint(0, 9)}.{rng.randint(0, 20)}",
            )
        )

    # Defects only come from failed or blocked runs, which is how real trackers behave.
    defects = []
    d_id = 1
    for run in test_runs:
        if run.status == "pass" or rng.random() > 0.35:
            continue
        reported = run.run_date
        resolved_flag = rng.random() < 0.6
        resolved = reported + dt.timedelta(days=rng.randint(1, 30)) if resolved_flag else None
        status = (
            rng.choice(["resolved", "closed"])
            if resolved_flag
            else rng.choice(["open", "in_progress"])
        )
        defects.append(
            Defect(
                id=d_id,
                test_run_id=run.id,
                title=f"Failure in test run {run.id}",
                severity=rng.choices(DEFECT_SEVERITIES, weights=[0.3, 0.4, 0.2, 0.1], k=1)[0],
                status=status,
                assigned_to=rng.choice(users).id,
                reported_date=reported,
                resolved_date=resolved,
            )
        )
        d_id += 1

    scans = []
    for i in range(v.scans):
        scan_type = rng.choice(SCAN_TYPES)
        scans.append(
            Scan(
                id=i + 1,
                asset_id=rng.choice(assets).id,
                scan_type=scan_type,
                tool_name=rng.choice(SCAN_TOOLS[scan_type]),
                scan_date=_random_date(rng, start, end),
            )
        )

    vulnerabilities = []
    for i in range(v.vulnerabilities):
        severity = _weighted_choice(rng, VULN_SEVERITY_WEIGHTS)
        lo, hi = VULN_SEVERITY_CVSS[severity]
        discovered = _random_date(rng, start, end)
        status = rng.choices(["open", "fixed", "accepted"], weights=[0.35, 0.55, 0.10], k=1)[0]
        fixed = discovered + dt.timedelta(days=rng.randint(2, 120)) if status == "fixed" else None
        cve = (
            f"CVE-{rng.randint(2024, 2026)}-{rng.randint(1000, 49999)}"
            if rng.random() < 0.6
            else None
        )
        vulnerabilities.append(
            Vulnerability(
                id=i + 1,
                asset_id=rng.choice(assets).id,
                cve_id=cve,
                title=rng.choice(VULN_TITLES),
                severity=severity,
                cvss_score=round(rng.uniform(lo, hi), 1),
                status=status,
                discovered_date=discovered,
                fixed_date=fixed,
            )
        )

    # Each scan reports a random subset of vulnerabilities on the same asset.
    scan_findings = []
    vulns_by_asset: dict[int, list[Vulnerability]] = {}
    for vuln in vulnerabilities:
        vulns_by_asset.setdefault(vuln.asset_id, []).append(vuln)
    for scan in scans:
        candidates = vulns_by_asset.get(scan.asset_id, [])
        if not candidates:
            continue
        k = min(len(candidates), rng.randint(1, v.max_findings_per_scan))
        for vuln in rng.sample(candidates, k):
            scan_findings.append(ScanFinding(scan_id=scan.id, vulnerability_id=vuln.id))

    return {
        "users": users,
        "assets": assets,
        "test_cases": test_cases,
        "test_runs": test_runs,
        "defects": defects,
        "scans": scans,
        "vulnerabilities": vulnerabilities,
        "scan_findings": scan_findings,
    }


def seed_database(engine: Engine, seed: int = 42, reset: bool = False) -> dict[str, int]:
    """
    Create the 8 tables and insert the generated rows. Returns row counts.
    With reset=True, drops existing tables first. Without it, refuses to
    seed a non-empty database so real data is never overwritten by accident.
    """
    if reset:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        existing = session.query(User).count()
        if existing and not reset:
            raise RuntimeError(
                "Database already contains data. Re-run with reset=True to replace it."
            )

    rows = generate(seed=seed)
    with Session(engine) as session:
        # Insert parents before children so foreign keys are satisfied.
        for key in [
            "users",
            "assets",
            "test_cases",
            "test_runs",
            "defects",
            "scans",
            "vulnerabilities",
            "scan_findings",
        ]:
            session.add_all(rows[key])
        session.commit()

    return {key: len(value) for key, value in rows.items()}


def _resolve_database_url(cli_url: str | None) -> str:
    """
    An explicit --url always wins. Otherwise fall back to DATABASE_URL from
    backend/.env — the same setting the running app uses — so
    `python -m app.db.seed_data --reset` seeds the database the server will
    actually read from, instead of needing the URL retyped every time.
    """
    return cli_url or get_settings().database_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the sample AppSec tracker database.")
    parser.add_argument(
        "--url",
        default=None,
        help="SQLAlchemy database URL (defaults to DATABASE_URL from backend/.env)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset", action="store_true", help="Drop and recreate tables first")
    args = parser.parse_args()

    url = _resolve_database_url(args.url)
    print(f"Seeding {url} ...")
    engine = create_engine(url)
    counts = seed_database(engine, seed=args.seed, reset=args.reset)
    for table, count in counts.items():
        print(f"{table}: {count}")


if __name__ == "__main__":
    main()
