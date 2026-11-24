"""
Tests for Step 1: synthetic data generator and models.

Covers: volumes match the plan, foreign keys are satisfied, generation is
deterministic for a given seed, the non-reset guard protects existing data,
and the Step 0 introspector sees all 8 tables in a seeded database.
"""

import pytest
from sqlalchemy import create_engine, text

from app.config import get_settings
from app.core.schema_introspector import get_schema_snapshot
from app.db.models import Base
from app.db.seed_data import SeedVolumes, _resolve_database_url, generate, seed_database

EXPECTED_TABLES = {
    "users", "assets", "test_cases", "test_runs", "defects",
    "scans", "vulnerabilities", "scan_findings",
}


@pytest.fixture
def seeded_engine():
    engine = create_engine("sqlite:///:memory:")
    seed_database(engine, seed=42)
    return engine


def test_default_volumes_meet_plan():
    rows = generate(seed=42)
    assert len(rows["assets"]) == 20
    assert len(rows["test_runs"]) >= 500
    assert len(rows["vulnerabilities"]) >= 200


def test_same_seed_produces_identical_data():
    a = generate(seed=7)
    b = generate(seed=7)
    assert [(v.cve_id, v.severity, v.cvss_score) for v in a["vulnerabilities"]] == [
        (v.cve_id, v.severity, v.cvss_score) for v in b["vulnerabilities"]
    ]
    assert [(r.status, r.build_version) for r in a["test_runs"]] == [
        (r.status, r.build_version) for r in b["test_runs"]
    ]


def test_different_seed_produces_different_data():
    a = generate(seed=1)
    b = generate(seed=2)
    assert [v.cvss_score for v in a["vulnerabilities"]] != [
        v.cvss_score for v in b["vulnerabilities"]
    ]


def test_only_failed_or_blocked_runs_have_defects():
    rows = generate(seed=42)
    status_by_run = {r.id: r.status for r in rows["test_runs"]}
    for defect in rows["defects"]:
        assert status_by_run[defect.test_run_id] != "pass"


def test_cvss_matches_severity_band():
    rows = generate(seed=42)
    bands = {"critical": (9.0, 10.0), "high": (7.0, 8.9), "medium": (4.0, 6.9), "low": (0.1, 3.9)}
    for vuln in rows["vulnerabilities"]:
        lo, hi = bands[vuln.severity]
        assert lo <= vuln.cvss_score <= hi


def test_fixed_vulns_have_fixed_date_and_others_do_not():
    rows = generate(seed=42)
    for vuln in rows["vulnerabilities"]:
        if vuln.status == "fixed":
            assert vuln.fixed_date is not None
            assert vuln.fixed_date >= vuln.discovered_date
        else:
            assert vuln.fixed_date is None


def test_foreign_keys_are_satisfied_after_seeding(seeded_engine):
    with seeded_engine.connect() as conn:
        # PRAGMA foreign_key_check returns one row per violation; empty means clean.
        violations = conn.execute(text("PRAGMA foreign_key_check")).fetchall()
    assert violations == []


def test_seeded_row_counts_match_generator(seeded_engine):
    with seeded_engine.connect() as conn:
        counts = {
            t: conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
            for t in EXPECTED_TABLES
        }
    assert counts["assets"] == 20
    assert counts["test_runs"] >= 500
    assert counts["vulnerabilities"] >= 200


def test_refuses_to_seed_non_empty_database_without_reset(seeded_engine):
    with pytest.raises(RuntimeError, match="already contains data"):
        seed_database(seeded_engine, seed=42, reset=False)


def test_reset_replaces_existing_data(seeded_engine):
    counts = seed_database(seeded_engine, seed=99, reset=True)
    assert counts["assets"] == 20


def test_introspector_sees_all_eight_tables(seeded_engine):
    """Ties Step 1 back to Step 0: the agent should discover this schema unaided."""
    snapshot = get_schema_snapshot(seeded_engine)
    assert set(snapshot.table_names()) == EXPECTED_TABLES
    scan_findings = snapshot.tables["scan_findings"]
    fk_targets = {fk.references_table for fk in scan_findings.foreign_keys}
    assert fk_targets == {"scans", "vulnerabilities"}


def test_models_metadata_matches_expected_tables():
    assert set(Base.metadata.tables.keys()) == EXPECTED_TABLES


def test_custom_volumes_are_respected():
    small = SeedVolumes(users=3, assets=4, test_cases_per_asset=1, test_runs=20,
                        scans=5, vulnerabilities=10)
    rows = generate(seed=42, volumes=small)
    assert len(rows["users"]) == 3
    assert len(rows["assets"]) == 4
    assert len(rows["test_runs"]) == 20
    assert len(rows["vulnerabilities"]) == 10
