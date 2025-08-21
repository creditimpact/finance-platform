import sqlite3

from backend.analytics import views


def setup_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE tri_merge (
            session_id TEXT,
            account_id TEXT,
            family_id TEXT,
            cycle_id INTEGER,
            tri_merge_snapshot_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE planner (
            session_id TEXT,
            account_id TEXT,
            family_id TEXT,
            cycle_id INTEGER,
            tri_merge_snapshot_id TEXT,
            plan_id TEXT,
            step_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE outcome (
            session_id TEXT,
            account_id TEXT,
            family_id TEXT,
            cycle_id INTEGER,
            tri_merge_snapshot_id TEXT,
            plan_id TEXT,
            step_id TEXT,
            outcome_id TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO tri_merge VALUES (?,?,?,?,?)",
        ("s1", "a1", "f1", 1, "snap1"),
    )
    conn.execute(
        "INSERT INTO planner VALUES (?,?,?,?,?,?,?)",
        ("s1", "a1", "f1", 1, "snap1", "plan1", "step1"),
    )
    conn.execute(
        "INSERT INTO outcome VALUES (?,?,?,?,?,?,?,?)",
        ("s1", "a1", "f1", 1, "snap1", "plan1", "step1", "out1"),
    )
    views.create_views(conn)
    return conn


def test_tri_merge_view() -> None:
    conn = setup_db()
    row = conn.execute("SELECT * FROM analytics_tri_merge_view").fetchone()
    assert row == ("s1", "a1", "f1", 1, "snap1", None, None, None)


def test_planner_view() -> None:
    conn = setup_db()
    row = conn.execute("SELECT * FROM analytics_planner_view").fetchone()
    assert row == ("s1", "a1", "f1", 1, "snap1", "plan1", "step1", None)


def test_outcome_view() -> None:
    conn = setup_db()
    row = conn.execute("SELECT * FROM analytics_outcome_view").fetchone()
    assert row == ("s1", "a1", "f1", 1, "snap1", "plan1", "step1", "out1")
