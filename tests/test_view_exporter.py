import sqlite3

from backend.analytics.view_exporter import (
    ExportFilters,
    fetch_joined,
    stream_csv,
    stream_json,
)


def setup_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE tri_merge (session_id TEXT, family_id TEXT, action_tag TEXT)")
    conn.execute(
        """
        CREATE TABLE planner (
            session_id TEXT,
            family_id TEXT,
            action_tag TEXT,
            cycle_id INTEGER,
            status TEXT,
            last_sent_at TEXT,
            next_eligible_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE outcome (
            session_id TEXT,
            family_id TEXT,
            action_tag TEXT,
            cycle_id INTEGER,
            outcome TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO tri_merge VALUES (?,?,?)",
        [("s1", "f1", "t1"), ("s2", "f2", "t2")],
    )
    conn.executemany(
        "INSERT INTO planner VALUES (?,?,?,?,?,?,?)",
        [
            ("s1", "f1", "t1", 1, "planned", "2023-01-01", "2023-02-01"),
            ("s2", "f2", "t2", 2, "planned", "2023-03-01", "2023-04-01"),
        ],
    )
    conn.executemany(
        "INSERT INTO outcome VALUES (?,?,?,?,?)",
        [
            ("s1", "f1", "t1", 1, "ok"),
            ("s2", "f2", "t2", 2, "fail"),
        ],
    )
    conn.execute(
        """
        CREATE VIEW analytics_planner_outcomes AS
        SELECT tm.session_id, tm.family_id, tm.action_tag,
               p.cycle_id, o.outcome, p.status AS planner_status,
               p.last_sent_at, p.next_eligible_at
        FROM tri_merge tm
        JOIN planner p ON tm.session_id=p.session_id AND tm.family_id=p.family_id AND tm.action_tag=p.action_tag
        LEFT JOIN outcome o ON (
            tm.session_id=o.session_id AND tm.family_id=o.family_id AND tm.action_tag=o.action_tag AND p.cycle_id=o.cycle_id
        )
        """
    )
    return conn


def test_fetch_joined_and_streaming() -> None:
    conn = setup_db()
    filters = ExportFilters(
        action_tags=["t1"],
        cycle_range=(1, 1),
        start_ts="2023-01-01",
        end_ts="2023-02-01",
    )
    rows = list(fetch_joined(conn, filters))
    assert rows == [
        {
            "session_id": "s1",
            "family_id": "f1",
            "action_tag": "t1",
            "cycle_id": 1,
            "outcome": "ok",
            "planner_status": "planned",
            "last_sent_at": "2023-01-01",
            "next_eligible_at": "2023-02-01",
        }
    ]

    csv_text = "".join(stream_csv(iter(rows)))
    lines = csv_text.strip().splitlines()
    assert lines[0] == "export_version=1"
    assert lines[1].startswith("session_id")
    assert lines[2].startswith("s1,f1,t1,1,ok,planned,2023-01-01,2023-02-01")

    json_text = "".join(stream_json(iter(rows)))
    assert json_text.startswith("{\"export_version\":1")
    assert "\"session_id\":\"s1\"" in json_text
