from backend.api.session_manager import populate_from_history, update_session


def test_backfill_fields_across_cycles(tmp_path, monkeypatch):
    sess_file = tmp_path / "sessions.json"
    monkeypatch.setattr("backend.api.session_manager.SESSION_FILE", sess_file)

    session_id = "s1"
    update_session(
        session_id,
        tri_merge={"evidence": {"fam1": {"a": 1}}},
        outcome_history={"acct1": [{"outcome": "Deleted"}]},
    )

    fresh = {"session_id": session_id}
    populate_from_history(fresh)
    assert fresh["tri_merge"]["evidence"] == {"fam1": {"a": 1}}
    assert fresh["outcome_history"] == {"acct1": [{"outcome": "Deleted"}]}
    assert fresh["_provenance"] == {
        "tri_merge.evidence": "history",
        "outcome_history": "history",
    }

    newer = {"session_id": session_id}
    populate_from_history(newer)
    assert newer["tri_merge"] == fresh["tri_merge"]
    assert newer["outcome_history"] == fresh["outcome_history"]
    assert newer["_provenance"] == {
        "tri_merge.evidence": "history",
        "outcome_history": "history",
    }
