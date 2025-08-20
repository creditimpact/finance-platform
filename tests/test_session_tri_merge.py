from backend.api.session_manager import get_session, update_session


def test_tri_merge_evidence_preserved(tmp_path, monkeypatch):
    sess_file = tmp_path / "sessions.json"
    monkeypatch.setattr(
        "backend.api.session_manager.SESSION_FILE", sess_file
    )

    session_id = "sess1"
    update_session(session_id, tri_merge={"evidence": {"snap1": {"a": 1}}})
    update_session(session_id, tri_merge={"evidence": {"snap2": {"b": 2}}})

    session = get_session(session_id)
    assert session["tri_merge"]["evidence"]["snap1"] == {"a": 1}
    assert session["tri_merge"]["evidence"]["snap2"] == {"b": 2}
