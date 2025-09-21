import json
from pathlib import Path

from backend.pipeline import auto_ai
from backend.pipeline.runs import RunManifest


def test_maybe_run_auto_ai_pipeline_runs_full_flow(monkeypatch, tmp_path):
    sid = "SID123"
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("ENABLE_AUTO_AI_PIPELINE", "1")

    manifest = RunManifest.for_sid(sid)
    run_dir = manifest.path.parent
    accounts_dir = run_dir / "cases" / "accounts"
    (accounts_dir / "1").mkdir(parents=True, exist_ok=True)
    (accounts_dir / "2").mkdir(parents=True, exist_ok=True)

    recorded: dict[str, object] = {}

    class DummyScore:
        def __init__(self, indices):
            self.indices = indices

    def fake_score_accounts(
        sid_arg: str,
        *,
        runs_root: Path,
        write_tags: bool,
        only_ai_rows: bool = False,
    ):
        recorded["score"] = {
            "sid": sid_arg,
            "runs_root": Path(runs_root),
            "write_tags": write_tags,
            "only_ai_rows": only_ai_rows,
        }
        return DummyScore(indices=[1])

    def fake_build(argv):
        recorded["build"] = list(argv)
        packs_dir = run_dir / "ai_packs"
        packs_dir.mkdir(parents=True, exist_ok=True)
        index_payload = [{"a": 1, "b": 2, "file": "001-002.json"}]
        (packs_dir / "index.json").write_text(
            json.dumps(index_payload), encoding="utf-8"
        )

    def fake_send(argv):
        recorded["send"] = list(argv)

    compact_calls: list[Path] = []

    def fake_compact(account_dir: Path):
        compact_calls.append(Path(account_dir))

    monkeypatch.setattr(auto_ai, "score_accounts", fake_score_accounts)
    monkeypatch.setattr(auto_ai, "build_ai_merge_packs_main", fake_build)
    monkeypatch.setattr(auto_ai, "send_ai_merge_packs_main", fake_send)
    monkeypatch.setattr(auto_ai, "compact_tags_for_account", fake_compact)

    auto_ai.maybe_run_auto_ai_pipeline(sid, summary={"cases": {"dir": str(accounts_dir)}})

    assert recorded["score"] == {
        "sid": sid,
        "runs_root": run_dir.parent,
        "write_tags": True,
        "only_ai_rows": False,
    }
    assert recorded["build"] == ["--sid", sid, "--runs-root", str(run_dir.parent)]
    assert recorded["send"] == ["--sid", sid]
    assert sorted(path.name for path in compact_calls) == ["1", "2"]


def test_maybe_run_auto_ai_pipeline_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("ENABLE_AUTO_AI_PIPELINE", raising=False)

    called: list[object] = []

    def fake_runner(*args, **kwargs):
        called.append((args, kwargs))

    monkeypatch.setattr(auto_ai, "_run_auto_ai_pipeline", fake_runner)

    auto_ai.maybe_run_auto_ai_pipeline("SID", summary=None)

    assert not called
