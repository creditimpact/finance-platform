from backend.pipeline.runs import RUNS_ROOT_ENV, RunManifest
from scripts.run_manifest import main


def test_run_manifest_cli_basic(tmp_path, monkeypatch, capsys):
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))
    sid = "cli-test"

    sample_dir = tmp_path / "data"
    sample_dir.mkdir()
    sample_file = sample_dir / "foo.txt"
    sample_file.write_text("hi", encoding="utf-8")

    main(["set-base-dir", sid, "data_dir", str(sample_dir)])
    main(["set-artifact", sid, "data", "foo", str(sample_file)])
    main(["get", sid, "data", "foo"])
    out = capsys.readouterr().out.strip()
    assert out == str(sample_file.resolve())

    m = RunManifest.for_sid(sid)
    assert m.data["base_dirs"]["data_dir"] == str(sample_dir.resolve())
    assert m.get("data", "foo") == str(sample_file.resolve())
