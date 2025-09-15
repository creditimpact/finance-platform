import json
from pathlib import Path

from backend.pipeline.runs import RUNS_ROOT_ENV
from scripts.split_accounts_from_tsv import main as split_accounts_main


def _write_tsv(path: Path) -> None:
    header = "page\tline\ty0\ty1\tx0\tx1\ttext\n"
    rows = [
        "1\t1\t10\t11\t60\t100\tTransUnion\n",
        "1\t1\t10\t11\t160\t200\tExperian\n",
        "1\t1\t10\t11\t260\t300\tEquifax\n",
        "1\t2\t20\t21\t0\t20\tAccount #\n",
        "1\t2\t20\t21\t60\t100\t208743***\n",
        "1\t2\t20\t21\t160\t200\t208743***\n",
        "1\t2\t20\t21\t260\t300\t208743***\n",
    ]
    path.write_text(header + "".join(rows), encoding="utf-8")


def test_split_accounts_registers_manifest(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))

    accounts_dir = tmp_path / "traces" / "blocks" / "sid123" / "accounts_table"
    accounts_dir.mkdir(parents=True)
    tsv_path = accounts_dir / "_debug_full.tsv"
    _write_tsv(tsv_path)

    general_json = accounts_dir / "general_info_from_full.json"
    general_json.write_text("{}", encoding="utf-8")
    accounts_json = accounts_dir / "accounts_from_full.json"

    split_accounts_main(["--full", str(tsv_path), "--json_out", str(accounts_json)])

    manifest_path = runs_root / "sid123" / "manifest.json"
    data = json.loads(manifest_path.read_text())
    assert data["base_dirs"]["traces_accounts_table"] == str(accounts_dir.resolve())
    art = data["artifacts"]["traces"]["accounts_table"]
    assert art["accounts_json"] == str(accounts_json.resolve())
    assert art["general_json"] == str(general_json.resolve())
    assert (accounts_dir / ".manifest").read_text() == str(manifest_path.resolve())
