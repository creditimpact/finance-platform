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

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    tsv_path = source_dir / "_debug_full.tsv"
    _write_tsv(tsv_path)

    manifest_path = runs_root / "sid123" / "manifest.json"

    split_accounts_main(
        [
            "--full",
            str(tsv_path),
            "--sid",
            "sid123",
            "--manifest",
            str(manifest_path),
        ]
    )

    data = json.loads(manifest_path.read_text())
    accounts_dir = runs_root / "sid123" / "traces" / "accounts_table"
    assert data["base_dirs"]["traces_accounts_table"] == str(
        accounts_dir.resolve()
    )
    art = data["artifacts"]["traces"]["accounts_table"]
    accounts_json = accounts_dir / "accounts_from_full.json"
    general_json = accounts_dir / "general_info_from_full.json"
    debug_full = accounts_dir / "_debug_full.tsv"
    per_account_dir = accounts_dir / "per_account_tsv"
    assert art["accounts_json"] == str(accounts_json.resolve())
    assert art["general_json"] == str(general_json.resolve())
    assert art["debug_full_tsv"] == str(debug_full.resolve())
    expected_per_acc = accounts_dir / "per_account_tsv"
    assert art["per_account_tsv_dir"] == str(expected_per_acc.resolve())
    assert (accounts_dir / ".manifest").read_text() == str(manifest_path.resolve())
    assert general_json.exists()
    assert per_account_dir.exists()
