import json
from pathlib import Path

import logging
import pytest

import backend.core.logic.report_analysis.block_exporter as be

SAMPLE_PDF = Path(__file__).parent / "fixtures" / "sample_block.pdf"


@pytest.fixture
def chdir_tmp(tmp_path, monkeypatch):
    """Run tests inside a temporary directory so traces/* are isolated."""
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.fixture
def stub_layout(monkeypatch):
    layout = {
        "pages": [
            {
                "width": 612,
                "height": 792,
                "tokens": [
                    {
                        "line": 1,
                        "x0": 0,
                        "x1": 10,
                        "y0": 10,
                        "y1": 20,
                        "text": "Sample",
                    },
                    {"line": 1, "x0": 11, "x1": 20, "y0": 10, "y1": 20, "text": "Bank"},
                    {
                        "line": 2,
                        "x0": 0,
                        "x1": 20,
                        "y0": 30,
                        "y1": 40,
                        "text": "TransUnion",
                    },
                    {
                        "line": 2,
                        "x0": 21,
                        "x1": 40,
                        "y0": 30,
                        "y1": 40,
                        "text": "Experian",
                    },
                    {
                        "line": 2,
                        "x0": 41,
                        "x1": 60,
                        "y0": 30,
                        "y1": 40,
                        "text": "Equifax",
                    },
                    {
                        "line": 3,
                        "x0": 0,
                        "x1": 10,
                        "y0": 50,
                        "y1": 60,
                        "text": "Account",
                    },
                    {"line": 3, "x0": 11, "x1": 15, "y0": 50, "y1": 60, "text": "#"},
                    {"line": 3, "x0": 16, "x1": 20, "y0": 50, "y1": 60, "text": "1234"},
                    {"line": 3, "x0": 21, "x1": 25, "y0": 50, "y1": 60, "text": "1234"},
                    {"line": 3, "x0": 26, "x1": 30, "y0": 50, "y1": 60, "text": "1234"},
                    {
                        "line": 4,
                        "x0": 0,
                        "x1": 20,
                        "y0": 70,
                        "y1": 80,
                        "text": "Equifax",
                    },
                    {"line": 4, "x0": 21, "x1": 40, "y0": 70, "y1": 80, "text": "30:0"},
                    {"line": 4, "x0": 41, "x1": 60, "y0": 70, "y1": 80, "text": "60:0"},
                    {"line": 4, "x0": 61, "x1": 80, "y0": 70, "y1": 80, "text": "90:0"},
                ],
            }
        ]
    }
    monkeypatch.setattr(be, "load_text_with_layout", lambda _p: layout)
    return layout


def _sample_text():
    return (
        "Sample Bank\n"
        "TransUnion Experian Equifax\n"
        "Account # 1234 1234 1234\n"
        "Equifax 30:0 60:0 90:0\n"
    )


def test_export_writes_files(chdir_tmp, monkeypatch, stub_layout, caplog):
    monkeypatch.setattr(
        be, "load_cached_text", lambda sid: {"full_text": _sample_text()}
    )
    caplog.set_level(logging.INFO)
    _blocks, meta = be.export_account_blocks("sess1", SAMPLE_PDF)

    out_dir = Path("traces") / "blocks" / "sess1"
    assert (out_dir / "_index.json").exists()
    assert (out_dir / "block_01.json").exists()

    accounts_dir = out_dir / "accounts_table"
    assert (accounts_dir / "_debug_full.tsv").exists()
    json_path = accounts_dir / "accounts_from_full.json"
    assert json_path.exists()
    assert json_path.parent == accounts_dir
    data = json.loads(json_path.read_text(encoding="utf-8"))
    accounts = data["accounts"]
    assert isinstance(accounts, list) and len(accounts) == 1
    assert data["stop_marker_seen"] is False
    assert (accounts_dir / "per_account_tsv" / "_debug_account_1.tsv").exists()

    # Ensure index tracks artifacts and no stray copies exist
    idx_path = accounts_dir / "_table_index.json"
    assert idx_path.exists()
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    extras = idx.get("extras", [])
    paths = {e.get("type"): e.get("path") for e in extras}
    assert paths.get("full_tsv") == str(accounts_dir / "_debug_full.tsv")
    assert paths.get("accounts_from_full") == str(json_path)

    # Metadata dict returned from export_account_blocks
    assert meta["full_tsv"] == str(accounts_dir / "_debug_full.tsv")
    assert meta["accounts_json"] == str(json_path)

    # Logs should contain explicit paths
    assert f"Stage-A: wrote full TSV: {accounts_dir / '_debug_full.tsv'}" in caplog.text
    assert f"Stage-A: wrote accounts JSON: {json_path}" in caplog.text
    assert (
        "Stage-A: accounts summary: total=1 collections=0 stop_marker_seen=False"
        in caplog.text
    )

    assert not Path("_debug_full.tsv").exists()
    assert not Path("accounts_from_full.json").exists()


def test_rerun_overwrites_files(chdir_tmp, monkeypatch, stub_layout):
    monkeypatch.setattr(
        be, "load_cached_text", lambda sid: {"full_text": _sample_text()}
    )
    be.export_account_blocks("sess1", SAMPLE_PDF)
    be.export_account_blocks("sess1", SAMPLE_PDF)
    accounts_dir = Path("traces") / "blocks" / "sess1" / "accounts_table"
    # Ensure only one TSV/JSON exists with stable names
    assert len(list(accounts_dir.glob("_debug_full*.tsv"))) == 1
    assert len(list(accounts_dir.glob("accounts_from_full*.json"))) == 1


def test_accounts_table_index_tracks_enriched_json(chdir_tmp, monkeypatch, stub_layout):
    monkeypatch.setattr(
        be, "load_cached_text", lambda sid: {"full_text": _sample_text()}
    )

    orig_split = be.split_accounts_from_tsv

    def _stub_split(full_tsv, json_out, write_tsv=True):
        result = orig_split(full_tsv, json_out, write_tsv)
        enriched = json_out.with_name(json_out.stem + ".enriched.json")
        enriched.write_text(json_out.read_text(encoding="utf-8"), encoding="utf-8")
        return result

    monkeypatch.setattr(be, "split_accounts_from_tsv", _stub_split)

    be.export_account_blocks("sess_enr", SAMPLE_PDF)

    accounts_dir = Path("traces") / "blocks" / "sess_enr" / "accounts_table"
    idx_path = accounts_dir / "_table_index.json"
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    extras = idx.get("extras", [])
    paths = {e.get("type"): e.get("path") for e in extras}
    enriched_path = accounts_dir / "accounts_from_full.enriched.json"
    assert paths.get("accounts_from_full_enriched") == str(enriched_path)

def test_load_account_blocks_reads_back(chdir_tmp, monkeypatch, stub_layout):
    monkeypatch.setattr(
        be, "load_cached_text", lambda sid: {"full_text": _sample_text()}
    )
    be.export_account_blocks("sess2", SAMPLE_PDF)

    blocks = be.load_account_blocks("sess2")
    assert isinstance(blocks, list)
    assert blocks and isinstance(blocks[0], dict)
    first = blocks[0]
    assert first["heading"] == "Sample Bank"
    assert first["lines"][0] == "Sample Bank"


def test_fail_fast_on_empty(chdir_tmp, monkeypatch, stub_layout):
    monkeypatch.setattr(be, "load_cached_text", lambda sid: {"full_text": ""})
    empty_pdf = chdir_tmp / "empty.pdf"
    empty_pdf.write_bytes(b"")
    with pytest.raises(ValueError, match="No blocks extracted"):
        be.export_account_blocks("sess3", empty_pdf)
