from pathlib import Path

import pytest

import backend.core.logic.report_analysis.block_exporter as be

SAMPLE_PDF = Path(__file__).parent / "fixtures" / "sample_block.pdf"


@pytest.fixture
def chdir_tmp(tmp_path, monkeypatch):
    """Run tests inside a temporary directory so traces/* are isolated."""
    monkeypatch.chdir(tmp_path)
    yield tmp_path


def _sample_text():
    return (
        "Sample Bank\n"
        "TransUnion Experian Equifax\n"
        "Account # 1234 1234 1234\n"
        "Equifax 30:0 60:0 90:0\n"
    )


def test_export_writes_files(chdir_tmp, monkeypatch):
    monkeypatch.setattr(be, "extract_text_from_pdf", lambda _p: _sample_text())

    be.export_account_blocks("sess1", SAMPLE_PDF)

    out_dir = Path("traces") / "blocks" / "sess1"
    assert (out_dir / "_index.json").exists()
    assert (out_dir / "block_01.json").exists()


def test_load_account_blocks_reads_back(chdir_tmp, monkeypatch):
    monkeypatch.setattr(be, "extract_text_from_pdf", lambda _p: _sample_text())
    be.export_account_blocks("sess2", SAMPLE_PDF)

    blocks = be.load_account_blocks("sess2")
    assert isinstance(blocks, list)
    assert blocks and isinstance(blocks[0], dict)
    first = blocks[0]
    assert first["heading"] == "Sample Bank"
    assert first["lines"][0] == "Sample Bank"


def test_fail_fast_on_empty(chdir_tmp, monkeypatch):
    monkeypatch.setattr(be, "extract_text_from_pdf", lambda _p: "")
    empty_pdf = chdir_tmp / "empty.pdf"
    empty_pdf.write_bytes(b"")
    with pytest.raises(ValueError, match="No blocks extracted"):
        be.export_account_blocks("sess3", empty_pdf)
