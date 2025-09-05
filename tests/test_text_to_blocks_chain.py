from pathlib import Path

import pytest

import backend.core.logic.report_analysis.text_provider as tp
import backend.core.logic.report_analysis.block_exporter as be

SAMPLE_PDF = Path(__file__).parent / "fixtures" / "sample_block.pdf"


@pytest.fixture
def chdir_tmp(tmp_path, monkeypatch):
    """Run tests inside a temporary directory so traces/* are isolated."""
    monkeypatch.chdir(tmp_path)
    yield tmp_path


def test_extract_and_cache_text_creates_files(chdir_tmp, monkeypatch):
    monkeypatch.setattr(tp, "_extract_text_per_page", lambda p: ["hello world"])
    tp.extract_and_cache_text("sess1", SAMPLE_PDF, ocr_enabled=False)
    out_dir = Path("traces") / "texts" / "sess1"
    assert (out_dir / "full.txt").exists()
    assert (out_dir / "page_001.txt").exists()
    assert (out_dir / "meta.json").exists()


def test_export_account_blocks_with_cache(chdir_tmp, monkeypatch):
    monkeypatch.setattr(tp, "_extract_text_per_page", lambda p: ["Sample Bank\nAccount # 1234"])
    monkeypatch.setattr(
        be,
        "extract_account_blocks",
        lambda text: [["Sample Bank", "Account # 1234"]],
    )
    monkeypatch.setattr(be, "ENRICH_ENABLED", False)
    tp.extract_and_cache_text("sess2", SAMPLE_PDF, ocr_enabled=False)
    be.export_account_blocks("sess2", SAMPLE_PDF)
    out_dir = Path("traces") / "blocks" / "sess2"
    assert (out_dir / "_index.json").exists()
    assert (out_dir / "block_01.json").exists()


def test_fail_fast_without_cache(chdir_tmp):
    with pytest.raises(ValueError, match="no_cached_text_for_session"):
        be.export_account_blocks("sess3", SAMPLE_PDF)
