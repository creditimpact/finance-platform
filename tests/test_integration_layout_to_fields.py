import json
from pathlib import Path

import pytest

import backend.core.logic.report_analysis.block_exporter as be
from tests.fixtures.tokens_synthetic import three_bureau_tokens


@pytest.fixture
def chdir_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path


def _full_text_one_block():
    # Minimal tri-header + a couple of label lines to create an account block
    return (
        "Sample Issuer\n"
        "TransUnion Experian Equifax\n"
        "Account # 1111 2222 3333\n"
        "High Balance: $1,000 $2,000 $3,000\n"
        "Account Type: Closed Paid Conventional\n"
    )


def test_layout_to_fields_end_to_end(chdir_tmp, monkeypatch):
    # Monkeypatch cached text + layout to avoid heavy dependencies
    monkeypatch.setattr(
        be, "load_cached_text", lambda sid: {"pages": [_full_text_one_block()], "full_text": _full_text_one_block()}
    )

    def fake_layout(pdf_path: str):
        return {"pages": [{"number": 1, "width": 700, "height": 900, "tokens": three_bureau_tokens()}], "full_text": _full_text_one_block()}

    monkeypatch.setattr(be, "load_text_with_layout", fake_layout)

    # Create a dummy PDF file
    sample_pdf = chdir_tmp / "dummy.pdf"
    sample_pdf.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    be.export_account_blocks("tp6sid", sample_pdf)

    out_dir = Path("traces") / "blocks" / "tp6sid"
    block_files = sorted(out_dir.glob("block_*.json"))
    assert block_files
    data = json.loads(block_files[0].read_text(encoding="utf-8"))
    fields = data["fields"]

    # Verify required bureaus and fields exist
    bureaus = ("transunion", "experian", "equifax")
    for b in bureaus:
        assert b in fields
        sub = fields[b]
        assert sub.get("account_number_display") is not None
        assert sub.get("high_balance") is not None
        assert sub.get("account_type") is not None

    # Values from synthetic tokens should be mapped
    assert fields["transunion"]["account_number_display"].endswith("1111")
    assert fields["experian"]["account_number_display"].endswith("2222")
    assert fields["equifax"]["account_number_display"].endswith("3333")
    # Money cleaned (no commas/$)
    assert fields["transunion"]["high_balance"] == "1000"
    assert fields["experian"]["high_balance"] == "2000"
    assert fields["equifax"]["high_balance"] == "3000"
    # Multiline join for EQ account_type
    assert fields["equifax"]["account_type"].startswith("Conventional real estate mortgage")

