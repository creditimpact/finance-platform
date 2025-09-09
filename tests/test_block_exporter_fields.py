from pathlib import Path
import json

import pytest

import backend.core.logic.report_analysis.block_exporter as be


@pytest.fixture
def chdir_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path


def _sample_text():
    return (
        "Sample Bank\n"
        "TransUnion Experian Equifax\n"
        "Account # 1234 5678 9012\n"
        "High Balance: $1,000 $2,000 $3,000\n"
        "Account Status: Closed Closed Closed\n"
    )


def test_all_bureau_keys_exist_even_when_empty(chdir_tmp, monkeypatch):
    # Patch cached text and layout to avoid heavy deps
    monkeypatch.setattr(
        be, "load_cached_text", lambda sid: {"pages": ["p1"], "full_text": _sample_text()}
    )

    def fake_layout(pdf_path: str):
        return {"pages": [] , "full_text": ""}
    monkeypatch.setattr(be, "load_text_with_layout", fake_layout)

    sample_pdf = chdir_tmp / "sample.pdf"
    sample_pdf.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    be.export_account_blocks("tp3sid", sample_pdf)

    out_dir = Path("traces") / "blocks" / "tp3sid"
    # Find the first block file
    block_files = sorted(out_dir.glob("block_*.json"))
    assert block_files, "no blocks exported"
    data = json.loads(block_files[0].read_text(encoding="utf-8"))
    fields = data.get("fields") or {}
    # Ensure three bureaus exist
    assert set(fields.keys()) >= {"transunion", "experian", "equifax"}
    # Ensure each bureau has the full set of mapped keys (at least 22)
    for bureau in ("transunion", "experian", "equifax"):
        sub = fields[bureau]
        assert isinstance(sub, dict)
        # Must contain canonical mapping keys
        required = {
            "account_number_display",
            "high_balance",
            "last_verified",
            "date_of_last_activity",
            "date_reported",
            "date_opened",
            "balance_owed",
            "closed_date",
            "account_rating",
            "account_description",
            "dispute_status",
            "creditor_type",
            "account_status",
            "payment_status",
            "creditor_remarks",
            "payment_amount",
            "last_payment",
            "term_length",
            "past_due_amount",
            "account_type",
            "payment_frequency",
            "credit_limit",
        }
        assert required.issubset(sub.keys())
        assert len(sub.keys()) >= len(required)
