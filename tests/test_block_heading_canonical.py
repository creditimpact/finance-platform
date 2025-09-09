from pathlib import Path

import json
import backend.core.logic.report_analysis.block_exporter as be


def test_normalize_heading_variants():
    cases = ["BK OF AMER", "BANKAMERICA", "BofA"]
    out = [be.normalize_heading(c) for c in cases]
    for d in out:
        assert d["canonical"] == "Bank of America"
        assert d["slug"] == "bank-of-america"


def test_tail_digits():
    lines = [
        "Account # ****1234",
        "AMERICAN EXPRESS",
    ]
    tail = be.tail_digits_from_lines(lines)
    assert tail == "1234"


def _sample_text_with_presence():
    return (
        "Sample Bank\n"
        "TransUnion Experian Equifax\n"
        "Account # ****1234 ****5678 ****9012\n"
        "Payment Status: Current  Delinquent  \n"
        "Credit Limit: $1,000 $   $\n"
    )


def test_meta_presence_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(
        be, "load_cached_text", lambda sid: {"full_text": _sample_text_with_presence()}
    )
    sample_pdf = tmp_path / "s.pdf"
    sample_pdf.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    be.export_account_blocks("sess_meta", sample_pdf)
    out_dir = Path("traces") / "blocks" / "sess_meta"
    block_path = out_dir / "block_01.json"
    data = json.loads(block_path.read_text(encoding="utf-8"))
    meta = data.get("meta", {})
    assert meta.get("issuer_canonical")
    assert meta.get("issuer_slug")
    presence = meta.get("bureau_presence", {})
    assert isinstance(presence, dict)
    # TU should be present from Payment Status / Credit Limit
    assert presence.get("transunion") is True


def test_index_unchanged_format(tmp_path, monkeypatch):
    monkeypatch.setattr(
        be, "load_cached_text", lambda sid: {"full_text": _sample_text_with_presence()}
    )
    sample_pdf = tmp_path / "s2.pdf"
    sample_pdf.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    be.export_account_blocks("sess_idx", sample_pdf)
    out_dir = Path("traces") / "blocks" / "sess_idx"
    idx = json.loads((out_dir / "_index.json").read_text(encoding="utf-8"))
    for row in idx:
        assert set(row.keys()) == {"i", "heading", "file"}

