import json
import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault(
    "requests", types.SimpleNamespace(post=lambda *args, **kwargs: None)
)

from backend.core.logic.consistency import compute_inconsistent_fields
from backend.core.logic.consistency import compute_field_consistency
from backend.core.logic.validation_requirements import (
    apply_validation_summary,
    build_summary_payload,
    build_validation_requirements,
    build_validation_requirements_for_account,
    sync_validation_tag,
)


def test_compute_inconsistent_fields_detects_money_and_text():
    bureaus = {
        "transunion": {"balance_owed": "$100.00", "account_status": "Open"},
        "experian": {"balance_owed": "100", "account_status": "open"},
        "equifax": {"balance_owed": "200", "account_status": "Closed"},
    }

    inconsistencies = compute_inconsistent_fields(bureaus)

    assert "balance_owed" in inconsistencies
    assert inconsistencies["balance_owed"]["normalized"]["equifax"] == 200.0
    assert inconsistencies["balance_owed"]["consensus"] == "majority"
    assert "account_status" in inconsistencies
    assert (
        inconsistencies["account_status"]["normalized"]["transunion"] == "open"
    )
    assert inconsistencies["account_status"]["disagreeing_bureaus"] == ["equifax"]


def test_build_validation_requirements_uses_config_and_defaults():
    bureaus = {
        "transunion": {"balance_owed": "100", "mystery_field": "abc"},
        "experian": {"balance_owed": "200", "mystery_field": "xyz"},
        "equifax": {"balance_owed": "200", "mystery_field": "xyz"},
    }

    requirements, inconsistencies, field_consistency = build_validation_requirements(bureaus)
    fields = [entry["field"] for entry in requirements]

    assert fields == ["balance_owed"]

    balance_rule = next(entry for entry in requirements if entry["field"] == "balance_owed")
    assert balance_rule["category"] == "activity"
    assert balance_rule["min_days"] == 8
    assert "monthly_statement" in balance_rule["documents"]
    assert balance_rule["strength"] == "strong"
    assert balance_rule["ai_needed"] is False
    assert balance_rule["bureaus"] == ["equifax", "experian", "transunion"]

    assert set(inconsistencies.keys()) == {"balance_owed", "mystery_field"}
    assert {"balance_owed", "mystery_field"}.issubset(field_consistency.keys())


def test_build_summary_payload_includes_field_consistency():
    requirements = [
        {
            "field": "balance_owed",
            "category": "activity",
            "min_days": 8,
            "documents": ["monthly_statement"],
            "strength": "strong",
            "ai_needed": False,
            "bureaus": ["experian", "transunion"],
        }
    ]
    field_consistency = {
        "balance_owed": {
            "consensus": "split",
            "normalized": {"transunion": 100.0, "experian": 150.0},
            "raw": {"transunion": "100", "experian": "150"},
            "disagreeing_bureaus": ["experian"],
        }
    }

    payload = build_summary_payload(
        requirements, field_consistency=field_consistency
    )

    assert payload["count"] == 1
    expected_consistency = {
        "balance_owed": {
            "consensus": "split",
            "normalized": {"transunion": 100.0, "experian": 150.0},
            "disagreeing_bureaus": ["experian"],
        }
    }
    assert payload["field_consistency"] == expected_consistency
    assert "requirements" not in payload

    assert len(payload["findings"]) == 1
    finding = payload["findings"][0]
    assert finding["reason_code"] == "C4_TWO_MATCH_ONE_DIFF"
    assert finding["reason_label"] == "two bureaus agree, one differs"
    assert finding["is_missing"] is False
    assert finding["is_mismatch"] is True
    assert finding["missing_count"] == 0
    assert finding["present_count"] == 2
    assert finding["distinct_values"] == 2
    assert finding["send_to_ai"] is False


def test_build_summary_payload_can_disable_reason_enrichment(monkeypatch):
    monkeypatch.setenv("VALIDATION_REASON_ENABLED", "0")

    requirements = [
        {
            "field": "balance_owed",
            "category": "activity",
            "min_days": 8,
            "documents": ["monthly_statement"],
            "strength": "strong",
            "ai_needed": False,
            "bureaus": ["experian", "transunion"],
        }
    ]
    field_consistency = {
        "balance_owed": {
            "consensus": "split",
            "normalized": {"transunion": 100.0, "experian": 150.0},
            "raw": {"transunion": "100", "experian": "150"},
            "disagreeing_bureaus": ["experian"],
        }
    }

    payload = build_summary_payload(
        requirements, field_consistency=field_consistency
    )

    assert payload["count"] == 1
    assert len(payload["findings"]) == 1
    assert payload["findings"][0]["field"] == "balance_owed"
    assert "reason_code" not in payload["findings"][0]
    # raw values should be preserved when enrichment is disabled
    balance_consistency = payload["field_consistency"]["balance_owed"]
    assert "raw" in balance_consistency
    assert balance_consistency["raw"]["transunion"] == "100"


@pytest.mark.parametrize(
    "field, values, expected_code, expected_ai",
    [
        (
            "balance_owed",
            {"experian": "200", "equifax": "200", "transunion": None},
            "C1_TWO_PRESENT_ONE_MISSING",
            False,
        ),
        (
            "balance_owed",
            {"experian": "open", "equifax": None, "transunion": "--"},
            "C2_ONE_MISSING",
            False,
        ),
        (
            "account_type",
            {"experian": "revolving", "equifax": "installment", "transunion": None},
            "C3_TWO_PRESENT_CONFLICT",
            True,
        ),
        (
            "account_rating",
            {"experian": "1", "equifax": "1", "transunion": "2"},
            "C4_TWO_MATCH_ONE_DIFF",
            True,
        ),
        (
            "creditor_remarks",
            {
                "experian": "interest only",
                "equifax": "account closed",
                "transunion": "charge-off",
            },
            "C5_ALL_DIFF",
            True,
        ),
        (
            "account_status",
            {"experian": None, "equifax": "", "transunion": "--"},
            "C6_ALL_MISSING",
            False,
        ),
    ],
)
def test_build_summary_payload_reason_codes_send_to_ai(
    field: str,
    values: dict[str, object],
    expected_code: str,
    expected_ai: bool,
) -> None:
    requirements = [
        {
            "field": field,
            "category": "status",
            "min_days": 0,
            "documents": ["stub"],
            "strength": "weak",
            "ai_needed": False,
        }
    ]

    payload = build_summary_payload(
        requirements,
        field_consistency={field: {"normalized": values}},
    )

    assert "findings" in payload
    assert payload["count"] == 1

    finding = payload["findings"][0]
    assert finding["field"] == field
    assert finding["reason_code"] == expected_code
    assert finding["send_to_ai"] is expected_ai


def test_compute_inconsistent_fields_handles_histories():
    bureaus = {
        "transunion": {"account_status": "Open"},
        "experian": {"account_status": "Open"},
        "equifax": {"account_status": "Open"},
        "two_year_payment_history": {
            "transunion": ["OK", "30", "60"],
            "experian": ["ok", "30", "60"],
            "equifax": ["OK", "60", "90"],
        },
        "seven_year_history": {
            "transunion": {"late30": 0, "late60": 0, "late90": 0},
            "experian": {"late30": 0, "late60": 0, "late90": 0},
            "equifax": {"late30": 1, "late60": 0, "late90": 0},
        },
    }

    inconsistencies = compute_inconsistent_fields(bureaus)

    assert "two_year_payment_history" in inconsistencies
    history_norm = inconsistencies["two_year_payment_history"]["normalized"]
    assert history_norm["transunion"]["tokens"] == ["OK", "30", "60"]
    assert history_norm["experian"]["tokens"] == ["OK", "30", "60"]
    assert history_norm["equifax"]["tokens"] == ["OK", "60", "90"]
    assert history_norm["equifax"]["counts"]["late90"] == 1

    assert "seven_year_history" in inconsistencies
    seven_norm = inconsistencies["seven_year_history"]["normalized"]
    assert seven_norm["transunion"]["late30"] == 0
    assert seven_norm["equifax"]["late30"] == 1


def test_compute_field_consistency_reads_history_from_branch():
    bureaus = {
        "transunion": {
            "two_year_payment_history": ["OK", "30", "OK"],
            "seven_year_history": {"late30": "1", "late60": 0},
        },
        "experian": {
            "two_year_payment_history": "ok,30,OK",
            "seven_year_history": {"late30": "0", "late60": 0},
        },
        "equifax": {
            "two_year_payment_history": [
                {"date": "2024-01", "status": "OK"},
                {"date": "2024-02", "status": "60"},
            ],
            "seven_year_history": "CO,CO,30",
        },
    }

    details = compute_field_consistency(bureaus)

    history = details["two_year_payment_history"]
    assert history["consensus"] in {"split", "majority"}
    assert history["normalized"]["experian"]["tokens"] == ["OK", "30", "OK"]
    assert history["normalized"]["equifax"]["counts"]["late60"] == 1

    seven_year = details["seven_year_history"]
    assert seven_year["consensus"] in {"split", "majority"}
    assert seven_year["normalized"]["transunion"]["late30"] == 1
    assert seven_year["normalized"]["equifax"]["late90"] == 2
    assert seven_year["normalized"]["equifax"]["late30"] == 1


def test_seven_year_history_canonicalizes_bucket_names():
    bureaus = {
        "transunion": {
            "seven_year_history": {"30 Days Late": "2", "Charge-Off Count": 1}
        },
        "experian": {"seven_year_history": {"late30": 2, "co_count": "1"}},
        "equifax": {
            "seven_year_history": {
                "past due 30": 3,
                "charge offs": 1,
            }
        },
    }

    details = compute_field_consistency(bureaus)

    history = details["seven_year_history"]
    assert history["normalized"]["transunion"]["late30"] == 2
    assert history["normalized"]["transunion"]["late90"] == 1
    assert history["normalized"]["experian"]["late30"] == 2
    assert history["normalized"]["experian"]["late90"] == 1
    assert history["normalized"]["equifax"]["late30"] == 3
    assert history["normalized"]["equifax"]["late90"] == 1
    assert history["consensus"] in {"majority", "split"}


def test_compute_field_consistency_handles_dates_and_account_numbers():
    bureaus = {
        "transunion": {
            "date_opened": "2023-05-01",
            "account_number_display": "****1234",
            "remarks": "Charge-Off filed",
        },
        "experian": {
            "date_opened": "5/1/2023",
            "account_number_display": "XXXX-1234",
            "remarks": "charge off filed!",
        },
        "equifax": {
            "date_opened": "28.7.2025",
            "account_number_display": "****5678",
            "remarks": "Different note",
        },
    }

    details = compute_field_consistency(bureaus)

    assert details["date_opened"]["normalized"]["experian"] == "2023-05-01"
    assert details["date_opened"]["normalized"]["equifax"] == "2025-07-28"
    assert details["account_number_display"]["consensus"] == "majority"
    assert details["account_number_display"]["disagreeing_bureaus"] == ["equifax"]
    assert details["account_number_display"]["normalized"]["transunion"]["last4"] == "1234"
    assert details["account_number_display"]["normalized"]["equifax"]["last4"] == "5678"
    assert details["remarks"]["normalized"]["transunion"] == "charge off filed"
    assert details["remarks"]["normalized"]["equifax"] == "different note"


def test_history_missing_vs_present_requires_strong_documents():
    bureaus = {
        "transunion": {},
        "experian": {},
        "equifax": {},
        "two_year_payment_history": {
            "transunion": ["OK", "30", "OK"],
            "experian": None,
            "equifax": None,
        },
        "seven_year_history": {
            "transunion": {"late30": 2, "late60": 0, "late90": 0},
            "experian": None,
            "equifax": {"late30": 0, "late60": 0, "late90": 0},
        },
    }

    requirements, inconsistencies, _ = build_validation_requirements(bureaus)
    fields = {entry["field"]: entry for entry in requirements}

    assert "two_year_payment_history" in fields
    two_year_req = fields["two_year_payment_history"]
    assert two_year_req["category"] == "history"
    assert two_year_req["min_days"] == 18
    assert two_year_req["documents"] == [
        "monthly_statements_2y",
        "internal_payment_history",
    ]
    assert two_year_req["strength"] == "soft"
    assert two_year_req["ai_needed"] is True
    assert two_year_req["bureaus"] == ["equifax", "experian", "transunion"]
    assert (
        inconsistencies["two_year_payment_history"]["consensus"] != "unanimous"
    )

    assert "seven_year_history" in fields
    seven_req = fields["seven_year_history"]
    assert seven_req["category"] == "history"
    assert seven_req["min_days"] == 25
    assert seven_req["documents"] == [
        "cra_report_7y",
        "cra_audit_logs",
        "collection_history",
    ]
    assert seven_req["strength"] == "soft"
    assert seven_req["ai_needed"] is True
    assert seven_req["bureaus"] == ["equifax", "experian", "transunion"]
    assert inconsistencies["seven_year_history"]["consensus"] != "unanimous"


def test_build_validation_requirements_for_account_respects_summary_consensus(
    tmp_path: Path,
) -> None:
    account_dir = tmp_path / "acct"
    account_dir.mkdir()

    bureaus = {
        "transunion": {},
        "experian": {},
        "equifax": {},
        "two_year_payment_history": {
            "transunion": ["OK", "30", "OK"],
            "experian": None,
            "equifax": ["CO", "CO"],
        },
        "seven_year_history": {
            "transunion": {"late30": 2, "late60": 0, "late90": 0},
            "experian": None,
            "equifax": {"late30": 0, "late60": 0, "late90": 1},
        },
    }

    bureaus_path = account_dir / "bureaus.json"
    bureaus_path.write_text(json.dumps(bureaus), encoding="utf-8")

    _, _, field_consistency = build_validation_requirements(bureaus)
    for field in ("two_year_payment_history", "seven_year_history"):
        snapshot = field_consistency.get(field)
        if isinstance(snapshot, dict):
            snapshot["consensus"] = "unanimous"

    summary_path = account_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"field_consistency": field_consistency}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = build_validation_requirements_for_account(account_dir)

    assert result["status"] == "ok"
    assert result["count"] == 0
    assert result["fields"] == []

    summary_after = json.loads(summary_path.read_text(encoding="utf-8"))
    validation_block = summary_after["validation_requirements"]
    assert validation_block["count"] == 0
    assert validation_block["findings"] == []
    assert "requirements" not in validation_block
    assert (
        validation_block["field_consistency"]["two_year_payment_history"]["consensus"]
        == "unanimous"
    )
    assert (
        validation_block["field_consistency"]["seven_year_history"]["consensus"]
        == "unanimous"
    )


def test_two_year_history_free_text_requires_ai() -> None:
    bureaus = {
        "transunion": {},
        "experian": {},
        "equifax": {},
        "two_year_payment_history": {
            "transunion": ["OK", "OK"],
            "experian": "SEE REMARKS",
            "equifax": ["OK", "OK"],
        },
    }

    requirements, _, _ = build_validation_requirements(bureaus)
    fields = {entry["field"]: entry for entry in requirements}

    assert "two_year_payment_history" in fields
    history_req = fields["two_year_payment_history"]
    assert history_req["strength"] == "soft"
    assert history_req["ai_needed"] is True


def test_two_year_history_partial_months_requires_ai() -> None:
    bureaus = {
        "transunion": {},
        "experian": {},
        "equifax": {},
        "two_year_payment_history": {
            "transunion": ["OK"] * 24,
            "experian": ["OK"] * 6,
            "equifax": ["OK"] * 12,
        },
    }

    requirements, _, _ = build_validation_requirements(bureaus)
    fields = {entry["field"]: entry for entry in requirements}

    assert "two_year_payment_history" in fields
    history_req = fields["two_year_payment_history"]
    assert history_req["strength"] == "soft"
    assert history_req["ai_needed"] is True


def test_two_year_history_delinquency_remains_strong() -> None:
    bureaus = {
        "transunion": {},
        "experian": {},
        "equifax": {},
        "two_year_payment_history": {
            "transunion": ["OK"] * 23 + ["30"],
            "experian": ["OK"] * 24,
            "equifax": ["OK"] * 24,
        },
    }

    requirements, _, _ = build_validation_requirements(bureaus)
    fields = {entry["field"]: entry for entry in requirements}

    assert "two_year_payment_history" in fields
    history_req = fields["two_year_payment_history"]
    assert history_req["strength"] == "strong"
    assert history_req["ai_needed"] is False


def test_apply_validation_summary_and_sync_validation_tag(tmp_path):
    summary_path = tmp_path / "summary.json"
    tag_path = tmp_path / "tags.json"

    summary_path.write_text(json.dumps({"existing": True}), encoding="utf-8")
    tag_payload = [
        {"kind": "other", "value": 1},
        {"kind": "validation_required", "fields": ["old"], "at": "2024-01-01T00:00:00Z"},
    ]
    tag_path.write_text(json.dumps(tag_payload), encoding="utf-8")

    requirements = [
        {
            "field": "balance_owed",
            "category": "activity",
            "min_days": 8,
            "documents": [],
            "strength": "strong",
            "ai_needed": False,
        }
    ]
    payload = build_summary_payload(requirements)

    apply_validation_summary(summary_path, payload)
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    validation_block = summary_data["validation_requirements"]
    assert validation_block["count"] == 1
    assert summary_data["existing"] is True
    assert "requirements" not in validation_block
    assert len(validation_block["findings"]) == 1
    assert validation_block["findings"][0]["field"] == "balance_owed"

    sync_validation_tag(tag_path, ["balance_owed"], emit=True)
    tags = json.loads(tag_path.read_text(encoding="utf-8"))
    validation_tags = [tag for tag in tags if tag.get("kind") == "validation_required"]
    assert len(validation_tags) == 1
    assert validation_tags[0]["fields"] == ["balance_owed"]
    assert validation_tags[0]["at"].endswith("Z")
    other_tags = [tag for tag in tags if tag.get("kind") == "other"]
    assert other_tags == [{"kind": "other", "value": 1}]

    empty_payload = build_summary_payload([])
    apply_validation_summary(summary_path, empty_payload)
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "validation_requirements" not in summary_data

    sync_validation_tag(tag_path, [], emit=True)
    tags = json.loads(tag_path.read_text(encoding="utf-8"))
    assert all(tag.get("kind") != "validation_required" for tag in tags)


def test_build_validation_requirements_for_account_writes_summary_and_tags(
    tmp_path, monkeypatch
):
    account_dir = tmp_path / "0"
    account_dir.mkdir()

    bureaus = {
        "transunion": {"balance_owed": "100", "payment_status": "late"},
        "experian": {"balance_owed": "150", "payment_status": "ok"},
        "equifax": {"balance_owed": "200", "payment_status": "late"},
    }
    (account_dir / "bureaus.json").write_text(json.dumps(bureaus), encoding="utf-8")

    field_consistency = compute_field_consistency(bureaus)
    existing_summary = {"existing": True, "field_consistency": field_consistency}
    (account_dir / "summary.json").write_text(
        json.dumps(existing_summary), encoding="utf-8"
    )

    existing_tags = [
        {"kind": "other", "value": 1},
        {"kind": "validation_required", "fields": ["old"], "at": "2024"},
    ]
    (account_dir / "tags.json").write_text(json.dumps(existing_tags), encoding="utf-8")

    monkeypatch.setenv("WRITE_VALIDATION_TAGS", "1")

    result = build_validation_requirements_for_account(account_dir)

    assert result["status"] == "ok"
    assert result["count"] == 2
    assert set(result["fields"]) == {"balance_owed", "payment_status"}
    validation_payload = result["validation_requirements"]
    assert validation_payload["count"] == 2
    assert "requirements" not in validation_payload
    assert {entry["field"] for entry in validation_payload["findings"]} == {
        "balance_owed",
        "payment_status",
    }
    for entry in validation_payload["findings"]:
        assert entry["reason_code"].startswith("C")
        assert entry["bureaus"] == ["equifax", "experian", "transunion"]

    summary = json.loads((account_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["existing"] is True
    validation_block = summary["validation_requirements"]
    assert validation_block["count"] == 2
    assert "requirements" not in validation_block
    assert {entry["field"] for entry in validation_block["findings"]} == {
        "balance_owed",
        "payment_status",
    }
    for entry in validation_block["findings"]:
        assert entry["reason_code"].startswith("C")
        assert entry["bureaus"] == ["equifax", "experian", "transunion"]
    field_consistency = validation_block["field_consistency"]
    assert {"balance_owed", "payment_status"}.issubset(field_consistency.keys())
    assert field_consistency["balance_owed"]["consensus"] in {"majority", "split"}
    assert field_consistency["payment_status"]["disagreeing_bureaus"]

    tags = json.loads((account_dir / "tags.json").read_text(encoding="utf-8"))
    assert {tag["kind"] for tag in tags} == {"other", "validation_required"}
    validation_tag = next(tag for tag in tags if tag["kind"] == "validation_required")
    assert validation_tag["fields"] == ["balance_owed", "payment_status"]


def test_build_validation_requirements_for_account_clears_when_empty(tmp_path, monkeypatch):
    account_dir = tmp_path / "1"
    account_dir.mkdir()

    consistent = {
        "transunion": {"balance_owed": "100"},
        "experian": {"balance_owed": "100"},
        "equifax": {"balance_owed": "100"},
    }
    (account_dir / "bureaus.json").write_text(json.dumps(consistent), encoding="utf-8")

    seed_summary = {
        "field_consistency": compute_field_consistency(consistent),
        "validation_requirements": {
            "count": 1,
            "findings": [
                {
                    "field": "balance_owed",
                    "category": "activity",
                    "min_days": 8,
                    "documents": [],
                    "strength": "strong",
                    "ai_needed": False,
                }
            ],
        },
    }
    (account_dir / "summary.json").write_text(
        json.dumps(seed_summary), encoding="utf-8"
    )

    (account_dir / "tags.json").write_text(
        json.dumps(
            [
                {"kind": "validation_required", "fields": ["balance_owed"], "at": "old"},
                {"kind": "other", "value": 1},
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("WRITE_VALIDATION_TAGS", raising=False)

    result = build_validation_requirements_for_account(account_dir)

    assert result["status"] == "ok"
    assert result["count"] == 0
    assert result["fields"] == []
    validation_payload = result["validation_requirements"]
    assert validation_payload["count"] == 0
    assert "requirements" not in validation_payload
    assert validation_payload["findings"] == []
    assert "field_consistency" in validation_payload
    field_consistency = validation_payload["field_consistency"]
    assert "balance_owed" in field_consistency
    assert field_consistency["balance_owed"]["consensus"] == "unanimous"

    summary = json.loads((account_dir / "summary.json").read_text(encoding="utf-8"))
    validation_block = summary["validation_requirements"]
    assert validation_block["count"] == 0
    assert "requirements" not in validation_block
    assert validation_block["findings"] == []
    assert "balance_owed" in validation_block["field_consistency"]

    tags = json.loads((account_dir / "tags.json").read_text(encoding="utf-8"))
    assert all(tag.get("kind") != "validation_required" for tag in tags)


def _write_basic_bureaus(account_dir: Path) -> None:
    bureaus = {
        "transunion": {"balance_owed": "100"},
        "experian": {"balance_owed": "150"},
        "equifax": {"balance_owed": "200"},
    }
    (account_dir / "bureaus.json").write_text(
        json.dumps(bureaus, ensure_ascii=False), encoding="utf-8"
    )


def test_validation_debug_excluded_when_flag_off(tmp_path, monkeypatch):
    account_dir = tmp_path / "A1"
    account_dir.mkdir()
    _write_basic_bureaus(account_dir)
    (account_dir / "tags.json").write_text("[]", encoding="utf-8")
    monkeypatch.delenv("VALIDATION_DEBUG", raising=False)

    result = build_validation_requirements_for_account(account_dir)
    assert result["status"] == "ok"

    summary = json.loads((account_dir / "summary.json").read_text(encoding="utf-8"))
    assert "validation_debug" not in summary


def test_validation_debug_included_when_flag_on(tmp_path, monkeypatch):
    account_dir = tmp_path / "A2"
    account_dir.mkdir()
    _write_basic_bureaus(account_dir)
    (account_dir / "tags.json").write_text("[]", encoding="utf-8")
    monkeypatch.setenv("VALIDATION_DEBUG", "1")

    result = build_validation_requirements_for_account(account_dir)
    assert result["status"] == "ok"

    summary = json.loads((account_dir / "summary.json").read_text(encoding="utf-8"))
    assert "validation_debug" in summary
