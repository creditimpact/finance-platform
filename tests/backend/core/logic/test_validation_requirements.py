import json
import logging
import sys
import types
from pathlib import Path
from typing import Any, Mapping

import pytest

sys.modules.setdefault(
    "requests", types.SimpleNamespace(post=lambda *args, **kwargs: None)
)

if "backend.ai.validation_builder" not in sys.modules:
    stub_builder = types.ModuleType("backend.ai.validation_builder")

    def _noop_build_validation_pack_for_account(*args: Any, **kwargs: Any) -> list[str]:
        return []

    stub_builder.build_validation_pack_for_account = _noop_build_validation_pack_for_account  # type: ignore[attr-defined]
    sys.modules["backend.ai.validation_builder"] = stub_builder

if "backend.validation.decision_matrix" not in sys.modules:
    stub_decision_matrix = types.ModuleType("backend.validation.decision_matrix")

    def _noop_decide_default(*args: Any, **kwargs: Any) -> str:
        return ""

    stub_decision_matrix.decide_default = _noop_decide_default  # type: ignore[attr-defined]
    sys.modules["backend.validation.decision_matrix"] = stub_decision_matrix

from backend.core.logic.consistency import compute_inconsistent_fields
from backend.core.logic.consistency import compute_field_consistency
from backend.core.logic.validation_requirements import (
    _raw_value_provider_for_account_factory,
    _should_redact_pii,
    _clear_tolerance_state,
    apply_validation_summary,
    build_findings,
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


def test_finding_embeds_bureau_values_amount():
    bureaus = {
        "transunion": {"credit_limit": "$5,000"},
        "experian": {"credit_limit": "5000"},
        "equifax": {"credit_limit": 5000},
    }

    requirements = [
        {
            "field": "credit_limit",
            "category": "limits",
            "min_days": 5,
            "documents": ["statement"],
            "strength": "strong",
            "ai_needed": False,
            "bureaus": ["equifax", "experian", "transunion"],
        }
    ]

    field_consistency = compute_field_consistency(dict(bureaus))
    payload = build_summary_payload(
        requirements,
        field_consistency=field_consistency,
        raw_value_provider=_raw_value_provider_for_account_factory(bureaus),
    )

    finding = payload["findings"][0]
    bureau_values = finding["bureau_values"]

    assert bureau_values["equifax"] == {
        "present": True,
        "raw": 5000,
        "normalized": 5000.0,
    }
    assert bureau_values["experian"] == {
        "present": True,
        "raw": "5000",
        "normalized": 5000.0,
    }
    assert bureau_values["transunion"] == {
        "present": True,
        "raw": "$5,000",
        "normalized": 5000.0,
    }


def test_finding_embeds_missing_bureau():
    bureaus = {
        "transunion": {"date_opened": "01/02/2023"},
        "experian": {"date_opened": "--"},
        "equifax": {"date_opened": "2023-01-05"},
    }

    requirements = [
        {
            "field": "date_opened",
            "category": "dates",
            "min_days": 10,
            "documents": [],
            "strength": "medium",
            "ai_needed": False,
            "bureaus": ["equifax", "experian", "transunion"],
        }
    ]

    field_consistency = compute_field_consistency(dict(bureaus))
    payload = build_summary_payload(
        requirements,
        field_consistency=field_consistency,
        raw_value_provider=_raw_value_provider_for_account_factory(bureaus),
    )

    finding = payload["findings"][0]
    bureau_values = finding["bureau_values"]

    assert bureau_values["transunion"] == {
        "present": True,
        "raw": "01/02/2023",
        "normalized": "2023-01-02",
    }
    assert bureau_values["equifax"] == {
        "present": True,
        "raw": "2023-01-05",
        "normalized": "2023-01-05",
    }
    assert bureau_values["experian"] == {
        "present": False,
        "raw": None,
        "normalized": None,
    }


def test_finding_embeds_histories_shape():
    bureaus = {
        "transunion": {"two_year_payment_history": ["OK", "30", "OK", "60"]},
        "experian": {"two_year_payment_history": "OK,30,OK"},
        "equifax": {},
        "two_year_payment_history": {
            "transunion": ["OK", "30", "OK", "60"],
            "experian": "OK,30,OK",
        },
    }

    requirements = [
        {
            "field": "two_year_payment_history",
            "category": "history",
            "min_days": 12,
            "documents": ["payment_history"],
            "strength": "medium",
            "ai_needed": True,
            "bureaus": ["equifax", "experian", "transunion"],
        }
    ]

    field_consistency = compute_field_consistency(dict(bureaus))
    payload = build_summary_payload(
        requirements,
        field_consistency=field_consistency,
        raw_value_provider=_raw_value_provider_for_account_factory(bureaus),
    )

    finding = payload["findings"][0]
    bureau_values = finding["bureau_values"]

    transunion_normalized = bureau_values["transunion"]["normalized"]
    experian_normalized = bureau_values["experian"]["normalized"]

    assert isinstance(transunion_normalized, dict)
    assert set(transunion_normalized.keys()) == {"tokens", "counts"}
    assert isinstance(transunion_normalized["tokens"], list)
    assert isinstance(transunion_normalized["counts"], dict)
    assert isinstance(experian_normalized, dict)
    assert set(experian_normalized.keys()) == {"tokens", "counts"}
    assert bureau_values["equifax"] == {
        "present": False,
        "raw": None,
        "normalized": None,
    }

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

    assert payload["schema_version"] == 3
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

    bureau_values = finding["bureau_values"]
    assert set(bureau_values.keys()) == {"equifax", "experian", "transunion"}
    assert bureau_values["transunion"] == {
        "present": True,
        "raw": "100",
        "normalized": 100.0,
    }
    assert bureau_values["experian"] == {
        "present": True,
        "raw": "150",
        "normalized": 150.0,
    }
    assert bureau_values["equifax"] == {
        "present": False,
        "raw": None,
        "normalized": None,
    }


def test_build_summary_payload_can_optionally_include_legacy_requirements(monkeypatch):
    monkeypatch.setenv("VALIDATION_SUMMARY_INCLUDE_REQUIREMENTS", "1")

    requirements = [
        {
            "field": "balance_owed",
            "category": "activity",
            "min_days": 8,
            "documents": ["monthly_statement"],
            "strength": "strong",
            "ai_needed": False,
            "bureaus": ["experian", "transunion"],
        },
        {
            "field": "account_status",
            "category": "status",
            "min_days": 10,
            "documents": ["account_statement"],
            "strength": "soft",
            "ai_needed": True,
            "bureaus": ["experian", "equifax"],
        },
    ]

    payload = build_summary_payload(requirements)

    assert payload["schema_version"] == 3
    assert payload["findings"]
    assert payload["requirements"] == [
        {
            "field": "balance_owed",
            "category": "activity",
            "min_days": 8,
            "documents": ["monthly_statement"],
            "strength": "strong",
            "ai_needed": False,
            "bureaus": ["experian", "transunion"],
        },
        {
            "field": "account_status",
            "category": "status",
            "min_days": 10,
            "documents": ["account_statement"],
            "strength": "soft",
            "ai_needed": True,
            "bureaus": ["experian", "equifax"],
        },
    ]


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

    assert payload["schema_version"] == 3
    assert len(payload["findings"]) == 1
    assert payload["findings"][0]["field"] == "balance_owed"
    assert "reason_code" not in payload["findings"][0]
    # raw values should be preserved when enrichment is disabled
    balance_consistency = payload["field_consistency"]["balance_owed"]
    assert "raw" in balance_consistency
    assert balance_consistency["raw"]["transunion"] == "100"

    bureau_values = payload["findings"][0]["bureau_values"]
    assert bureau_values["transunion"] == {
        "present": True,
        "raw": "100",
        "normalized": 100.0,
    }
    assert bureau_values["experian"] == {
        "present": True,
        "raw": "150",
        "normalized": 150.0,
    }
    assert bureau_values["equifax"] == {
        "present": False,
        "raw": None,
        "normalized": None,
    }


def test_bureau_values_use_raw_provider_when_raw_missing():
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
            "disagreeing_bureaus": ["experian"],
        }
    }

    bureaus = {
        "transunion": {"balance_owed": "$100.00"},
        "experian": {"balance_owed": "150"},
        "equifax": {},
    }

    raw_provider = _raw_value_provider_for_account_factory(bureaus)

    payload = build_summary_payload(
        requirements,
        field_consistency=field_consistency,
        raw_value_provider=raw_provider,
    )

    finding = payload["findings"][0]
    bureau_values = finding["bureau_values"]

    assert bureau_values["transunion"] == {
        "present": True,
        "raw": "$100.00",
        "normalized": 100.0,
    }
    assert bureau_values["experian"] == {
        "present": True,
        "raw": "150",
        "normalized": 150.0,
    }
    assert bureau_values["equifax"] == {
        "present": False,
        "raw": None,
        "normalized": None,
    }


def test_bureau_values_use_raw_provider_for_history_blocks():
    requirements = [
        {
            "field": "two_year_payment_history",
            "category": "history",
            "min_days": 12,
            "documents": ["payment_history"],
            "strength": "soft",
            "ai_needed": True,
            "bureaus": ["equifax", "experian", "transunion"],
        }
    ]

    bureaus = {
        "transunion": {},
        "experian": {},
        "equifax": {},
        "two_year_payment_history": {
            "transunion": ["OK", "30", "OK"],
            "experian": "ok,30,CO",
            "equifax": [],
        },
    }

    full_consistency = compute_field_consistency(dict(bureaus))
    history_details = dict(full_consistency["two_year_payment_history"])
    history_details.pop("raw", None)

    field_consistency = {"two_year_payment_history": history_details}

    raw_provider = _raw_value_provider_for_account_factory(bureaus)

    payload = build_summary_payload(
        requirements,
        field_consistency=field_consistency,
        raw_value_provider=raw_provider,
    )

    finding = payload["findings"][0]
    bureau_values = finding["bureau_values"]
    normalized = history_details["normalized"]

    assert bureau_values["transunion"] == {
        "present": True,
        "raw": ["OK", "30", "OK"],
        "normalized": normalized["transunion"],
    }
    assert bureau_values["experian"] == {
        "present": True,
        "raw": "ok,30,CO",
        "normalized": normalized["experian"],
    }
    assert bureau_values["equifax"] == {
        "present": False,
        "raw": None,
        "normalized": None,
    }


def test_account_number_raw_can_be_redacted(monkeypatch):
    _should_redact_pii.cache_clear()
    monkeypatch.setenv("VALIDATION_REDACT_PII", "1")

    requirements = [
        {
            "field": "account_number_display",
            "category": "identity",
            "min_days": 5,
            "documents": [],
            "strength": "medium",
            "ai_needed": False,
            "bureaus": ["experian", "transunion"],
        }
    ]

    bureaus = {
        "transunion": {"account_number_display": "1234 5678 9012 3456"},
        "experian": {"account_number_display": "****-5678"},
        "equifax": {},
    }

    field_consistency = compute_field_consistency(dict(bureaus))

    payload = build_summary_payload(
        requirements,
        field_consistency=field_consistency,
        raw_value_provider=_raw_value_provider_for_account_factory(bureaus),
    )

    finding = payload["findings"][0]
    bureau_values = finding["bureau_values"]

    assert bureau_values["transunion"] == {
        "present": True,
        "raw": "**** **** **** 3456",
        "normalized": field_consistency["account_number_display"]["normalized"][
            "transunion"
        ],
    }
    assert bureau_values["experian"] == {
        "present": True,
        "raw": "****-5678",
        "normalized": field_consistency["account_number_display"]["normalized"][
            "experian"
        ],
    }
    assert bureau_values["equifax"] == {
        "present": False,
        "raw": None,
        "normalized": None,
    }

    _should_redact_pii.cache_clear()


def _make_requirement(field: str) -> dict[str, Any]:
    return {
        "field": field,
        "category": "test",
        "min_days": 0,
        "documents": [],
        "strength": "soft",
        "ai_needed": False,
        "bureaus": ["equifax", "experian", "transunion"],
    }


def test_only_semantic_fields_send_to_ai(monkeypatch):
    monkeypatch.delenv("VALIDATION_REASON_ENABLED", raising=False)

    requirements = [
        _make_requirement("account_type"),
        _make_requirement("creditor_type"),
        _make_requirement("account_rating"),
        _make_requirement("balance_owed"),
        _make_requirement("creditor_remarks"),
    ]

    field_consistency = {
        "account_type": {
            "normalized": {
                "experian": "credit_card",
                "equifax": "installment",
                "transunion": "credit_card",
            },
            "consensus": "split",
        },
        "creditor_type": {
            "normalized": {
                "experian": "bank",
                "equifax": "collection_agency",
                "transunion": None,
            },
            "consensus": "split",
        },
        "account_rating": {
            "normalized": {
                "experian": "positive",
                "equifax": "neutral",
                "transunion": "negative",
            },
            "consensus": "split",
        },
        "balance_owed": {
            "normalized": {
                "experian": 100.0,
                "equifax": 200.0,
                "transunion": 300.0,
            },
            "consensus": "split",
        },
        "creditor_remarks": {
            "normalized": {
                "experian": "updated remarks",
                "equifax": "other remarks",
                "transunion": "updated remarks",
            },
            "consensus": "split",
        },
    }

    findings = build_findings(requirements, field_consistency=field_consistency)
    findings_by_field = {finding["field"]: finding for finding in findings}

    assert findings_by_field["account_type"]["send_to_ai"] is True
    assert findings_by_field["creditor_type"]["send_to_ai"] is True
    assert findings_by_field["account_rating"]["send_to_ai"] is True
    assert findings_by_field["balance_owed"]["send_to_ai"] is False
    assert findings_by_field["creditor_remarks"]["send_to_ai"] is False


def test_semantic_missing_values_do_not_trigger_ai(monkeypatch):
    monkeypatch.delenv("VALIDATION_REASON_ENABLED", raising=False)

    requirements = [_make_requirement("account_type")]
    field_consistency = {
        "account_type": {
            "normalized": {
                "experian": None,
                "equifax": None,
                "transunion": "credit_card",
            },
            "consensus": "majority",
        }
    }

    findings = build_findings(requirements, field_consistency=field_consistency)
    finding = findings[0]

    assert finding["reason_code"] == "C2_ONE_MISSING"
    assert finding["send_to_ai"] is False


def test_empty_history_generates_missing_reason(monkeypatch):
    monkeypatch.delenv("VALIDATION_REASON_ENABLED", raising=False)

    bureaus = {
        "transunion": {"two_year_payment_history": {}},
        "experian": {"two_year_payment_history": ["30", "CO"]},
        "equifax": {"two_year_payment_history": []},
    }

    requirements, inconsistencies, field_consistency = build_validation_requirements(
        bureaus
    )

    assert "two_year_payment_history" in inconsistencies

    findings = build_findings(requirements, field_consistency=field_consistency)
    history_finding = next(
        item for item in findings if item["field"] == "two_year_payment_history"
    )

    assert history_finding["reason_code"] == "C2_ONE_MISSING"
    assert history_finding["is_missing"] is True
    assert history_finding["is_mismatch"] is False
    assert history_finding["send_to_ai"] is False


def test_build_summary_payload_applies_amount_tolerance(monkeypatch, tmp_path):
    from backend.core.logic import consistency as consistency_mod

    monkeypatch.setattr(consistency_mod, "_AMOUNT_TOL_ABS", 0.0, raising=False)
    monkeypatch.setattr(consistency_mod, "_AMOUNT_TOL_RATIO", 0.0, raising=False)

    bureaus = {
        "transunion": {"balance_owed": "110"},
        "experian": {"balance_owed": "100"},
        "equifax": {"balance_owed": "102"},
    }

    requirements = [
        {
            "field": "balance_owed",
            "category": "activity",
            "min_days": 8,
            "documents": ["monthly_statement"],
            "strength": "strong",
            "ai_needed": False,
            "bureaus": ["experian", "equifax", "transunion"],
        }
    ]

    field_consistency = compute_field_consistency(dict(bureaus))
    payload = build_summary_payload(
        requirements,
        field_consistency=field_consistency,
        raw_value_provider=_raw_value_provider_for_account_factory(bureaus),
        sid="SID123",
        runs_root=tmp_path,
    )

    assert payload["findings"] == []
    assert payload.get("tolerance", {}).get("suppressed_fields") == ["balance_owed"]
    assert "tolerance_notes" not in payload


def test_build_summary_payload_applies_date_tolerance(monkeypatch, tmp_path):
    from backend.core.logic import consistency as consistency_mod

    monkeypatch.setattr(consistency_mod, "_DATE_TOLERANCE_DAYS", 0, raising=False)
    _clear_tolerance_state()

    sid = "SIDDATE"
    run_dir = tmp_path / sid
    convention_path = run_dir / "trace" / "date_convention.json"
    convention_path.parent.mkdir(parents=True, exist_ok=True)
    convention_path.write_text("{\"convention\": \"MDY\"}", encoding="utf-8")

    bureaus = {
        "transunion": {"date_opened": "01/02/2023"},
        "experian": {"date_opened": "01/03/2023"},
        "equifax": {"date_opened": "01/04/2023"},
    }

    requirements = [
        {
            "field": "date_opened",
            "category": "dates",
            "min_days": 10,
            "documents": [],
            "strength": "medium",
            "ai_needed": False,
            "bureaus": ["equifax", "experian", "transunion"],
        }
    ]

    field_consistency = compute_field_consistency(dict(bureaus))
    payload = build_summary_payload(
        requirements,
        field_consistency=field_consistency,
        raw_value_provider=_raw_value_provider_for_account_factory(bureaus),
        sid=sid,
        runs_root=tmp_path,
    )

    assert payload["findings"] == []
    assert payload.get("tolerance", {}).get("suppressed_fields") == ["date_opened"]
    assert "tolerance_notes" not in payload
    _clear_tolerance_state()


def test_build_summary_payload_applies_last_verified_tolerance(monkeypatch, tmp_path):
    from backend.core.logic import consistency as consistency_mod

    monkeypatch.setattr(consistency_mod, "_DATE_TOLERANCE_DAYS", 0, raising=False)
    _clear_tolerance_state()

    sid = "SIDLASTVERIFIED"
    run_dir = tmp_path / sid
    convention_path = run_dir / "trace" / "date_convention.json"
    convention_path.parent.mkdir(parents=True, exist_ok=True)
    convention_path.write_text("{\"convention\": \"MDY\"}", encoding="utf-8")

    bureaus = {
        "transunion": {"last_verified": "2024-03-01"},
        "experian": {"last_verified": "03/04/2024"},
        "equifax": {"last_verified": "2024-03-02"},
    }

    requirements = [
        {
            "field": "last_verified",
            "category": "dates",
            "min_days": 10,
            "documents": [],
            "strength": "medium",
            "ai_needed": False,
            "bureaus": ["equifax", "experian", "transunion"],
        }
    ]

    field_consistency = compute_field_consistency(dict(bureaus))
    payload = build_summary_payload(
        requirements,
        field_consistency=field_consistency,
        raw_value_provider=_raw_value_provider_for_account_factory(bureaus),
        sid=sid,
        runs_root=tmp_path,
    )

    assert payload["findings"] == []
    assert payload.get("tolerance", {}).get("suppressed_fields") == ["last_verified"]
    assert "tolerance_notes" not in payload
    _clear_tolerance_state()


def test_build_summary_payload_flags_last_verified_outside_tolerance(monkeypatch, tmp_path):
    from backend.core.logic import consistency as consistency_mod

    monkeypatch.setattr(consistency_mod, "_DATE_TOLERANCE_DAYS", 0, raising=False)
    _clear_tolerance_state()

    sid = "SIDLASTVERIFIEDMISS"
    run_dir = tmp_path / sid
    convention_path = run_dir / "trace" / "date_convention.json"
    convention_path.parent.mkdir(parents=True, exist_ok=True)
    convention_path.write_text("{\"convention\": \"MDY\"}", encoding="utf-8")

    bureaus = {
        "transunion": {"last_verified": "2024-03-01"},
        "experian": {"last_verified": "03/09/2024"},
        "equifax": {"last_verified": "2024-03-01"},
    }

    requirements = [
        {
            "field": "last_verified",
            "category": "dates",
            "min_days": 10,
            "documents": [],
            "strength": "medium",
            "ai_needed": False,
            "bureaus": ["equifax", "experian", "transunion"],
        }
    ]

    field_consistency = compute_field_consistency(dict(bureaus))
    payload = build_summary_payload(
        requirements,
        field_consistency=field_consistency,
        raw_value_provider=_raw_value_provider_for_account_factory(bureaus),
        sid=sid,
        runs_root=tmp_path,
    )

    assert payload.get("tolerance", {}) == {}
    assert len(payload["findings"]) == 1
    finding = payload["findings"][0]
    assert finding["field"] == "last_verified"
    assert finding["reason_code"] == "C4_TWO_MATCH_ONE_DIFF"
    _clear_tolerance_state()


def _write_date_convention(runs_root: Path, sid: str, convention: str = "MDY") -> None:
    trace_dir = runs_root / sid / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "date_convention.json").write_text(
        json.dumps({"convention": convention}),
        encoding="utf-8",
    )


def test_integration_last_verified_within_tolerance_has_no_finding(tmp_path: Path) -> None:
    _clear_tolerance_state()
    try:
        sid = "SID_LV_INTEGRATION_WITHIN"
        _write_date_convention(tmp_path, sid)

        bureaus = {
            "transunion": {"last_verified": "2024-03-01"},
            "experian": {"last_verified": "03/04/2024"},
            "equifax": {"last_verified": "2024-03-04"},
        }

        requirements, _, field_consistency = build_validation_requirements(bureaus)
        assert requirements == []

        payload = build_summary_payload(
            requirements,
            field_consistency=field_consistency,
            raw_value_provider=_raw_value_provider_for_account_factory(bureaus),
            sid=sid,
            runs_root=tmp_path,
        )

        assert payload["findings"] == []
        assert payload.get("tolerance", {}) == {}
    finally:
        _clear_tolerance_state()


def test_integration_last_verified_outside_tolerance_raises_finding(tmp_path: Path) -> None:
    _clear_tolerance_state()
    try:
        sid = "SID_LV_INTEGRATION_MISMATCH"
        _write_date_convention(tmp_path, sid)

        bureaus = {
            "transunion": {"last_verified": "2024-03-01"},
            "experian": {"last_verified": "03/09/2024"},
            "equifax": {"last_verified": "2024-03-01"},
        }

        requirements, _, field_consistency = build_validation_requirements(bureaus)
        assert [entry["field"] for entry in requirements] == ["last_verified"]

        payload = build_summary_payload(
            requirements,
            field_consistency=field_consistency,
            raw_value_provider=_raw_value_provider_for_account_factory(bureaus),
            sid=sid,
            runs_root=tmp_path,
        )

        assert len(payload["findings"]) == 1
        finding = payload["findings"][0]
        assert finding["field"] == "last_verified"
        assert finding["reason_code"] == "C4_TWO_MATCH_ONE_DIFF"
    finally:
        _clear_tolerance_state()


def test_integration_account_rating_synonyms_produce_no_finding(tmp_path: Path) -> None:
    _clear_tolerance_state()
    try:
        sid = "SID_RATING_INTEGRATION"
        _write_date_convention(tmp_path, sid)

        bureaus = {
            "transunion": {"account_rating": "Paid as agreed"},
            "experian": {"account_rating": "current"},
            "equifax": {"account_rating": "PAID ON TIME"},
        }

        requirements, _, field_consistency = build_validation_requirements(bureaus)
        assert requirements == []

        payload = build_summary_payload(
            requirements,
            field_consistency=field_consistency,
            raw_value_provider=_raw_value_provider_for_account_factory(bureaus),
            sid=sid,
            runs_root=tmp_path,
        )

        assert payload["findings"] == []
    finally:
        _clear_tolerance_state()


def test_build_summary_payload_records_tolerance_notes_when_debug(monkeypatch, tmp_path):
    from backend.core.logic import consistency as consistency_mod

    monkeypatch.setattr(consistency_mod, "_AMOUNT_TOL_ABS", 0.0, raising=False)
    monkeypatch.setattr(consistency_mod, "_AMOUNT_TOL_RATIO", 0.0, raising=False)
    monkeypatch.setenv("VALIDATION_DEBUG", "1")

    bureaus = {
        "transunion": {"balance_owed": "110"},
        "experian": {"balance_owed": "100"},
        "equifax": {"balance_owed": "102"},
    }

    requirements = [
        {
            "field": "balance_owed",
            "category": "activity",
            "min_days": 8,
            "documents": ["monthly_statement"],
            "strength": "strong",
            "ai_needed": False,
            "bureaus": ["experian", "equifax", "transunion"],
        }
    ]

    field_consistency = compute_field_consistency(dict(bureaus))
    payload = build_summary_payload(
        requirements,
        field_consistency=field_consistency,
        raw_value_provider=_raw_value_provider_for_account_factory(bureaus),
        sid="SID123",
        runs_root=tmp_path,
    )

    notes = payload.get("tolerance_notes")
    assert isinstance(notes, list) and len(notes) == 1
    assert notes[0]["field"] == "balance_owed"
    assert notes[0]["status"] == "within"
    assert notes[0]["diff"] == 10.0
    assert notes[0]["ceil"] == 50.0


@pytest.mark.parametrize(
    "field, values, expected_code, expected_ai, expected_decision",
    [
        (
            "balance_owed",
            {"experian": "200", "equifax": "200", "transunion": None},
            "C1_TWO_PRESENT_ONE_MISSING",
            False,
            "supportive_needs_companion",
        ),
        (
            "balance_owed",
            {"experian": "open", "equifax": None, "transunion": "--"},
            "C2_ONE_MISSING",
            False,
            "supportive_needs_companion",
        ),
        (
            "account_type",
            {"experian": "revolving", "equifax": "installment", "transunion": None},
            "C3_TWO_PRESENT_CONFLICT",
            True,
            "supportive_needs_companion",
        ),
        (
            "account_rating",
            {"experian": "1", "equifax": "1", "transunion": "2"},
            "C4_TWO_MATCH_ONE_DIFF",
            True,
            "strong_actionable",
        ),
        (
            "account_status",
            {"experian": None, "equifax": "", "transunion": "--"},
            "C6_ALL_MISSING",
            False,
            None,
        ),
    ],
)
def test_build_summary_payload_reason_codes_send_to_ai(
    field: str,
    values: dict[str, object],
    expected_code: str,
    expected_ai: bool,
    expected_decision: str | None,
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
    assert len(payload["findings"]) == 1

    finding = payload["findings"][0]
    assert finding["field"] == field
    assert finding["reason_code"] == expected_code
    assert finding["send_to_ai"] is expected_ai
    assert finding.get("decision") == expected_decision

    bureau_values = finding["bureau_values"]
    assert set(bureau_values.keys()) == {"equifax", "experian", "transunion"}
    for bureau, expected in values.items():
        snapshot = bureau_values[bureau]
        is_missing = expected is None
        if isinstance(expected, str) and expected.strip() in {"", "--"}:
            is_missing = True

        if is_missing:
            assert snapshot["present"] is False
            assert snapshot["normalized"] is None
        else:
            assert snapshot["present"] is True
            assert snapshot["normalized"] == expected


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
    assert seven_norm["transunion"] is None
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


def test_validation_reads_convention(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "sid"
    account_dir = run_dir / "cases" / "accounts" / "000"
    account_dir.mkdir(parents=True)

    bureaus = {
        "transunion": {"date_opened": "15 אוג׳ 2022"},
        "experian": {"date_opened": "15 אוג׳ 2022"},
        "equifax": {"date_opened": "15 אוג׳ 2022"},
    }

    (account_dir / "bureaus.json").write_text(
        json.dumps(bureaus, ensure_ascii=False),
        encoding="utf-8",
    )
    (account_dir / "summary.json").write_text("{}", encoding="utf-8")

    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True)
    (traces_dir / "date_convention.json").write_text(
        json.dumps(
            {
                "date_convention": {
                    "convention": "DMY",
                    "month_language": "he",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = build_validation_requirements_for_account(account_dir, build_pack=False)

    assert result["status"] == "ok"
    summary_after = json.loads((account_dir / "summary.json").read_text(encoding="utf-8"))
    normalized = (
        summary_after["validation_requirements"]["field_consistency"]["date_opened"][
            "normalized"
        ]
    )
    assert normalized["transunion"] == "2022-08-15"


def test_build_validation_requirements_for_account_logs_missing_date_convention(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    run_dir = tmp_path / "runs" / "sid"
    account_dir = run_dir / "cases" / "accounts" / "001"
    account_dir.mkdir(parents=True)

    bureaus = {
        "transunion": {"date_opened": "1/2/23"},
        "experian": {"date_opened": "1/2/23"},
        "equifax": {"date_opened": "1/2/23"},
    }

    (account_dir / "bureaus.json").write_text(
        json.dumps(bureaus, ensure_ascii=False),
        encoding="utf-8",
    )
    (account_dir / "summary.json").write_text("{}", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        result = build_validation_requirements_for_account(account_dir, build_pack=False)

    assert result["status"] == "ok"
    assert "DATE_DETECT_MISSING" in caplog.text

    summary_after = json.loads((account_dir / "summary.json").read_text(encoding="utf-8"))
    normalized = (
        summary_after["validation_requirements"]["field_consistency"]["date_opened"][
            "normalized"
        ]
    )
    assert normalized["experian"] == "2023-01-02"


def test_build_validation_requirements_for_account_accepts_explicit_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "runs_root"
    sid = "SID123"
    run_dir = runs_root / sid
    (run_dir / "traces").mkdir(parents=True)
    (run_dir / "traces" / "date_convention.json").write_text(
        json.dumps({"date_convention": {"convention": "MDY"}}, ensure_ascii=False),
        encoding="utf-8",
    )

    account_dir = tmp_path / "isolated_account"
    account_dir.mkdir()
    (account_dir / "bureaus.json").write_text("{}", encoding="utf-8")
    (account_dir / "summary.json").write_text("{}", encoding="utf-8")

    captured: dict[str, Path] = {}

    def fake_read_date_convention(path: Path) -> dict[str, str]:
        captured["run_dir"] = Path(path)
        return {"convention": "MDY"}

    monkeypatch.setattr(
        "backend.core.logic.validation_requirements.read_date_convention",
        fake_read_date_convention,
    )

    monkeypatch.setattr(
        "backend.core.logic.validation_requirements.build_validation_requirements",
        lambda bureaus, field_consistency=None: ([], {}, {}),
    )
    monkeypatch.setattr(
        "backend.core.logic.validation_requirements._raw_value_provider_for_account_factory",
        lambda bureaus: None,
    )

    applied: dict[str, Mapping[str, Any]] = {}

    def fake_apply(summary_path: Path, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        applied["payload"] = dict(payload)
        return dict(payload)

    monkeypatch.setattr(
        "backend.core.logic.validation_requirements.apply_validation_summary",
        fake_apply,
    )

    result = build_validation_requirements_for_account(
        account_dir,
        build_pack=False,
        sid=sid,
        runs_root=runs_root,
    )

    assert result["status"] == "ok"
    assert captured["run_dir"] == run_dir
    assert "schema_version" in applied["payload"]


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
    assert validation_block["schema_version"] == 3
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
    assert validation_block["schema_version"] == 3
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
    validation_block = summary_data["validation_requirements"]
    assert validation_block["schema_version"] == 3
    assert validation_block["findings"] == []

    sync_validation_tag(tag_path, [], emit=True)
    tags = json.loads(tag_path.read_text(encoding="utf-8"))
    assert all(tag.get("kind") != "validation_required" for tag in tags)



def test_seed_arguments_propagate_to_summary(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")

    normalized_map = {
        "equifax": "open",
        "experian": "open",
        "transunion": "closed",
    }

    requirements = [
        {
            "field": "account_status",
            "category": "status",
            "min_days": 0,
            "documents": [],
            "strength": "strong",
            "ai_needed": False,
        }
    ]

    payload = build_summary_payload(
        requirements,
        field_consistency={
            "account_status": {"normalized": normalized_map}
        },
    )

    findings = payload["findings"]
    assert len(findings) == 1
    seed = findings[0]["argument"]["seed"]
    assert seed["id"] == "account_status__C4_TWO_MATCH_ONE_DIFF"

    assert payload["arguments"]["seeds"] == [seed]
    assert payload["arguments"]["composites"] == []

    apply_validation_summary(summary_path, payload)
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary_data["arguments"]["seeds"] == [seed]
    assert summary_data["arguments"]["composites"] == []

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
    assert validation_payload["schema_version"] == 3
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
    assert validation_block["schema_version"] == 3
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


def test_build_validation_requirements_for_account_dry_run(tmp_path, monkeypatch):
    account_dir = tmp_path / "dry"
    account_dir.mkdir()

    bureaus = {
        "transunion": {"balance_owed": "100", "payment_status": "late"},
        "experian": {"balance_owed": "150", "payment_status": "ok"},
        "equifax": {"balance_owed": "200", "payment_status": "late"},
    }
    (account_dir / "bureaus.json").write_text(json.dumps(bureaus), encoding="utf-8")

    existing_summary = {
        "validation_requirements": {
            "schema_version": 2,
            "findings": [{"field": "legacy", "reason_code": "L0"}],
        }
    }
    (account_dir / "summary.json").write_text(
        json.dumps(existing_summary), encoding="utf-8"
    )

    existing_tags = [
        {"kind": "other", "value": 2},
        {"kind": "validation_required", "fields": ["old"], "at": "2024"},
    ]
    (account_dir / "tags.json").write_text(json.dumps(existing_tags), encoding="utf-8")

    monkeypatch.setenv("WRITE_VALIDATION_TAGS", "1")
    monkeypatch.setattr(
        "backend.core.logic.validation_requirements.backend_config.VALIDATION_DRY_RUN",
        True,
    )
    monkeypatch.setattr(
        "backend.core.logic.validation_requirements.backend_config.VALIDATION_CANARY_PERCENT",
        100,
    )

    build_called = False

    def _fail_build(*_args, **_kwargs):
        nonlocal build_called
        build_called = True
        raise AssertionError("build_validation_pack_for_account should not be called")

    monkeypatch.setattr(
        "backend.core.logic.validation_requirements.build_validation_pack_for_account",
        _fail_build,
    )

    result = build_validation_requirements_for_account(account_dir)

    assert result["status"] == "ok"
    assert result["dry_run"] is True
    assert set(result["fields"]) == {"balance_owed", "payment_status"}
    assert build_called is False

    summary = json.loads((account_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["validation_requirements"] == existing_summary["validation_requirements"]
    shadow_block = summary["validation_requirements_dry_run"]
    assert shadow_block["schema_version"] == 3
    assert {entry["field"] for entry in shadow_block["findings"]} == {
        "balance_owed",
        "payment_status",
    }

    tags = json.loads((account_dir / "tags.json").read_text(encoding="utf-8"))
    assert tags == existing_tags


def test_build_validation_requirements_for_account_canary_skip(tmp_path, monkeypatch):
    account_dir = tmp_path / "skip"
    account_dir.mkdir()

    bureaus = {
        "transunion": {"balance_owed": "100", "payment_status": "late"},
        "experian": {"balance_owed": "150", "payment_status": "ok"},
        "equifax": {"balance_owed": "200", "payment_status": "late"},
    }
    (account_dir / "bureaus.json").write_text(json.dumps(bureaus), encoding="utf-8")

    original_summary = {"existing": True}
    (account_dir / "summary.json").write_text(
        json.dumps(original_summary), encoding="utf-8"
    )

    monkeypatch.setattr(
        "backend.core.logic.validation_requirements.backend_config.VALIDATION_CANARY_PERCENT",
        0,
    )
    monkeypatch.setattr(
        "backend.core.logic.validation_requirements.backend_config.VALIDATION_DRY_RUN",
        False,
    )

    result = build_validation_requirements_for_account(account_dir)

    assert result["status"] == "canary_skipped"
    assert result["count"] == 0
    assert result["fields"] == []
    assert result["validation_requirements"] is None
    assert result["dry_run"] is False

    summary = json.loads((account_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary == original_summary
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
    assert validation_payload["schema_version"] == 3
    assert "requirements" not in validation_payload
    assert validation_payload["findings"] == []
    assert "field_consistency" in validation_payload
    field_consistency = validation_payload["field_consistency"]
    assert "balance_owed" in field_consistency
    assert field_consistency["balance_owed"]["consensus"] == "unanimous"

    summary = json.loads((account_dir / "summary.json").read_text(encoding="utf-8"))
    validation_block = summary["validation_requirements"]
    assert validation_block["schema_version"] == 3
    assert "requirements" not in validation_block
    assert validation_block["findings"] == []
    assert "balance_owed" in validation_block["field_consistency"]

    tags = json.loads((account_dir / "tags.json").read_text(encoding="utf-8"))
    assert all(tag.get("kind") != "validation_required" for tag in tags)


def test_build_validation_requirements_writes_summary_when_missing(tmp_path, monkeypatch):
    account_dir = tmp_path / "2"
    account_dir.mkdir()

    bureaus = {
        "transunion": {"balance_owed": "100"},
        "experian": {"balance_owed": "100"},
        "equifax": {"balance_owed": "100"},
    }
    (account_dir / "bureaus.json").write_text(json.dumps(bureaus), encoding="utf-8")

    meta = {
        "account_index": 2,
        "pointers": {
            "raw": "raw_lines.json",
            "bureaus": "bureaus.json",
            "flat": "fields_flat.json",
            "tags": "tags.json",
            "summary": "summary.json",
        },
        "account_id": "acct-002",
    }
    (account_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (account_dir / "tags.json").write_text("[]", encoding="utf-8")

    monkeypatch.delenv("WRITE_VALIDATION_TAGS", raising=False)

    result = build_validation_requirements_for_account(account_dir)

    assert result["status"] == "ok"
    assert result["count"] == 0
    assert result["validation_requirements"]["findings"] == []

    summary_path = account_dir / "summary.json"
    assert summary_path.is_file()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["account_index"] == 2
    assert summary["account_id"] == "acct-002"
    assert summary["problem_reasons"] == []
    assert summary["problem_tags"] == []

    pointers = summary["pointers"]
    assert pointers["summary"] == "summary.json"
    assert pointers["bureaus"] == "bureaus.json"

    validation_block = summary["validation_requirements"]
    assert validation_block["findings"] == []


def _write_basic_bureaus(account_dir: Path) -> None:
    bureaus = {
        "transunion": {"balance_owed": "100"},
        "experian": {"balance_owed": "150"},
        "equifax": {"balance_owed": "200"},
    }
    (account_dir / "bureaus.json").write_text(
        json.dumps(bureaus, ensure_ascii=False), encoding="utf-8"
    )


def _write_tolerance_bureaus(account_dir: Path) -> None:
    bureaus = {
        "transunion": {"balance_owed": "110"},
        "experian": {"balance_owed": "100"},
        "equifax": {"balance_owed": "102"},
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


def test_tolerance_notes_absent_when_debug_off(tmp_path, monkeypatch):
    from backend.core.logic import consistency as consistency_mod

    monkeypatch.setattr(consistency_mod, "_AMOUNT_TOL_ABS", 0.0, raising=False)
    monkeypatch.setattr(consistency_mod, "_AMOUNT_TOL_RATIO", 0.0, raising=False)

    run_dir = tmp_path / "SID-NODEBUG"
    account_dir = run_dir / "cases" / "accounts" / "1"
    account_dir.mkdir(parents=True)
    _write_tolerance_bureaus(account_dir)
    (account_dir / "tags.json").write_text("[]", encoding="utf-8")
    monkeypatch.delenv("VALIDATION_DEBUG", raising=False)

    result = build_validation_requirements_for_account(account_dir)
    assert result["status"] == "ok"

    summary = json.loads((account_dir / "summary.json").read_text(encoding="utf-8"))
    assert "tolerance_notes" not in summary


def test_tolerance_notes_present_when_debug_on(tmp_path, monkeypatch):
    from backend.core.logic import consistency as consistency_mod

    monkeypatch.setattr(consistency_mod, "_AMOUNT_TOL_ABS", 0.0, raising=False)
    monkeypatch.setattr(consistency_mod, "_AMOUNT_TOL_RATIO", 0.0, raising=False)

    run_dir = tmp_path / "SID-DEBUG"
    account_dir = run_dir / "cases" / "accounts" / "1"
    account_dir.mkdir(parents=True)
    _write_tolerance_bureaus(account_dir)
    (account_dir / "tags.json").write_text("[]", encoding="utf-8")
    monkeypatch.setenv("VALIDATION_DEBUG", "1")

    result = build_validation_requirements_for_account(account_dir)
    assert result["status"] == "ok"

    summary = json.loads((account_dir / "summary.json").read_text(encoding="utf-8"))
    notes = summary.get("tolerance_notes")
    assert isinstance(notes, list) and len(notes) == 1
    note = notes[0]
    assert note["field"] == "balance_owed"
    assert note["status"] == "within"
    assert note["diff"] == 10.0
    assert note["ceil"] == 50.0
