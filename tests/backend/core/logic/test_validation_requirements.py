import json

from backend.core.logic.consistency import compute_inconsistent_fields
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
    assert "account_status" in inconsistencies
    assert inconsistencies["account_status"]["normalized"]["transunion"] == "open"


def test_build_validation_requirements_uses_config_and_defaults():
    bureaus = {
        "transunion": {"balance_owed": "100", "mystery_field": "abc"},
        "experian": {"balance_owed": "200", "mystery_field": "xyz"},
        "equifax": {"balance_owed": "200", "mystery_field": "xyz"},
    }

    requirements = build_validation_requirements(bureaus)
    fields = [entry["field"] for entry in requirements]

    assert fields == ["balance_owed", "mystery_field"]

    balance_rule = next(entry for entry in requirements if entry["field"] == "balance_owed")
    assert balance_rule["category"] == "activity"
    assert balance_rule["min_days"] == 8
    assert "monthly_statement" in balance_rule["documents"]

    mystery_rule = next(entry for entry in requirements if entry["field"] == "mystery_field")
    assert mystery_rule["category"] == "unspecified"
    assert mystery_rule["min_days"] == 3
    assert mystery_rule["documents"] == []


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
        {"field": "balance_owed", "category": "activity", "min_days": 8, "documents": []}
    ]
    payload = build_summary_payload(requirements)

    apply_validation_summary(summary_path, payload)
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_data["validation_requirements"]["count"] == 1
    assert summary_data["existing"] is True

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

    existing_summary = {"existing": True}
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

    summary = json.loads((account_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["existing"] is True
    validation_block = summary["validation_requirements"]
    assert validation_block["count"] == 2
    fields = {entry["field"] for entry in validation_block["requirements"]}
    assert fields == {"balance_owed", "payment_status"}

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
        "validation_requirements": {
            "count": 1,
            "requirements": [
                {
                    "field": "balance_owed",
                    "category": "activity",
                    "min_days": 8,
                    "documents": [],
                }
            ],
        }
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

    summary = json.loads((account_dir / "summary.json").read_text(encoding="utf-8"))
    assert "validation_requirements" not in summary

    tags = json.loads((account_dir / "tags.json").read_text(encoding="utf-8"))
    assert all(tag.get("kind") != "validation_required" for tag in tags)
