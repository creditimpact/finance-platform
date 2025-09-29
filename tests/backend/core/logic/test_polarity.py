import json
from pathlib import Path

from backend.core.logic import polarity


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_apply_polarity_checks_updates_summary(tmp_path: Path) -> None:
    accounts_dir = tmp_path / "accounts"
    account_dir = accounts_dir / "1"

    bureaus_payload = {
        "order": ["transunion", "experian"],
        "transunion": {
            "balance_owed": "$100",
            "past_due_amount": "$0",
            "payment_status": "Paid in Full",
            "account_status": "Closed - Paid",
            "closed_date": "2024-01-01",
            "last_payment": "--",
            "creditor_remarks": "Paid/Settled",
            "account_type": "Installment",
            "creditor_type": "Bank Credit Cards",
        },
        "experian": {
            "balance_owed": "$0",
            "past_due_amount": "$50",
            "payment_status": "Collection account",
            "account_status": "Collection",
            "closed_date": "--",
            "last_payment": "2023-12-01",
            "creditor_remarks": "Charged off",
            "account_type": "Collection",
            "creditor_type": "Debt Buyers",
        },
    }
    summary_payload = {"account_index": 1}

    _write_json(account_dir / "bureaus.json", bureaus_payload)
    _write_json(account_dir / "summary.json", summary_payload)

    result = polarity.apply_polarity_checks(accounts_dir, [1], enable_tags_probe=False)

    assert result.processed_accounts == 1
    assert result.updated_accounts == [1]
    assert result.config_digest
    assert result.results[1]["schema_version"] == 1

    summary_after = json.loads((account_dir / "summary.json").read_text(encoding="utf-8"))
    block = summary_after.get("polarity_check")
    assert block and block.get("schema_version") == 1
    assert block.get("config_digest") == result.config_digest

    transunion = block["bureaus"]["transunion"]
    experian = block["bureaus"]["experian"]

    assert transunion["balance_owed"]["polarity"] == "bad"
    assert transunion["past_due_amount"]["polarity"] == "good"
    assert transunion["past_due_amount"]["severity"] == "medium"
    assert transunion["payment_status"]["polarity"] == "good"
    assert experian["past_due_amount"]["polarity"] == "bad"
    assert experian["creditor_remarks"]["polarity"] == "bad"
    assert experian["creditor_type"]["polarity"] == "bad"

    second = polarity.apply_polarity_checks(accounts_dir, [1], enable_tags_probe=False)
    assert second.processed_accounts == 1
    assert second.updated_accounts == []

    # Probe tags when enabled
    polarity.apply_polarity_checks(accounts_dir, [1], enable_tags_probe=True)
    tags_path = account_dir / "tags.json"
    assert tags_path.exists()
    tags = json.loads(tags_path.read_text(encoding="utf-8"))
    probe = next(
        tag
        for tag in tags
        if tag.get("kind") == "polarity_probe"
        and tag.get("bureau") == "experian"
        and tag.get("field") == "past_due_amount"
    )
    assert probe["polarity"] == "bad"
    assert probe.get("config_digest") == result.config_digest
