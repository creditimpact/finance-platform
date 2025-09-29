from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.core.io.tags import read_tags
from backend.core.logic.intra_polarity import analyze_account_polarity


@pytest.fixture()
def account_dir(tmp_path: Path) -> Path:
    account_path = tmp_path / "runs" / "sid123" / "cases" / "accounts" / "0"
    account_path.mkdir(parents=True)
    return account_path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_analyze_account_polarity_updates_summary(account_dir: Path) -> None:
    _write_json(
        account_dir / "bureaus.json",
        {
            "transunion": {
                "balance_owed": "$120.00",
                "payment_status": "Account in Collection",
            },
            "experian": {
                "balance_owed": "$0",
                "payment_status": "Paid in Full",
            },
        },
    )
    _write_json(account_dir / "summary.json", {"existing": {"keep": True}})

    result = analyze_account_polarity("sid123", account_dir)

    assert result == {
        "balance_owed": {
            "transunion": {"polarity": "bad", "severity": "high"},
            "experian": {"polarity": "good", "severity": "medium"},
        },
        "payment_status": {
            "transunion": {"polarity": "bad", "severity": "high"},
            "experian": {"polarity": "good", "severity": "medium"},
        },
    }

    summary_data = json.loads((account_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_data["existing"] == {"keep": True}
    assert summary_data["polarity_check"] == result


def test_analyze_account_polarity_writes_probe_tags(account_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WRITE_POLARITY_PROBES", "1")

    _write_json(
        account_dir / "bureaus.json",
        {
            "equifax": {
                "closed_date": "--",
            }
        },
    )

    analyze_account_polarity("sid456", account_dir)
    analyze_account_polarity("sid456", account_dir)

    tags = read_tags(account_dir)
    assert tags == [
        {
            "source": "intra_polarity",
            "kind": "polarity_probe",
            "field": "closed_date",
            "bureau": "equifax",
            "polarity": "neutral",
            "severity": "low",
        }
    ]
