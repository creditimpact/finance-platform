from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from backend.core.logic import polarity


def _write_config(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _prepare_config(monkeypatch, tmp_path: Path, data: Dict[str, Any]) -> Path:
    config_path = tmp_path / "polarity_config.yml"
    _write_config(config_path, data)
    monkeypatch.setattr(polarity, "_POLARITY_CONFIG_PATH", config_path)
    polarity._CONFIG_CACHE = None
    return config_path


def test_load_polarity_config_reload(monkeypatch, tmp_path: Path) -> None:
    config_path = _prepare_config(
        monkeypatch,
        tmp_path,
        {"fields": {"status": {"type": "text", "default": "unknown"}}},
    )

    first = polarity.load_polarity_config()
    assert first["fields"]["status"]["type"] == "text"

    updated = {"fields": {"status": {"type": "text", "default": "neutral"}}}
    _write_config(config_path, updated)

    stat = config_path.stat()
    os.utime(config_path, (stat.st_atime + 1, stat.st_mtime + 1))

    second = polarity.load_polarity_config()
    assert second["fields"]["status"]["default"] == "neutral"


def test_parse_money_variants() -> None:
    assert polarity.parse_money("$1,200.50") == 1200.50
    assert polarity.parse_money("(45.00)") == -45.0
    assert polarity.parse_money("--") is None
    assert polarity.parse_money(None) is None


def test_norm_text_and_blank_detection() -> None:
    assert polarity.norm_text("  Hello   World  ") == "hello world"
    assert polarity.norm_text(None) == ""
    assert polarity.is_blank("--") is True
    assert polarity.is_blank(0) is False


def test_classify_money_field(monkeypatch, tmp_path: Path) -> None:
    _prepare_config(
        monkeypatch,
        tmp_path,
        {
            "fields": {
                "balance": {
                    "type": "money",
                    "rules": [
                        {"if": "value > 0", "polarity": "bad", "severity": "high"},
                        {"if": "value == 0", "polarity": "good", "severity": "medium"},
                    ],
                }
            }
        },
    )

    positive = polarity.classify_field_value("balance", "$120.00")
    assert positive["polarity"] == "bad"
    assert positive["severity"] == "high"
    assert positive["value_norm"] == 120.0
    assert positive["rule_hit"] == "value > 0"
    assert positive["reason"] == 'condition "value > 0" matched'
    assert positive["evidence"]["matched_rule"] == "value > 0"
    assert positive["evidence"]["parsed"] == 120.0

    zero = polarity.classify_field_value("balance", "$0")
    assert zero["polarity"] == "good"
    assert zero["severity"] == "medium"
    assert zero["value_norm"] == 0.0
    assert zero["rule_hit"] == "value == 0"
    assert zero["reason"] == 'condition "value == 0" matched'

    missing = polarity.classify_field_value("balance", "--")
    assert missing["polarity"] == "unknown"
    assert missing["severity"] == "low"
    assert missing["value_norm"] is None
    assert missing["rule_hit"] is None
    assert missing["reason"] == "value missing or invalid"
    assert missing["evidence"]["parsed"] is None


def test_classify_text_field(monkeypatch, tmp_path: Path) -> None:
    _prepare_config(
        monkeypatch,
        tmp_path,
        {
            "fields": {
                "payment_status": {
                    "type": "text",
                    "bad_keywords": [
                        {"id": "PAYMENT_STATUS_COLLECTION", "pattern": "collection"}
                    ],
                    "good_keywords": [
                        {"id": "PAYMENT_STATUS_PAID_IN_FULL", "pattern": "paid in full"}
                    ],
                    "neutral_keywords": [
                        {"id": "PAYMENT_STATUS_PLACEHOLDER", "pattern": "--"}
                    ],
                    "default": "unknown",
                    "weights": {"bad": "high", "good": "medium", "neutral": "low"},
                }
            }
        },
    )

    bad = polarity.classify_field_value("payment_status", "Account in Collection")
    assert bad["polarity"] == "bad"
    assert bad["severity"] == "high"
    assert bad["value_norm"] == "account in collection"
    assert bad["rule_hit"] == "PAYMENT_STATUS_COLLECTION"
    assert bad["reason"] == "matched bad keyword 'collection'"
    assert bad["evidence"]["matched_keyword"] == "collection"

    good = polarity.classify_field_value("payment_status", "Paid in Full")
    assert good["polarity"] == "good"
    assert good["severity"] == "medium"
    assert good["value_norm"] == "paid in full"
    assert good["rule_hit"] == "PAYMENT_STATUS_PAID_IN_FULL"
    assert good["reason"] == "matched good keyword 'paid in full'"

    default = polarity.classify_field_value("payment_status", "unknown status")
    assert default["polarity"] == "unknown"
    assert default["severity"] == "low"
    assert default["value_norm"] == "unknown status"
    assert default["rule_hit"] is None
    assert default["reason"] == "default polarity 'unknown'"


def test_classify_date_field(monkeypatch, tmp_path: Path) -> None:
    _prepare_config(
        monkeypatch,
        tmp_path,
        {
            "fields": {
                "closed_date": {
                    "type": "date",
                    "rules": [
                        {"if": "is_present == true", "polarity": "good", "severity": "low"},
                        {"if": "is_present == false", "polarity": "neutral", "severity": "low"},
                    ],
                }
            }
        },
    )

    present = polarity.classify_field_value("closed_date", "2024-01-01")
    assert present["polarity"] == "good"
    assert present["evidence"]["matched_rule"] == "is_present == true"
    assert present["value_norm"] is True
    assert present["rule_hit"] == "is_present == true"
    assert present["reason"] == 'condition "is_present == true" matched'

    missing = polarity.classify_field_value("closed_date", "--")
    assert missing["polarity"] == "neutral"
    assert missing["severity"] == "low"
    assert missing["value_norm"] is False
    assert missing["rule_hit"] == "is_present == false"
    assert missing["reason"] == 'condition "is_present == false" matched'


def test_classify_unknown_field(monkeypatch, tmp_path: Path) -> None:
    _prepare_config(monkeypatch, tmp_path, {"fields": {}})
    result = polarity.classify_field_value("nonexistent", "value")
    assert result == {
        "polarity": "unknown",
        "severity": "low",
        "value_norm": None,
        "rule_hit": None,
        "reason": "field not configured",
        "evidence": {"parsed": None},
    }

