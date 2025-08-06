from pathlib import Path
import sys
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from logic import rules_loader


def test_load_rules():
    rules = rules_loader.load_rules()
    assert isinstance(rules, list)
    assert rules[0]["id"] == "RULE_NO_ADMISSION"


def test_load_neutral_phrases():
    phrases = rules_loader.load_neutral_phrases()
    assert "not_mine" in phrases
    assert phrases["not_mine"][0].startswith("I do not recognize")


def test_load_state_rules():
    state_rules = rules_loader.load_state_rules()
    assert state_rules["CA"]["requires"][0] == "license_number"
    assert state_rules["GA"]["prohibit_service"] is True


def test_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(rules_loader, "RULES_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        rules_loader.load_rules()


def test_invalid_yaml(tmp_path, monkeypatch):
    (tmp_path / "dispute_rules.yaml").write_text(": bad\n", encoding="utf-8")
    monkeypatch.setattr(rules_loader, "RULES_DIR", tmp_path)
    with pytest.raises(RuntimeError):
        rules_loader.load_rules()
