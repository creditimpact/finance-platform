from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.core.logic.json_utils import parse_json


def test_parse_valid_json():
    data, err = parse_json('{"a": 1}')
    assert data == {"a": 1}
    assert err is None


def test_parse_trailing_comma():
    data, err = parse_json('{"a": 1,}')
    assert data == {"a": 1}
    assert err is None


def test_parse_missing_comma():
    data, err = parse_json('{"a": 1 "b": 2}')
    assert data == {"a": 1, "b": 2}
    assert err is None


def test_parse_single_quotes():
    data, err = parse_json("{'a': 'b'}")
    assert data == {"a": "b"}
    assert err is None


def test_parse_unquoted_keys():
    data, err = parse_json('{advisor_comment: "text here"}')
    assert data == {"advisor_comment": "text here"}
    assert err is None


def test_parse_mismatched_braces():
    data, err = parse_json('{"a": 1, "b": 2')
    assert data == {"a": 1, "b": 2}
    assert err is None


def test_parse_invalid_json():
    data, err = parse_json("not json")
    assert data == {}
    assert err == "invalid_json"
