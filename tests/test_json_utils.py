from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from logic.json_utils import parse_json

def test_parse_valid_json():
    assert parse_json('{"a": 1}') == {"a": 1}

def test_parse_trailing_comma():
    assert parse_json('{"a": 1,}') == {"a": 1}

def test_parse_missing_comma():
    assert parse_json('{"a": 1 "b": 2}') == {"a": 1, "b": 2}

def test_parse_single_quotes():
    assert parse_json("{'a': 'b'}") == {"a": "b"}

def test_parse_unquoted_keys():
    assert parse_json('{advisor_comment: "text here"}') == {"advisor_comment": "text here"}

def test_parse_mismatched_braces():
    assert parse_json('{"a": 1, "b": 2') == {"a": 1, "b": 2}
