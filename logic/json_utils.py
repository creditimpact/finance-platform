import json
import logging
import re
from pathlib import Path

try:
    from json_repair import repair_json as _jsonrepair
except Exception:  # pragma: no cover - optional dependency
    _jsonrepair = None

try:
    import dirtyjson as _dirtyjson
except Exception:  # pragma: no cover - fallback when lib missing
    _dirtyjson = None

try:
    import json5 as _json5
except Exception:  # pragma: no cover - optional dependency
    _json5 = None

_TRAILING_COMMA_RE = re.compile(r',(?=\s*[}\]])')
_SINGLE_QUOTE_KEY_RE = re.compile(r"'([^']*)'(?=\s*:)" )
_SINGLE_QUOTE_VALUE_RE = re.compile(r":\s*'([^']*)'" )
_UNQUOTED_KEY_RE = re.compile(r'(?P<prefix>^|[,{])\s*(?P<key>[A-Za-z_][A-Za-z0-9_-]*)\s*(?=:)')
_LOG_PATH = Path("invalid_ai_json.log")


def _log_invalid_json(raw: str) -> None:
    """Persist the raw AI output for debugging."""
    try:
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(raw)
            f.write("\n---\n")
    except Exception:  # pragma: no cover - best effort only
        logging.debug("Unable to log invalid JSON output.")


def _basic_clean(content: str) -> str:
    """Apply regex-based fixes for common JSON issues."""
    cleaned = _TRAILING_COMMA_RE.sub("", content)
    cleaned = _SINGLE_QUOTE_KEY_RE.sub(r'"\1"', cleaned)
    cleaned = _SINGLE_QUOTE_VALUE_RE.sub(r': "\1"', cleaned)
    cleaned = _UNQUOTED_KEY_RE.sub(lambda m: f'{m.group("prefix")}"{m.group("key")}"', cleaned)
    return cleaned


def _repair_json(content: str) -> str:
    """Attempt to repair malformed JSON using json_repair or basic regex fixes."""
    if _jsonrepair is not None:
        try:
            return _jsonrepair(content)
        except Exception:  # pragma: no cover - if repair fails, fall back
            logging.debug("json_repair failed to process input")
    return _basic_clean(content)


def parse_json(text: str):
    """Parse ``text`` as JSON, attempting repairs on failure."""
    normalized = _repair_json(text)
    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        _log_invalid_json(text)
        logging.warning("⚠️ The AI returned invalid JSON. Attempting to repair.")
        cleaned = _basic_clean(normalized)
        if cleaned != normalized:
            logging.debug("Cleaned JSON string: %s", cleaned)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        if _dirtyjson is not None:
            try:
                repaired = _dirtyjson.loads(normalized)
                logging.debug("Repaired JSON via dirtyjson: %s", repaired)
                return repaired
            except Exception:  # pragma: no cover - handle silently
                logging.debug("dirtyjson failed to parse input")
        if _json5 is not None:
            try:
                repaired = _json5.loads(normalized)
                logging.debug("Repaired JSON via json5: %s", repaired)
                return repaired
            except Exception:  # pragma: no cover - handle silently
                logging.debug("json5 failed to parse input")
        raise
