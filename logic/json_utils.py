import json
import logging
import re

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


def _basic_clean(content: str) -> str:
    """Apply simple regex-based fixes for common JSON issues."""
    cleaned = _TRAILING_COMMA_RE.sub(r"", content)
    cleaned = _SINGLE_QUOTE_KEY_RE.sub(r'"\1"', cleaned)
    cleaned = _SINGLE_QUOTE_VALUE_RE.sub(r': "\1"', cleaned)
    return cleaned


def parse_json(text: str):
    """Parse ``text`` as JSON, attempting repairs on failure.

    This first tries :func:`json.loads`. If that fails, it tries ``dirtyjson`` or
    ``json5`` when available. As a last resort it applies basic regex fixes for
    trailing commas and single quotes. If parsing still fails, the original
    ``JSONDecodeError`` is raised.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logging.warning("⚠️ The AI returned invalid JSON. Attempting to repair.")
        # Try dirtyjson if available
        if _dirtyjson is not None:
            try:
                repaired = _dirtyjson.loads(text)
                logging.debug("Repaired JSON via dirtyjson: %s", repaired)
                return repaired
            except Exception:  # pragma: no cover - handle silently
                logging.debug("dirtyjson failed to parse input")
        if _json5 is not None:
            try:
                repaired = _json5.loads(text)
                logging.debug("Repaired JSON via json5: %s", repaired)
                return repaired
            except Exception:  # pragma: no cover - handle silently
                logging.debug("json5 failed to parse input")
        cleaned = _basic_clean(text)
        if cleaned != text:
            logging.debug("Cleaned JSON string: %s", cleaned)
            return json.loads(cleaned)
        raise
