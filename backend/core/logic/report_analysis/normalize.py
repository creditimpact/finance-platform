import re
from datetime import datetime

def to_number(val):
    """Return ``val`` converted to ``float`` when unambiguous."""

    if val is None:
        return None

    s = str(val).strip()
    # Remove currency symbols, thousands separators and optional CR/DR tokens
    s = re.sub(r"[,$]", "", s)
    s = re.sub(r"\b(?:CR|DR)\b", "", s, flags=re.I)
    s = s.strip()

    try:
        return float(s)
    except Exception:
        return val


def to_iso_date(val):
    """Return ``val`` normalized to ``YYYY-MM-DD`` when possible."""

    if val is None:
        return None
    s = str(val).strip()
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%m/%Y", "%Y-%m"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    return val
