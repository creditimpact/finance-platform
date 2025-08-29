import re
from datetime import datetime

def to_number(val):
    if val is None:
        return None
    s = str(val).strip()
    s = re.sub(r"[,$]", "", s)
    try:
        return float(s)
    except Exception:
        return val


def to_iso_date(val):
    if val is None:
        return None
    s = str(val).strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    return val
