from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


def _mid(a: Any, b: Any) -> float:
    try:
        return (float(a) + float(b)) / 2.0
    except Exception:
        return 0.0


def _join_text(tokens: List[dict]) -> str:
    def midx(t: dict) -> float:
        return _mid(t.get("x0", 0.0), t.get("x1", 0.0))

    toks = sorted(tokens or [], key=midx)
    out: List[str] = []
    prev: dict | None = None
    for t in toks:
        txt = str(t.get("text", "")).strip()
        if not txt:
            continue
        if prev is None:
            out.append(txt)
            prev = t
            continue
        try:
            gap = float(t.get("x0", 0.0)) - float(prev.get("x1", 0.0))
        except Exception:
            gap = 0.0
        if gap >= 1.0 or (out[-1] and txt and out[-1][-1].isalnum() and txt[0].isalnum()):
            out.append(" ")
        out.append(txt)
        prev = t
    return "".join(out).strip()


def _norm(s: str | None) -> str:
    import re
    return re.sub(r"\W+", "", (s or "").lower())


_HIST_LABEL_NORM = "twoyearpaymenthistory"
_BUREAUS = ("transunion", "experian", "equifax")

_HEB_TO_EN_MONTH = {
    "ינו׳": "Jan",
    "פבר׳": "Feb",
    "מרץ": "Mar",
    "אפר׳": "Apr",
    "מאי": "May",
    "יוני": "Jun",
    "יולי": "Jul",
    "אוג׳": "Aug",
    "ספט׳": "Sep",
    "אוק׳": "Oct",
    "נוב׳": "Nov",
    "דצמ׳": "Dec",
}

_EN_MONTHS = {
    "jan": "Jan",
    "feb": "Feb",
    "mar": "Mar",
    "apr": "Apr",
    "may": "May",
    "jun": "Jun",
    "jul": "Jul",
    "aug": "Aug",
    "sep": "Sep",
    "oct": "Oct",
    "nov": "Nov",
    "dec": "Dec",
}

_STATUS = {"ok", "30", "60", "90", "120", "150", "180"}


def _is_month_token(txt: str) -> str | None:
    z = (txt or "").strip()
    if not z:
        return None
    if z in _HEB_TO_EN_MONTH:
        return _HEB_TO_EN_MONTH[z]
    nz = _norm(z)
    if nz[:3] in _EN_MONTHS:
        return _EN_MONTHS[nz[:3]]
    return None


def _is_status_token(txt: str) -> Optional[str]:
    z = (txt or "").strip().upper()
    if not z:
        return None
    if z == "OK":
        return "OK"
    # numeric codes like 30/60/90
    if z.isdigit() and z in _STATUS:
        return z
    return None


def _cluster_lines_y(tokens: List[dict], dy: float) -> List[List[dict]]:
    toks = sorted(tokens or [], key=lambda t: (_mid(t.get("y0", 0.0), t.get("y1", 0.0)), _mid(t.get("x0", 0.0), t.get("x1", 0.0))))
    groups: List[List[dict]] = []
    cur: List[dict] = []
    cur_y: float | None = None
    for t in toks:
        y = _mid(t.get("y0", 0.0), t.get("y1", 0.0))
        if cur_y is None or abs(y - cur_y) <= dy:
            cur.append(t)
            try:
                cur_y = sum(_mid(x.get("y0", 0.0), x.get("y1", 0.0)) for x in cur) / len(cur)
            except Exception:
                cur_y = y
        else:
            groups.append(cur)
            cur = [t]
            cur_y = y
    if cur:
        groups.append(cur)
    return groups


def extract_two_year_payment_history(
    session_id: str,
    block_id: int,
    heading: str | None,
    page_tokens: List[dict],
    window: Dict[str, Any],
    bands: Dict[str, Tuple[float, float]] | None,
    out_dir: Path,
) -> Optional[str]:
    # 1) Filter tokens to window Y range
    try:
        y_top = float(window.get("y_top", 0.0) or 0.0)
        y_bottom = float(window.get("y_bottom", 0.0) or 0.0)
    except Exception:
        return None
    yband = [t for t in (page_tokens or []) if float(t.get("y0", 0.0)) <= y_bottom and float(t.get("y1", 0.0)) >= y_top]

    # 2) Find heading line for Two-Year Payment History
    lines = _cluster_lines_y(yband, dy=1.5)
    hist_y_start: Optional[float] = None
    for line in lines:
        text = _join_text(line)
        if _norm(text) == _HIST_LABEL_NORM:
            try:
                hist_y_start = sum(_mid(x.get("y0", 0.0), x.get("y1", 0.0)) for x in line) / max(1, len(line))
            except Exception:
                hist_y_start = _mid(line[0].get("y0", 0.0), line[0].get("y1", 0.0))
            break
    if hist_y_start is None:
        return None

    # 3) Per-bureau collect (month,status) pairs
    values: Dict[str, List[str]] = {}
    months_axis: List[str] = []
    # Determine bands or fallback to whole window thirds
    bands_use: Dict[str, Tuple[float, float]] = {}
    if bands:
        for name, rng in bands.items():
            bands_use[name.lower()] = (float(rng[0]), float(rng[1]))
    else:
        try:
            x_min = float(window.get("x_min", 0.0)); x_max = float(window.get("x_max", 0.0))
        except Exception:
            return None
        w = (x_max - x_min) / 3.0 if (x_max - x_min) > 0 else 0.0
        bands_use = {
            "transunion": (x_min, x_min + w),
            "experian": (x_min + w, x_min + 2 * w),
            "equifax": (x_min + 2 * w, x_max),
        }

    # Prepare tokens by Y after historical header
    yband_after = [t for t in yband if _mid(t.get("y0", 0.0), t.get("y1", 0.0)) >= hist_y_start]

    def tokens_in_band(name: str) -> List[dict]:
        xL, xR = bands_use.get(name, (None, None))
        if xL is None:
            return []
        out: List[dict] = []
        for t in yband_after:
            mx = _mid(t.get("x0", 0.0), t.get("x1", 0.0))
            if xL <= mx <= xR:
                out.append(t)
        # reading-order
        out.sort(key=lambda t: (_mid(t.get("y0", 0.0), t.get("y1", 0.0)), _mid(t.get("x0", 0.0), t.get("x1", 0.0))))
        return out

    for bureau in _BUREAUS:
        toks = tokens_in_band(bureau)
        if not toks:
            continue
        # find bureau heading within these tokens to start after
        start_y = None
        for t in toks:
            if _norm(t.get("text")) == bureau:
                start_y = _mid(t.get("y0", 0.0), t.get("y1", 0.0))
                break
        if start_y is None:
            # Accept from hist_y_start if no heading present in band
            start_y = hist_y_start
        seq_status: List[str] = []
        seq_months: List[str] = []
        last_status: Optional[str] = None
        for t in toks:
            if _mid(t.get("y0", 0.0), t.get("y1", 0.0)) < start_y:
                continue
            txt = str(t.get("text", "")).strip()
            st = _is_status_token(txt)
            if st:
                last_status = st
                continue
            m = _is_month_token(txt)
            if m:
                seq_months.append(m)
                seq_status.append(last_status or "")
                last_status = None
        # limit to last 24 months if longer
        if len(seq_months) > 24:
            seq_months = seq_months[-24:]
            seq_status = seq_status[-24:]
        values[bureau] = seq_status
        if len(seq_months) > len(months_axis):
            months_axis = seq_months

    if not months_axis or not any(values.get(b) for b in _BUREAUS):
        return None

    # Normalize values length to months length
    for b in _BUREAUS:
        seq = values.get(b) or []
        if len(seq) < len(months_axis):
            seq = seq + [""] * (len(months_axis) - len(seq))
        elif len(seq) > len(months_axis):
            seq = seq[: len(months_axis)]
        values[b] = seq

    # Write artifact
    out_obj = {
        "session_id": session_id,
        "block_id": int(block_id),
        "heading": heading,
        "type": "two_year_payment_history",
        "bureaus": list(_BUREAUS),
        "months": months_axis,
        "values": values,
    }
    out_path = out_dir / f"account_table_{block_id:02d}__{(heading or f'block-{block_id}').lower().replace(' ', '-').replace('/', '-')}.history.json"
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def extract_seven_year_delinquency(
    session_id: str,
    block_id: int,
    heading: str | None,
    page_tokens: List[dict],
    window: Dict[str, Any],
    bands: Dict[str, Tuple[float, float]] | None,
    out_dir: Path,
) -> Optional[str]:
    """Extracts the "Days Late - 7 Year History" summary per bureau.
    Returns the output filepath if extracted, else None.
    """
    try:
        y_top = float(window.get("y_top", 0.0) or 0.0)
        y_bottom = float(window.get("y_bottom", 0.0) or 0.0)
    except Exception:
        return None
    yband = [t for t in (page_tokens or []) if float(t.get("y0", 0.0)) <= y_bottom and float(t.get("y1", 0.0)) >= y_top]

    # Find heading line tolerant of dash styles
    target_norm = "dayslate7yearhistory"
    lines = _cluster_lines_y(yband, dy=1.5)
    start_y = None
    for line in lines:
        text = _join_text(line)
        if _norm(text) == target_norm:
            try:
                start_y = sum(_mid(x.get("y0", 0.0), x.get("y1", 0.0)) for x in line) / max(1, len(line))
            except Exception:
                start_y = _mid(line[0].get("y0", 0.0), line[0].get("y1", 0.0))
            break
    if start_y is None:
        return None

    # Choose bands
    bands_use: Dict[str, Tuple[float, float]] = {}
    if bands:
        for name, rng in bands.items():
            bands_use[name.lower()] = (float(rng[0]), float(rng[1]))
    else:
        try:
            x_min = float(window.get("x_min", 0.0)); x_max = float(window.get("x_max", 0.0))
        except Exception:
            return None
        w = (x_max - x_min) / 3.0 if (x_max - x_min) > 0 else 0.0
        bands_use = {
            "transunion": (x_min, x_min + w),
            "experian": (x_min + w, x_min + 2 * w),
            "equifax": (x_min + 2 * w, x_max),
        }

    # Tokens after header only
    yband_after = [t for t in yband if _mid(t.get("y0", 0.0), t.get("y1", 0.0)) >= start_y]

    def tokens_in_band(name: str) -> List[dict]:
        xL, xR = bands_use.get(name, (None, None))
        if xL is None:
            return []
        out: List[dict] = []
        for t in yband_after:
            mx = _mid(t.get("x0", 0.0), t.get("x1", 0.0))
            if xL <= mx <= xR:
                out.append(t)
        return out

    import re
    pat = re.compile(r"\b(30|60|90)\s*:\s*(\d+)\b")

    values: Dict[str, Dict[str, int]] = {b: {"30": 0, "60": 0, "90": 0} for b in _BUREAUS}
    found_any = False
    for bureau in _BUREAUS:
        toks = tokens_in_band(bureau)
        if not toks:
            continue
        # Join by lines to get text like "30: 0"
        lines_b = _cluster_lines_y(toks, dy=1.0)
        for line in lines_b:
            text = _join_text(line)
            for m in pat.finditer(text):
                key, val = m.group(1), m.group(2)
                try:
                    values[bureau][key] = int(val)
                    found_any = True
                except Exception:
                    pass

    if not found_any:
        return None

    out_obj = {
        "session_id": session_id,
        "block_id": int(block_id),
        "type": "seven_year_delinquency",
        "values": values,
    }
    out_path = out_dir / f"account_table_{block_id:02d}__{(heading or f'block-{block_id}').lower().replace(' ', '-').replace('/', '-')}.delinquency.json"
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)

