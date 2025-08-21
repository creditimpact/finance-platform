from __future__ import annotations

import json
import os
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Mapping

from backend.api import session_manager
from backend.outcomes import OutcomeEvent
from backend.outcomes.models import Outcome
from backend.analytics.analytics_tracker import emit_counter
from backend.core.logic.report_analysis.tri_merge import normalize_and_match
from backend.core.logic.report_analysis.tri_merge_models import Tradeline, TradelineFamily

from . import ingest


# Cache of normalized tradeline families keyed by a hash of the raw report. This
# avoids re-running canonicalization when the same report is ingested multiple
# times (e.g. retries).
_FAMILY_CACHE: "OrderedDict[str, List[TradelineFamily]]" = OrderedDict()
_CACHE_MAX = 32


def _extract_tradelines(report: Mapping[str, Any]) -> List[Tradeline]:
    """Flatten bureau payload into tradeline objects."""
    tradelines: List[Tradeline] = []
    for bureau, payload in report.items():
        if not isinstance(payload, Mapping):
            continue
        for section, items in payload.items():
            if section == "inquiries" or not isinstance(items, list):
                continue
            for acc in items:
                tradelines.append(
                    Tradeline(
                        creditor=str(acc.get("name") or ""),
                        bureau=bureau,
                        account_number=acc.get("account_number"),
                        data=acc,
                    )
                )
    return tradelines


def _snapshot_families(families: Iterable[TradelineFamily]) -> Dict[str, Dict[str, Any]]:
    """Return a lightweight snapshot mapping family_id to bureau data."""
    snapshot: Dict[str, Dict[str, Any]] = {}
    for fam in families:
        family_id = getattr(fam, "family_id", None)
        if not family_id:
            continue
        snap = {b: tl.data for b, tl in fam.tradelines.items()}
        snapshot[family_id] = snap
    return snapshot


def ingest_report(account_id: str | None, new_report: Mapping[str, Any]) -> List[OutcomeEvent]:
    """Ingest a new credit report and emit outcome events.

    Args:
        account_id: Optional account id. If ``None``, all accounts present in
            ``new_report`` are processed.
        new_report: Parsed bureau payload mapping bureau name to sections.
    Returns:
        List of :class:`OutcomeEvent` instances that were persisted.
    """

    session_id = os.getenv("SESSION_ID")
    if not session_id:
        return []

    tradelines = _extract_tradelines(new_report)

    # Use a stable hash of the report payload as the cache key.
    report_key = json.dumps(new_report, sort_keys=True, default=str)
    cache_key = str(uuid.uuid5(uuid.NAMESPACE_OID, report_key))
    families = _FAMILY_CACHE.get(cache_key)
    if families is None:
        families = normalize_and_match(tradelines)
        _FAMILY_CACHE[cache_key] = families
        # simple LRU eviction
        if len(_FAMILY_CACHE) > _CACHE_MAX:
            _FAMILY_CACHE.popitem(last=False)

    # Group family snapshots by account id
    per_account: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for fam in families:
        fam_id = getattr(fam, "family_id", None)
        if not fam_id:
            continue
        snap = {b: tl.data for b, tl in fam.tradelines.items()}
        for tl in fam.tradelines.values():
            acc_id = str(tl.data.get("account_id") or "")
            if not acc_id:
                continue
            per_account.setdefault(acc_id, {})[fam_id] = snap

    # If a specific account_id was provided, filter to it
    if account_id is not None:
        per_account = {k: v for k, v in per_account.items() if k == str(account_id)}

    stored = session_manager.get_session(session_id) or {}
    tri_merge = stored.get("tri_merge", {}) or {}
    prev_snapshots: Dict[str, Dict[str, Dict[str, Any]]] = tri_merge.get("snapshots", {}) or {}

    events: List[OutcomeEvent] = []

    def _compute_event(
        acc_id: str,
        fid: str,
        prev_snap: Dict[str, Dict[str, Any]],
        new_snap: Dict[str, Dict[str, Any]],
    ) -> OutcomeEvent | None:
        if fid not in new_snap:
            outcome = Outcome.DELETED
            diff = {"previous": prev_snap[fid], "current": None}
        elif fid not in prev_snap:
            outcome = Outcome.NOCHANGE
            diff = None
        elif prev_snap[fid] != new_snap[fid]:
            outcome = Outcome.UPDATED
            diff = {"previous": prev_snap[fid], "current": new_snap[fid]}
        else:
            outcome = Outcome.VERIFIED
            diff = None
        emit_counter(f"outcome.{outcome.name.lower()}")
        event = OutcomeEvent(
            outcome_id=str(uuid.uuid4()),
            account_id=acc_id,
            cycle_id=0,
            family_id=fid,
            outcome=outcome,
            diff_snapshot=diff,
        )
        ingest({"session_id": session_id}, event)
        return event

    futures = []
    with ThreadPoolExecutor() as ex:
        for acc_id, new_snap in per_account.items():
            prev_snap = prev_snapshots.get(acc_id, {})
            if not prev_snap:
                prev_snapshots[acc_id] = new_snap
                continue
            all_fids = set(prev_snap) | set(new_snap)
            for fid in all_fids:
                futures.append(ex.submit(_compute_event, acc_id, fid, prev_snap, new_snap))
            prev_snapshots[acc_id] = new_snap

        for fut in futures:
            evt = fut.result()
            if evt:
                events.append(evt)

    # Persist updated snapshots
    tri_merge["snapshots"] = prev_snapshots
    session_manager.update_session(session_id, tri_merge=tri_merge)

    return events
