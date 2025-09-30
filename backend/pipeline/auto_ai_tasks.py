"""Celery task chain used by the automatic AI adjudication pipeline."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, MutableMapping

from celery import chain, shared_task

from backend.core.ai.paths import ensure_merge_paths, probe_legacy_ai_packs
from backend.core.logic import polarity
from backend.pipeline.auto_ai import (
    INFLIGHT_LOCK_FILENAME,
    LAST_OK_FILENAME,
    _build_ai_packs,
    _compact_accounts,
    _indices_from_index,
    _load_ai_index,
    _normalize_indices,
    _send_ai_packs,
    has_ai_merge_best_pairs,
    run_validation_requirements_for_all_accounts,
)
from backend.core.ai.validators import validate_ai_result
from backend.core.io.tags import read_tags, upsert_tag
from scripts.score_bureau_pairs import score_accounts

LEGACY_PIPELINE_DIRNAME = "ai_packs"
LEGACY_MARKER_FILENAME = "auto_ai_pipeline_in_progress.json"

logger = logging.getLogger(__name__)


_PAIR_TAG_BY_DECISION: dict[str, str] = {
    "same_account_same_debt": "same_account_pair",
    "same_account_diff_debt": "same_account_pair",
    "same_account_debt_unknown": "same_account_pair",
    "same_debt_diff_account": "same_debt_pair",
    "same_debt_account_unknown": "same_debt_pair",
}

_RESULT_FILENAME_PATTERN = re.compile(r"^pair_(\d{3})_(\d{3})\.result\.json$")


def _isoformat_timestamp(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)
    return current.isoformat(timespec="seconds").replace("+00:00", "Z")


def _serialize_match_flag(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false", "unknown"}:
            return lowered
    return "unknown"


def _load_ai_results(results_dir: Path) -> list[tuple[int, int, dict[str, object]]]:
    if not results_dir.exists():
        return []

    pairs: list[tuple[int, int, dict[str, object]]] = []
    for path in sorted(results_dir.glob("pair_*.result.json")):
        match = _RESULT_FILENAME_PATTERN.match(path.name)
        if not match:
            logger.debug("AUTO_AI_RESULT_SKIP_UNMATCHED path=%s", path)
            continue

        try:
            a_idx = int(match.group(1))
            b_idx = int(match.group(2))
        except (TypeError, ValueError):
            logger.debug("AUTO_AI_RESULT_SKIP_PARSE path=%s", path)
            continue

        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("AUTO_AI_RESULT_INVALID_JSON path=%s", path, exc_info=True)
            continue

        if not isinstance(loaded, Mapping):
            logger.warning("AUTO_AI_RESULT_INVALID_TYPE path=%s", path)
            continue

        payload = dict(loaded)
        flags_obj = payload.get("flags")
        if not isinstance(flags_obj, Mapping):
            logger.warning("AUTO_AI_RESULT_MISSING_FLAGS path=%s", path)
            continue

        try:
            valid, error = validate_ai_result(
                {"decision": payload.get("decision"), "reason": payload.get("reason"), "flags": dict(flags_obj)}
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.warning("AUTO_AI_RESULT_VALIDATION_ERROR path=%s", path, exc_info=True)
            continue

        if not valid:
            logger.warning(
                "AUTO_AI_RESULT_INVALID sid_pair=%s_%s path=%s error=%s",
                a_idx,
                b_idx,
                path,
                error or "unknown",
            )
            continue

        payload["flags"] = dict(flags_obj)
        pairs.append((a_idx, b_idx, payload))

    return pairs


def _prune_pair_tags(tag_path: Path, other_idx: int, *, keep_kind: str | None) -> None:
    existing_tags = read_tags(tag_path)
    if not existing_tags:
        return

    filtered: list[dict[str, object]] = []
    modified = False
    for entry in existing_tags:
        kind = str(entry.get("kind", "")).strip().lower()
        if kind not in {"same_account_pair", "same_debt_pair"}:
            filtered.append(dict(entry))
            continue

        source = str(entry.get("source", ""))
        if source != "ai_adjudicator":
            filtered.append(dict(entry))
            continue

        partner_raw = entry.get("with")
        try:
            partner_val = int(partner_raw) if partner_raw is not None else None
        except (TypeError, ValueError):
            partner_val = None

        if partner_val != other_idx:
            filtered.append(dict(entry))
            continue

        if keep_kind is not None and kind == keep_kind:
            filtered.append(dict(entry))
            continue

        modified = True

    if not modified:
        return

    tag_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _apply_ai_result_to_accounts(
    accounts_dir: Path, a_idx: int, b_idx: int, payload: Mapping[str, object]
) -> None:
    decision_raw = payload.get("decision")
    decision = str(decision_raw).strip().lower() if isinstance(decision_raw, str) else ""
    reason_raw = payload.get("reason")
    if isinstance(reason_raw, str):
        reason = reason_raw.strip()
    elif reason_raw is None:
        reason = ""
    else:
        reason = str(reason_raw)

    flags_raw = payload.get("flags")
    flags_serialized: dict[str, str] | None = None
    if isinstance(flags_raw, Mapping):
        flags_serialized = {
            "account_match": _serialize_match_flag(flags_raw.get("account_match")),
            "debt_match": _serialize_match_flag(flags_raw.get("debt_match")),
        }

    timestamp = _isoformat_timestamp()
    pair_tag_kind = _PAIR_TAG_BY_DECISION.get(decision)

    normalized_raw = payload.get("normalized")
    normalized_flag = normalized_raw if isinstance(normalized_raw, bool) else None
    raw_response = payload.get("raw_response") if isinstance(payload.get("raw_response"), Mapping) else None

    for source_idx, other_idx in ((a_idx, b_idx), (b_idx, a_idx)):
        account_dir = accounts_dir / f"{source_idx}"
        account_dir.mkdir(parents=True, exist_ok=True)
        tag_path = account_dir / "tags.json"

        decision_tag: dict[str, object] = {
            "kind": "ai_decision",
            "tag": "ai_decision",
            "source": "ai_adjudicator",
            "with": other_idx,
            "decision": decision,
            "reason": reason,
            "at": timestamp,
        }
        if flags_serialized is not None:
            decision_tag["flags"] = dict(flags_serialized)
        if normalized_flag is not None:
            decision_tag["normalized"] = normalized_flag
        if raw_response is not None:
            decision_tag["raw_response"] = dict(raw_response)

        upsert_tag(tag_path, decision_tag, unique_keys=("kind", "with", "source"))

        if pair_tag_kind is not None:
            pair_tag = {
                "kind": pair_tag_kind,
                "with": other_idx,
                "source": "ai_adjudicator",
                "reason": reason,
                "at": timestamp,
            }
            upsert_tag(tag_path, pair_tag, unique_keys=("kind", "with", "source"))
            _prune_pair_tags(tag_path, other_idx, keep_kind=pair_tag_kind)
        else:
            _prune_pair_tags(tag_path, other_idx, keep_kind=None)


def _append_run_log_entry(
    *,
    runs_root: Path,
    sid: str,
    packs: int,
    pairs: int,
    reason: str | None = None,
) -> None:
    """Append a compact JSON line describing the AI run outcome."""

    merge_paths = ensure_merge_paths(runs_root, sid, create=True)
    logs_path = merge_paths.log_file
    entry = {
        "sid": sid,
        "at": datetime.now(timezone.utc).isoformat(),
        "packs": int(packs),
        "pairs": int(pairs),
        "keywords": [
            "CANDIDATE_LOOP_START",
            "CANDIDATE_CONSIDERED",
            "CANDIDATE_SKIPPED",
            "CANDIDATE_LOOP_END",
            "MERGE_V2_ACCT_BEST",
        ],
        "verify": [
            f"rg \"CANDIDATE_(CONSIDERED|SKIPPED)\" {logs_path}",
            f"rg \"MERGE_V2_ACCT_BEST\" {logs_path}",
        ],
    }
    if reason:
        entry["reason"] = reason

    serialized_entry = json.dumps(entry, ensure_ascii=False) + "\n"

    try:
        logs_path.parent.mkdir(parents=True, exist_ok=True)
        with logs_path.open("a", encoding="utf-8") as handle:
            handle.write(serialized_entry)
    except Exception:  # pragma: no cover - defensive logging
        logger.warning(
            "AUTO_AI_LOG_APPEND_FAILED sid=%s path=%s", sid, logs_path, exc_info=True
        )

    legacy_logs_path = runs_root / sid / "ai_packs" / "logs.txt"
    if legacy_logs_path != logs_path:
        try:
            legacy_logs_path.parent.mkdir(parents=True, exist_ok=True)
            with legacy_logs_path.open("a", encoding="utf-8") as handle:
                handle.write(serialized_entry)
        except Exception:  # pragma: no cover - defensive logging
            logger.debug(
                "AUTO_AI_LOG_LEGACY_WRITE_FAILED sid=%s path=%s",
                sid,
                legacy_logs_path,
                exc_info=True,
            )



def _ensure_payload(prev: Mapping[str, object] | None) -> dict[str, object]:
    if isinstance(prev, Mapping):
        return dict(prev)
    return {}


def _resolve_runs_root(payload: Mapping[str, object], sid: str) -> Path:
    runs_root_value = payload.get("runs_root")
    if isinstance(runs_root_value, (str, os.PathLike)):
        return Path(runs_root_value)

    env_root = os.environ.get("RUNS_ROOT")
    if env_root:
        return Path(env_root)

    default_root = Path("runs")

    pipeline_dir = ensure_merge_paths(default_root, sid, create=False).base
    lock_path = pipeline_dir / INFLIGHT_LOCK_FILENAME
    if lock_path.exists():
        try:
            data = json.loads(lock_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.debug(
                "AUTO_AI_LOCK_READ_FAILED sid=%s path=%s", sid, lock_path, exc_info=True
            )
        else:
            marker_root = data.get("runs_root")
            if isinstance(marker_root, str) and marker_root:
                return Path(marker_root)
        return pipeline_dir.parent.parent

    legacy_dir = default_root / sid / LEGACY_PIPELINE_DIRNAME
    legacy_marker = legacy_dir / LEGACY_MARKER_FILENAME
    if legacy_marker.exists():
        try:
            data = json.loads(legacy_marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.debug(
                "AUTO_AI_LEGACY_MARKER_READ_FAILED sid=%s path=%s",
                sid,
                legacy_marker,
                exc_info=True,
            )
        else:
            marker_root = data.get("runs_root")
            if isinstance(marker_root, str) and marker_root:
                return Path(marker_root)
        return legacy_dir.parent.parent

    return default_root


def _populate_common_paths(payload: MutableMapping[str, object]) -> None:
    sid = str(payload.get("sid") or "")
    if not sid:
        return

    runs_root = _resolve_runs_root(payload, sid)
    accounts_dir = runs_root / sid / "cases" / "accounts"
    merge_paths = ensure_merge_paths(runs_root, sid, create=True)
    pipeline_dir = merge_paths.base
    lock_path = pipeline_dir / INFLIGHT_LOCK_FILENAME
    last_ok_path = pipeline_dir / LAST_OK_FILENAME

    payload["runs_root"] = str(runs_root)
    payload["accounts_dir"] = str(accounts_dir)
    payload["pipeline_dir"] = str(pipeline_dir)
    payload["lock_path"] = str(lock_path)
    payload["marker_path"] = str(lock_path)
    payload["last_ok_path"] = str(last_ok_path)


def _cleanup_lock(payload: Mapping[str, object], *, reason: str) -> bool:
    sid = str(payload.get("sid") or "")
    lock_value = payload.get("lock_path") or payload.get("marker_path")
    if not lock_value:
        return False

    lock_path = Path(str(lock_value))
    try:
        if lock_path.exists():
            lock_path.unlink()
            logger.info(
                "AUTO_AI_LOCK_REMOVED sid=%s reason=%s lock=%s",
                sid,
                reason,
                lock_path,
            )
            return True
        logger.info(
            "AUTO_AI_LOCK_MISSING sid=%s reason=%s lock=%s",
            sid,
            reason,
            lock_path,
        )
        return False
    except Exception:  # pragma: no cover - defensive logging
        logger.warning(
            "AUTO_AI_LOCK_CLEANUP_FAILED sid=%s reason=%s lock=%s",
            sid,
            reason,
            lock_path,
            exc_info=True,
        )
        return False


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def ai_score_step(self, sid: str, runs_root: str | None = None) -> dict[str, object]:
    """Recompute merge scores and persist merge tags for ``sid``."""

    logger.info("AI_SCORE_START sid=%s", sid)

    payload: dict[str, object] = {"sid": sid}
    if runs_root is not None:
        payload["runs_root"] = runs_root
    _populate_common_paths(payload)

    runs_root = Path(payload["runs_root"])

    try:
        result = score_accounts(sid, runs_root=runs_root, write_tags=True)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_SCORE_FAILED sid=%s", sid, exc_info=True)
        _cleanup_lock(payload, reason="score_failed")
        raise

    touched_accounts = sorted(_normalize_indices(result.indices))
    payload["touched_accounts"] = touched_accounts

    logger.info("AI_SCORE_END sid=%s touched=%d", sid, len(touched_accounts))
    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def ai_build_packs_step(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    """Build AI merge packs for accounts requiring AI decisions."""

    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("AUTO_AI_BUILD_SKIP payload=%s", payload)
        return payload

    _populate_common_paths(payload)
    runs_root = Path(payload["runs_root"])

    logger.info("AI_BUILD_START sid=%s", sid)

    if not has_ai_merge_best_pairs(sid, runs_root):
        logger.info("AUTO_AI_SKIP_NO_CANDIDATES sid=%s", sid)
        logger.info("AUTO_AI_BUILDER_BYPASSED_ZERO_DEBT sid=%s", sid)
        logger.info("AUTO_AI_SKIPPED sid=%s reason=no_candidates", sid)
        payload["ai_index"] = []
        payload["skip_reason"] = "no_candidates"
        return payload

    try:
        _build_ai_packs(sid, runs_root)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_BUILD_FAILED sid=%s", sid, exc_info=True)
        _cleanup_lock(payload, reason="build_failed")
        raise

    merge_paths = ensure_merge_paths(runs_root, sid, create=True)
    index_path = merge_paths.index_file
    if not index_path.exists():
        legacy_dir = probe_legacy_ai_packs(runs_root, sid)
        if legacy_dir is not None:
            legacy_index = legacy_dir / "index.json"
            if legacy_index.exists():
                index_path = legacy_index
    try:
        index_entries = _load_ai_index(index_path)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_BUILD_INVALID_INDEX sid=%s path=%s", sid, index_path, exc_info=True)
        _cleanup_lock(payload, reason="build_invalid_index")
        raise

    payload["ai_index"] = index_entries

    touched: set[int] = set(_normalize_indices(payload.get("touched_accounts", [])))
    touched.update(_indices_from_index(index_entries))
    payload["touched_accounts"] = sorted(touched)

    logger.info("AI_PACKS_INDEX sid=%s path=%s count=%d", sid, index_path, len(index_entries))
    logger.info("AI_BUILD_END sid=%s packs=%d", sid, len(index_entries))
    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def ai_send_packs_step(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    """Send AI merge packs for adjudication and persist AI decision tags."""

    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("AUTO_AI_SEND_SKIP payload=%s", payload)
        return payload

    _populate_common_paths(payload)
    runs_root = Path(payload["runs_root"])

    logger.info("AI_SEND_START sid=%s", sid)

    index_entries = payload.get("ai_index")
    if not index_entries:
        reason = str(payload.get("skip_reason") or "no_packs")
        logger.info("AI_SEND_SKIP sid=%s reason=%s", sid, reason)
        return payload

    try:
        _send_ai_packs(sid, runs_root=runs_root)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_SEND_FAILED sid=%s", sid, exc_info=True)
        _cleanup_lock(payload, reason="send_failed")
        raise

    logger.info("AI_SEND_END sid=%s", sid)
    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def ai_validation_requirements_step(
    self, prev: Mapping[str, object] | None
) -> dict[str, object]:
    """Populate validation requirements after AI adjudication results."""

    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("AUTO_AI_VALIDATION_SKIP payload=%s", payload)
        return payload

    _populate_common_paths(payload)

    runs_root_value = payload.get("runs_root")
    runs_root_path = Path(str(runs_root_value)) if runs_root_value else None

    logger.info("AI_VALIDATION_REQUIREMENTS_START sid=%s", sid)

    try:
        stats = run_validation_requirements_for_all_accounts(
            sid, runs_root=runs_root_path
        )
    except Exception:  # pragma: no cover - defensive logging
        logger.error(
            "AI_VALIDATION_REQUIREMENTS_FAILED sid=%s", sid, exc_info=True
        )
        _cleanup_lock(payload, reason="validation_requirements_failed")
        raise

    payload["validation_requirements"] = stats

    logger.info(
        "AI_VALIDATION_REQUIREMENTS_END sid=%s processed=%d requirements=%d",
        sid,
        stats.get("processed_accounts", 0),
        stats.get("requirements", 0),
    )

    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def ai_compact_tags_step(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    """Compact tags and summaries for accounts touched by the AI pipeline."""

    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("AUTO_AI_COMPACT_SKIP payload=%s", payload)
        return payload

    _populate_common_paths(payload)

    accounts_dir_value = payload.get("accounts_dir")
    accounts_dir = Path(str(accounts_dir_value)) if accounts_dir_value else None
    runs_root_value = payload.get("runs_root")
    runs_root = Path(str(runs_root_value)) if runs_root_value else None

    indices_set: set[int] = set(_normalize_indices(payload.get("touched_accounts", [])))

    result_pairs: list[tuple[int, int, dict[str, object]]] = []
    if runs_root is not None:
        try:
            merge_paths = ensure_merge_paths(runs_root, sid, create=True)
        except Exception:  # pragma: no cover - defensive logging
            logger.error(
                "AUTO_AI_RESULTS_PATH_FAILED sid=%s runs_root=%s", sid, runs_root, exc_info=True
            )
        else:
            result_pairs = _load_ai_results(merge_paths.results_dir)

    if result_pairs and accounts_dir is None:
        logger.warning("AUTO_AI_RESULTS_NO_ACCOUNTS_DIR sid=%s", sid)

    if result_pairs and accounts_dir is not None:
        for a_idx, b_idx, result_payload in result_pairs:
            _apply_ai_result_to_accounts(accounts_dir, a_idx, b_idx, result_payload)
            indices_set.add(int(a_idx))
            indices_set.add(int(b_idx))
        logger.info("AI_RESULTS_APPLIED sid=%s count=%d", sid, len(result_pairs))

    indices = sorted(indices_set)
    payload["touched_accounts"] = indices

    logger.info("AI_COMPACT_START sid=%s accounts=%d", sid, len(indices))

    if accounts_dir and accounts_dir.exists() and indices:
        try:
            _compact_accounts(accounts_dir, indices)
        except Exception:  # pragma: no cover - defensive logging
            logger.error(
                "AUTO_AI_COMPACT_FAILED sid=%s dir=%s", sid, accounts_dir, exc_info=True
            )
            _cleanup_lock(payload, reason="compact_failed")
            raise

    logger.info("AI_COMPACT_END sid=%s", sid)

    packs_count = len(payload.get("ai_index", []) or [])
    pairs_count = len(indices)
    payload["packs"] = packs_count
    payload["pairs"] = pairs_count

    last_ok_value = payload.get("last_ok_path")
    if last_ok_value:
        last_ok_path = Path(str(last_ok_value))
        last_ok_payload = {
            "sid": sid,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "packs": packs_count,
            "pairs": pairs_count,
        }
        try:
            last_ok_path.write_text(
                json.dumps(last_ok_payload, ensure_ascii=False), encoding="utf-8"
            )
            logger.info("AUTO_AI_LAST_OK sid=%s path=%s", sid, last_ok_path)
        except Exception:  # pragma: no cover - defensive logging
            logger.warning(
                "AUTO_AI_LAST_OK_WRITE_FAILED sid=%s path=%s",
                sid,
                last_ok_path,
                exc_info=True,
            )

    runs_root_value = payload.get("runs_root")
    if runs_root_value:
        try:
            runs_root_path = Path(str(runs_root_value))
        except Exception:  # pragma: no cover - defensive logging
            logger.debug(
                "AUTO_AI_LOG_ROOT_INVALID sid=%s runs_root=%r",
                sid,
                runs_root_value,
                exc_info=True,
            )
        else:
            skip_reason = payload.get("skip_reason")
            reason_text = str(skip_reason) if isinstance(skip_reason, str) else None
            _append_run_log_entry(
                runs_root=runs_root_path,
                sid=sid,
                packs=packs_count,
                pairs=pairs_count,
                reason=reason_text,
            )

    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def ai_polarity_check_step(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    """Run polarity classification after AI adjudication outputs are compacted."""

    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("AUTO_AI_POLARITY_SKIP payload=%s", payload)
        return payload

    _populate_common_paths(payload)

    accounts_dir_value = payload.get("accounts_dir")
    accounts_dir = Path(str(accounts_dir_value)) if accounts_dir_value else None
    if accounts_dir is None:
        logger.info("AUTO_AI_POLARITY_SKIP sid=%s reason=no_accounts_dir", sid)
        removed = _cleanup_lock(payload, reason="polarity_missing_accounts_dir")
        logger.info(
            "AUTO_AI_CHAIN_END sid=%s lock_removed=%s packs=%s pairs=%s",
            sid,
            1 if removed else 0,
            payload.get("packs"),
            payload.get("pairs"),
        )
        return payload

    indices = sorted(_normalize_indices(payload.get("touched_accounts", [])))
    if not indices:
        logger.info("AUTO_AI_POLARITY_SKIP sid=%s reason=no_indices", sid)
        removed = _cleanup_lock(payload, reason="polarity_no_indices")
        logger.info(
            "AUTO_AI_CHAIN_END sid=%s lock_removed=%s packs=%s pairs=%s",
            sid,
            1 if removed else 0,
            payload.get("packs"),
            payload.get("pairs"),
        )
        return payload

    logger.info("AI_POLARITY_START sid=%s accounts=%d", sid, len(indices))

    try:
        result = polarity.apply_polarity_checks(accounts_dir, indices, sid=sid)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_POLARITY_FAILED sid=%s dir=%s", sid, accounts_dir, exc_info=True)
        _cleanup_lock(payload, reason="polarity_failed")
        raise

    payload["polarity_processed"] = result.processed_accounts
    payload["polarity_updated"] = result.updated_accounts
    if result.config_digest:
        payload["polarity_config_digest"] = result.config_digest

    logger.info(
        "AI_POLARITY_END sid=%s processed=%d updated=%d",
        sid,
        result.processed_accounts,
        len(result.updated_accounts),
    )

    removed = _cleanup_lock(payload, reason="chain_complete")
    logger.info(
        "AUTO_AI_CHAIN_END sid=%s lock_removed=%s packs=%s pairs=%s polarity=%d",
        sid,
        1 if removed else 0,
        payload.get("packs"),
        payload.get("pairs"),
        result.processed_accounts,
    )

    return payload


def enqueue_auto_ai_chain(sid: str, runs_root: Path | str | None = None) -> str:
    """Queue the AI adjudication Celery chain and return the root task id."""

    runs_root_value = str(runs_root) if runs_root is not None else None

    logger.info("AUTO_AI_CHAIN_START sid=%s runs_root=%s", sid, runs_root_value)

    workflow = chain(
        ai_score_step.s(sid, runs_root_value),
        ai_build_packs_step.s(),
        ai_send_packs_step.s(),
        ai_validation_requirements_step.s(),
        ai_compact_tags_step.s(),
        ai_polarity_check_step.s(),
    )

    result = workflow.apply_async()
    task_id = str(result.id)

    logger.info(
        "AUTO_AI_CHAIN_ENQUEUED sid=%s task_id=%s runs_root=%s",
        sid,
        task_id,
        runs_root_value,
    )
    return task_id

