"""Celery task chain used by the automatic AI adjudication pipeline."""

from __future__ import annotations

import json
import logging
import os
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, MutableMapping

from celery import chain, shared_task

from backend.ai.manifest import ensure_validation_section
from backend.ai.validation_builder import build_validation_packs_for_run
from backend.core.ai.paths import (
    ensure_merge_paths,
    probe_legacy_ai_packs,
    validation_index_path,
)
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
    run_consistency_writeback_for_all_accounts,
    run_validation_requirements_for_all_accounts,
)
from backend.core.runflow import runflow_step
from backend.core.runflow.io import (
    runflow_stage_end,
    runflow_stage_error,
    runflow_stage_start,
)
from backend.runflow.decider import StageStatus, decide_next, record_stage
from backend.frontend.packs.generator import generate_frontend_packs_for_run
from backend.runflow.manifest import (
    update_manifest_frontend,
    update_manifest_state,
)
from backend.prevalidation.tasks import (
    detect_and_persist_date_convention,
    run_date_convention_detector,
)
from backend.core.ai.validators import validate_ai_result
from backend.core.io.tags import read_tags, upsert_tag
from backend.validation.manifest import rewrite_index_to_canonical_layout
from backend.validation.send_packs import send_validation_packs
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


def _merge_build_stage(payload: dict[str, object]) -> dict[str, object]:
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("AUTO_AI_BUILD_SKIP payload=%s", payload)
        return payload

    _populate_common_paths(payload)
    runs_root = Path(payload["runs_root"])

    logger.info("AI_BUILD_START sid=%s", sid)

    runflow_stage_start("merge", sid=sid)

    if not has_ai_merge_best_pairs(sid, runs_root):
        logger.info("AUTO_AI_SKIP_NO_CANDIDATES sid=%s", sid)
        logger.info("AUTO_AI_BUILDER_BYPASSED_ZERO_DEBT sid=%s", sid)
        logger.info("AUTO_AI_SKIPPED sid=%s reason=no_candidates", sid)
        payload["ai_index"] = []
        payload["skip_reason"] = "no_candidates"
        runflow_step(
            sid,
            "merge",
            "build",
            status="skipped",
            metrics={"packs": 0},
            out={"reason": "no_candidates"},
        )
        return payload

    try:
        _build_ai_packs(sid, runs_root)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_BUILD_FAILED sid=%s", sid, exc_info=True)
        _cleanup_lock(payload, reason="build_failed")
        runflow_step(
            sid,
            "merge",
            "build",
            status="error",
            out={"error": exc.__class__.__name__, "msg": str(exc)},
        )
        runflow_stage_error(
            "merge",
            sid=sid,
            error_type=exc.__class__.__name__,
            message=str(exc),
            traceback_tail=traceback.format_exc(),
            hint="merge build",
            summary={"phase": "build"},
        )
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
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(
            "AUTO_AI_BUILD_INVALID_INDEX sid=%s path=%s", sid, index_path, exc_info=True
        )
        _cleanup_lock(payload, reason="build_invalid_index")
        runflow_step(
            sid,
            "merge",
            "build",
            status="error",
            out={"error": exc.__class__.__name__, "msg": str(exc)},
        )
        runflow_stage_error(
            "merge",
            sid=sid,
            error_type=exc.__class__.__name__,
            message=str(exc),
            traceback_tail=traceback.format_exc(),
            hint="merge build",
            summary={"phase": "build"},
        )
        raise

    payload["ai_index"] = index_entries

    touched: set[int] = set(_normalize_indices(payload.get("touched_accounts", [])))
    touched.update(_indices_from_index(index_entries))
    payload["touched_accounts"] = sorted(touched)

    logger.info("AI_PACKS_INDEX sid=%s path=%s count=%d", sid, index_path, len(index_entries))
    logger.info("AI_BUILD_END sid=%s packs=%d", sid, len(index_entries))

    runflow_step(
        sid,
        "merge",
        "build",
        metrics={
            "packs": len(index_entries),
            "touched_accounts": len(touched),
        },
    )
    return payload


def _merge_send_stage(payload: dict[str, object]) -> dict[str, object]:
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
        runflow_step(
            sid,
            "merge",
            "send",
            status="skipped",
            out={"reason": reason},
        )
        return payload

    try:
        _send_ai_packs(sid, runs_root=runs_root)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_SEND_FAILED sid=%s", sid, exc_info=True)
        _cleanup_lock(payload, reason="send_failed")
        runflow_step(
            sid,
            "merge",
            "send",
            status="error",
            out={"error": exc.__class__.__name__, "msg": str(exc)},
        )
        runflow_stage_error(
            "merge",
            sid=sid,
            error_type=exc.__class__.__name__,
            message=str(exc),
            traceback_tail=traceback.format_exc(),
            hint="merge send",
            summary={"phase": "send"},
        )
        raise

    logger.info("AI_SEND_END sid=%s", sid)
    runflow_step(
        sid,
        "merge",
        "send",
        metrics={"packs": len(index_entries)},
    )
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

    detection_block: dict[str, object] | None = None
    try:
        detection_block = detect_and_persist_date_convention(sid, runs_root=runs_root_path)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AI_DATE_CONVENTION_FAILED sid=%s", sid, exc_info=True)

    try:
        stats = run_validation_requirements_for_all_accounts(
            sid, runs_root=runs_root_path
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(
            "AI_VALIDATION_REQUIREMENTS_FAILED sid=%s", sid, exc_info=True
        )
        _cleanup_lock(payload, reason="validation_requirements_failed")
        runflow_step(
            sid,
            "validation",
            "requirements",
            status="error",
            out={"error": exc.__class__.__name__, "msg": str(exc)},
        )
        raise

    payload["validation_requirements"] = stats
    if isinstance(detection_block, dict):
        payload["date_convention"] = dict(detection_block)

    logger.info(
        "AI_VALIDATION_REQUIREMENTS_END sid=%s processed=%d findings=%d",
        sid,
        stats.get("processed_accounts", 0),
        stats.get("findings", 0),
    )

    stage_status: StageStatus = "success" if stats.get("ok", True) else "error"
    findings_count = int(stats.get("findings_count", stats.get("findings", 0)) or 0)
    empty_ok = findings_count == 0
    notes_value = stats.get("notes")

    record_stage(
        sid,
        "validation",
        status=stage_status,
        counts={"findings_count": findings_count},
        empty_ok=empty_ok,
        notes=notes_value,
        runs_root=runs_root_path,
    )

    decision = decide_next(sid, runs_root=runs_root_path)
    next_action = decision.get("next")

    if next_action in {"gen_frontend_packs", "complete_no_action"}:
        try:
            fe_result = generate_frontend_packs_for_run(
                sid, runs_root=runs_root_path
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.error("AUTO_AI_FRONTEND_PACKS_FAILED sid=%s", sid, exc_info=True)
            record_stage(
                sid,
                "frontend",
                status="error",
                counts={"packs_count": 0},
                empty_ok=True,
                notes="generation_failed",
                runs_root=runs_root_path,
            )
            decision = decide_next(sid, runs_root=runs_root_path)
            next_action = decision.get("next")
        else:
            packs_count = int(fe_result.get("packs_count", 0) or 0)
            frontend_status_value = str(fe_result.get("status") or "success")
            frontend_stage_status: StageStatus = (
                "success" if frontend_status_value != "error" else "error"
            )
            notes_override = (
                frontend_status_value if frontend_status_value not in {"", "success"} else None
            )

            record_stage(
                sid,
                "frontend",
                status=frontend_stage_status,
                counts={"packs_count": packs_count},
                empty_ok=packs_count == 0,
                notes=notes_override,
                runs_root=runs_root_path,
            )

            if frontend_stage_status == "success":
                update_manifest_frontend(
                    sid,
                    packs_dir=fe_result.get("packs_dir"),
                    packs_count=packs_count,
                    built=bool(fe_result.get("built", False)),
                    last_built_at=fe_result.get("last_built_at"),
                    runs_root=runs_root_path,
                )

            decision = decide_next(sid, runs_root=runs_root_path)
            next_action = decision.get("next")

    final_next = decision.get("next") if next_action is None else next_action
    if final_next == "await_input":
        update_manifest_state(
            sid,
            "AWAITING_CUSTOMER_INPUT",
            runs_root=runs_root_path,
        )
    elif final_next == "complete_no_action":
        update_manifest_state(
            sid,
            "COMPLETE_NO_ACTION",
            runs_root=runs_root_path,
        )
    elif final_next == "stop_error":
        update_manifest_state(
            sid,
            "ERROR",
            runs_root=runs_root_path,
        )

    return payload


def _merge_compact_stage(payload: dict[str, object]) -> dict[str, object]:
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
    merge_paths = None
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
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "AUTO_AI_COMPACT_FAILED sid=%s dir=%s", sid, accounts_dir, exc_info=True
            )
            _cleanup_lock(payload, reason="compact_failed")
            runflow_step(
                sid,
                "merge",
                "compact",
                status="error",
                metrics={"accounts": len(indices)},
                out={"error": exc.__class__.__name__, "msg": str(exc)},
            )
            runflow_stage_error(
                "merge",
                sid=sid,
                error_type=exc.__class__.__name__,
                message=str(exc),
                traceback_tail=traceback.format_exc(),
                hint="merge compact",
                summary={"phase": "compact"},
            )
            raise

    logger.info("AI_COMPACT_END sid=%s", sid)

    packs_count = len(payload.get("ai_index", []) or [])
    pairs_count = len(indices)
    payload["packs"] = packs_count
    payload["created_packs"] = packs_count
    payload["pairs"] = pairs_count

    logger.info("MERGE_STAGE_DONE sid=%s", sid)
    runflow_step(
        sid,
        "merge",
        "compact",
        metrics={"packs": packs_count, "pairs": pairs_count},
    )

    scored_pairs_value = 0
    if merge_paths is not None:
        try:
            pairs_index_payload = json.loads(
                (merge_paths.base / "pairs_index.json").read_text(encoding="utf-8")
            )
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            scored_pairs_value = 0
        else:
            totals = pairs_index_payload.get("totals") if isinstance(pairs_index_payload, Mapping) else None
            if isinstance(totals, Mapping):
                try:
                    scored_pairs_value = int(totals.get("scored_pairs", 0) or 0)
                except (TypeError, ValueError):
                    scored_pairs_value = 0

    payload["merge_scored_pairs"] = scored_pairs_value
    return payload


def _finalize_stage(payload: dict[str, object]) -> dict[str, object]:
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("AUTO_AI_FINALIZE_SKIP payload=%s", payload)
        return payload

    last_ok_value = payload.get("last_ok_path")
    if last_ok_value:
        last_ok_path = Path(str(last_ok_value))
        last_ok_payload = {
            "sid": sid,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "packs": payload.get("packs", 0),
            "pairs": payload.get("pairs", 0),
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

    runs_root_path: Path | None = None
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
                packs=int(payload.get("packs", 0)),
                pairs=int(payload.get("pairs", 0)),
                reason=reason_text,
            )

    removed = _cleanup_lock(payload, reason="chain_complete")
    logger.info(
        "AUTO_AI_CHAIN_END sid=%s lock_removed=%s packs=%s pairs=%s polarity=%s",
        sid,
        1 if removed else 0,
        payload.get("packs"),
        payload.get("pairs"),
        payload.get("polarity_processed"),
    )

    disk_created_packs: int | None = None
    if runs_root_path is not None:
        try:
            merge_paths = ensure_merge_paths(runs_root_path, sid, create=False)
        except Exception:  # pragma: no cover - defensive logging
            logger.debug(
                "AUTO_AI_PACK_COUNT_FAILED sid=%s runs_root=%s",
                sid,
                runs_root_path,
                exc_info=True,
            )
        else:
            packs_dir = merge_paths.packs_dir
            if packs_dir.is_dir():
                disk_created_packs = sum(
                    1 for entry in packs_dir.glob("*.jsonl") if entry.is_file()
                )
            else:
                disk_created_packs = 0

    created_packs_value = (
        disk_created_packs
        if disk_created_packs is not None
        else int(payload.get("created_packs", payload.get("packs", 0)) or 0)
    )
    skip_reason = payload.get("skip_reason")
    scored_pairs_value = int(payload.get("merge_scored_pairs", 0) or 0)
    status_value = "success"
    if isinstance(skip_reason, str) and skip_reason:
        status_value = "skipped"

    empty_ok = False
    if status_value != "error":
        if created_packs_value == 0:
            empty_ok = True
        elif scored_pairs_value == 0:
            empty_ok = True

    summary_payload: dict[str, Any] = {
        "pairs_scored": scored_pairs_value,
        "packs_created": created_packs_value,
        "empty_ok": bool(empty_ok),
    }

    runflow_stage_end(
        "merge",
        sid=sid,
        status=status_value,
        summary=summary_payload,
        empty_ok=empty_ok,
    )

    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def merge_build_packs(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    payload = _ensure_payload(prev)
    return _merge_build_stage(payload)


ai_build_packs_step = merge_build_packs


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def merge_send(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    payload = _ensure_payload(prev)
    return _merge_send_stage(payload)


ai_send_packs_step = merge_send


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def merge_compact(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    payload = _ensure_payload(prev)
    return _merge_compact_stage(payload)


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def validation_build_packs(
    self, prev: Mapping[str, object] | None
) -> dict[str, object]:
    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("VALIDATION_BUILD_SKIP payload=%s", payload)
        return payload

    _populate_common_paths(payload)
    runs_root = Path(payload["runs_root"])

    ensure_validation_section(sid, runs_root=runs_root)
    logger.info("VALIDATION_STAGE_STARTED sid=%s", sid)

    try:
        results = build_validation_packs_for_run(sid, runs_root=runs_root)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("VALIDATION_BUILD_FAILED sid=%s", sid, exc_info=True)
        runflow_step(
            sid,
            "validation",
            "build",
            status="error",
            out={"error": exc.__class__.__name__, "msg": str(exc)},
        )
        runflow_stage_error(
            "validation",
            sid=sid,
            error_type=exc.__class__.__name__,
            message=str(exc),
            traceback_tail=traceback.format_exc(),
            hint="validation build",
            summary={"phase": "build"},
        )
        raise

    packs_written = sum(len(entries or []) for entries in results.values())
    payload["validation_packs"] = packs_written
    logger.info("VALIDATION_BUILD_DONE sid=%s packs=%d", sid, packs_written)
    runflow_step(
        sid,
        "validation",
        "build",
        metrics={"packs": packs_written},
    )
    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def validation_send(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("VALIDATION_SEND_SKIP payload=%s", payload)
        return payload

    _populate_common_paths(payload)
    runs_root = Path(payload["runs_root"])
    index_path = validation_index_path(sid, runs_root=runs_root, create=True)

    if not index_path.exists():
        logger.info(
            "VALIDATION_SEND_SKIP sid=%s reason=index_missing path=%s", sid, index_path
        )
        runflow_step(
            sid,
            "validation",
            "send",
            status="skipped",
            out={"reason": "index_missing"},
        )
        return payload

    logger.info("VALIDATION_SEND_START sid=%s path=%s", sid, index_path)

    try:
        send_validation_packs(index_path, stage="validation")
    except TypeError as exc:
        if "stage" not in str(exc):
            logger.error(
                "VALIDATION_SEND_FAILED sid=%s path=%s", sid, index_path, exc_info=True
            )
            runflow_step(
                sid,
                "validation",
                "send",
                status="error",
                out={"error": exc.__class__.__name__, "msg": str(exc)},
            )
            runflow_stage_error(
                "validation",
                sid=sid,
                error_type=exc.__class__.__name__,
                message=str(exc),
                traceback_tail=traceback.format_exc(),
                hint="validation send",
                summary={"phase": "send"},
            )
            raise
        send_validation_packs(index_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(
            "VALIDATION_SEND_FAILED sid=%s path=%s", sid, index_path, exc_info=True
        )
        runflow_step(
            sid,
            "validation",
            "send",
            status="error",
            out={"error": exc.__class__.__name__, "msg": str(exc)},
        )
        runflow_stage_error(
            "validation",
            sid=sid,
            error_type=exc.__class__.__name__,
            message=str(exc),
            traceback_tail=traceback.format_exc(),
            hint="validation send",
            summary={"phase": "send"},
        )
        raise

    payload["validation_sent"] = True
    logger.info("VALIDATION_SEND_DONE sid=%s", sid)
    runflow_step(
        sid,
        "validation",
        "send",
        metrics={"packs": int(payload.get("validation_packs", 0) or 0)},
    )
    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def validation_compact(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("VALIDATION_COMPACT_SKIP payload=%s", payload)
        return payload

    _populate_common_paths(payload)
    runs_root = Path(payload["runs_root"])
    index_path = validation_index_path(sid, runs_root=runs_root, create=True)

    if not index_path.exists():
        logger.info(
            "VALIDATION_COMPACT_SKIP sid=%s reason=index_missing path=%s", sid, index_path
        )
        runflow_step(
            sid,
            "validation",
            "compact",
            status="skipped",
            out={"reason": "index_missing"},
        )
        return payload

    logger.info("VALIDATION_COMPACT_START sid=%s", sid)

    try:
        rewrite_index_to_canonical_layout(index_path, runs_root=runs_root)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(
            "VALIDATION_COMPACT_FAILED sid=%s path=%s", sid, index_path, exc_info=True
        )
        runflow_step(
            sid,
            "validation",
            "compact",
            status="error",
            out={"error": exc.__class__.__name__, "msg": str(exc)},
        )
        runflow_stage_error(
            "validation",
            sid=sid,
            error_type=exc.__class__.__name__,
            message=str(exc),
            traceback_tail=traceback.format_exc(),
            hint="validation compact",
            summary={"phase": "compact"},
        )
        raise

    payload["validation_compacted"] = True
    logger.info("VALIDATION_COMPACT_DONE sid=%s", sid)
    runflow_step(
        sid,
        "validation",
        "compact",
        metrics={"compacted": True},
    )
    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def pipeline_finalize(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    payload = _ensure_payload(prev)
    return _finalize_stage(payload)


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def ai_compact_tags_step(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    """Backwards-compatible task that compacts merge data then finalizes the run."""

    payload = _ensure_payload(prev)
    payload = _merge_compact_stage(payload)
    return _finalize_stage(payload)


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
        return payload

    indices = sorted(_normalize_indices(payload.get("touched_accounts", [])))
    if not indices:
        logger.info("AUTO_AI_POLARITY_SKIP sid=%s reason=no_indices", sid)
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

    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def ai_consistency_step(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    """Persist field consistency snapshots for accounts touched by the AI flow."""

    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("AUTO_AI_CONSISTENCY_SKIP payload=%s", payload)
        return payload

    _populate_common_paths(payload)

    runs_root_value = payload.get("runs_root")
    runs_root_path = Path(str(runs_root_value)) if runs_root_value else None

    logger.info("AI_CONSISTENCY_START sid=%s", sid)

    try:
        stats = run_consistency_writeback_for_all_accounts(
            sid, runs_root=runs_root_path
        )
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AI_CONSISTENCY_FAILED sid=%s", sid, exc_info=True)
        _cleanup_lock(payload, reason="consistency_failed")
        raise

    payload["consistency"] = stats

    logger.info(
        "AI_CONSISTENCY_END sid=%s processed=%d fields=%d",
        sid,
        stats.get("processed_accounts", 0),
        stats.get("fields", 0),
    )

    return payload


def enqueue_auto_ai_chain(sid: str, runs_root: Path | str | None = None) -> str:
    """Queue the AI adjudication Celery chain and return the root task id."""

    runs_root_value = str(runs_root) if runs_root is not None else None

    logger.info("AUTO_AI_CHAIN_START sid=%s runs_root=%s", sid, runs_root_value)
    logger.info("STAGE_CHAIN_STARTED sid=%s", sid)

    workflow = chain(
        ai_score_step.s(sid, runs_root_value),
        merge_build_packs.s(),
        merge_send.s(),
        merge_compact.s(),
        run_date_convention_detector.s(),
        ai_validation_requirements_step.s(),
        validation_build_packs.s(),
        validation_send.s(),
        validation_compact.s(),
        ai_polarity_check_step.s(),
        ai_consistency_step.s(),
        pipeline_finalize.s(),
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

