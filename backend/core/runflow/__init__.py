from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import asyncio
import functools
import json
import logging
import os
import threading
from concurrent.futures import Future

_LOG = logging.getLogger(__name__)

from backend.core.ai.paths import (
    validation_result_json_filename_for_account,
    validation_result_jsonl_filename_for_account,
)
from backend.core.io.json_io import update_json_in_place
from backend.core.runflow_steps import (
    RUNS_ROOT,
    steps_append,
    steps_init,
    steps_stage_finish,
    steps_stage_start,
    steps_update_aggregate,
)
from backend.runflow.counters import frontend_answers_counters as _frontend_answers_counters


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    mode = 0o644
    fd = os.open(path, flags, mode)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def _events_path(sid: str) -> Path:
    return RUNS_ROOT / sid / "runflow_events.jsonl"


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        return default
    return value


def _review_attachment_required() -> bool:
    return _env_enabled("UMBRELLA_BARRIERS_REVIEW_REQUIRE_FILE", False)


def _review_explanation_required() -> bool:
    return _env_enabled("UMBRELLA_BARRIERS_REVIEW_REQUIRE_EXPLANATION", True)


def _document_verifier_enabled() -> bool:
    return _env_enabled("DOCUMENT_VERIFIER_ENABLED", False)


def _load_json_mapping(path: Path) -> Mapping[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if isinstance(payload, Mapping):
        return payload

    return None


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _has_review_attachments(payload: Mapping[str, Any]) -> bool:
    attachments = payload.get("attachments")
    if isinstance(attachments, Mapping):
        for value in attachments.values():
            if isinstance(value, str) and value.strip():
                return True
            if isinstance(value, Iterable) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                for entry in value:
                    if isinstance(entry, str) and entry.strip():
                        return True

    legacy = payload.get("evidence")
    if isinstance(legacy, Iterable) and not isinstance(legacy, (str, bytes, bytearray)):
        for item in legacy:
            if not isinstance(item, Mapping):
                continue
            docs = item.get("docs")
            if isinstance(docs, Iterable) and not isinstance(docs, (str, bytes, bytearray)):
                for doc in docs:
                    if not isinstance(doc, Mapping):
                        continue
                    doc_ids = doc.get("doc_ids")
                    if isinstance(doc_ids, Iterable) and not isinstance(
                        doc_ids, (str, bytes, bytearray)
                    ):
                        for doc_id in doc_ids:
                            if isinstance(doc_id, str) and doc_id.strip():
                                return True
    return False


def _resolve_review_path(
    run_dir: Path,
    env_name: str,
    canonical: Path,
    *,
    review_dir: Path,
    require_descendant: bool = False,
) -> Path:
    override = os.getenv(env_name)
    if not override:
        return canonical

    candidate = Path(override)
    if not candidate.is_absolute():
        candidate = (run_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()

    frontend_base = review_dir.parent
    try:
        candidate.relative_to(frontend_base)
        within_frontend = True
    except ValueError:
        within_frontend = False

    try:
        candidate.relative_to(review_dir)
        within_review = True
    except ValueError:
        within_review = False

    if within_frontend and not within_review:
        return canonical

    if require_descendant and within_review and candidate == review_dir:
        return canonical

    return candidate


def _load_response_payload(path: Path) -> Mapping[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None

    text = raw.strip()
    if not text:
        return None

    if path.suffix == ".jsonl":
        for line in reversed(text.splitlines()):
            stripped = line.strip()
            if stripped:
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    return None
                return payload if isinstance(payload, Mapping) else None
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    return payload if isinstance(payload, Mapping) else None


_ENABLE_STEPS = _env_enabled("RUNFLOW_VERBOSE")
_ENABLE_EVENTS = _env_enabled("RUNFLOW_EVENTS")
_STEP_SAMPLE_EVERY = max(_env_int("RUNFLOW_STEP_LOG_EVERY", 1), 1)
_PAIR_TOPN = max(_env_int("RUNFLOW_STEPS_PAIR_TOPN", 5), 0)
_ENABLE_SPANS = _env_enabled("RUNFLOW_STEPS_ENABLE_SPANS", True)
_ENABLE_ACCOUNT_STEPS = _env_enabled("RUNFLOW_ACCOUNT_STEPS", True)
_SUPPRESS_ACCOUNT_STEPS = _env_enabled("RUNFLOW_STEPS_SUPPRESS_PER_ACCOUNT")
_ONLY_AGGREGATES = _env_enabled("RUNFLOW_STEPS_ONLY_AGGREGATES")
_UMBRELLA_BARRIERS_ENABLED = _env_enabled("UMBRELLA_BARRIERS_ENABLED", True)
_UMBRELLA_BARRIERS_LOG = _env_enabled("UMBRELLA_BARRIERS_LOG", True)
_UMBRELLA_BARRIERS_WATCHDOG_INTERVAL_MS = max(
    _env_int("UMBRELLA_BARRIERS_WATCHDOG_INTERVAL_MS", 5000), 0
)


_STEP_CALL_COUNTS: dict[tuple[str, str, str, str], int] = defaultdict(int)
_STARTED_STAGES: set[tuple[str, str]] = set()
_STAGE_COUNTERS: dict[str, dict[str, dict[str, int]]] = {}
_STAGE_AGGREGATES: dict[str, dict[str, dict[str, int]]] = {}
_WATCHDOG_LOOP: Optional[asyncio.AbstractEventLoop] = None
_WATCHDOG_THREAD: Optional[threading.Thread] = None
_WATCHDOG_FUTURES: dict[str, Future[Any]] = {}
_WATCHDOG_LOCK = threading.Lock()

_VALIDATION_AGGREGATE_KEYS = frozenset({"packs_total", "packs_completed", "packs_pending"})
_REVIEW_AGGREGATE_KEYS = frozenset({"answers_received", "answers_required"})


def _append_event(sid: str, row: Mapping[str, Any]) -> None:
    if not _ENABLE_EVENTS:
        return
    _append_jsonl(_events_path(sid), row)


def steps_pair_topn() -> int:
    """Return the configured Top-N threshold for merge pair steps."""

    return _PAIR_TOPN


def runflow_account_steps_enabled() -> bool:
    """Return ``True`` when per-account runflow step logging is enabled."""

    return _ENABLE_ACCOUNT_STEPS


def _coerce_summary_counts(summary: Mapping[str, Any]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for key, value in summary.items():
        try:
            normalized[str(key)] = int(value)
        except (TypeError, ValueError):
            normalized[str(key)] = 0
    return normalized


def _store_stage_counter(sid: str, bucket: str, summary: Mapping[str, Any]) -> dict[str, int]:
    counters = _STAGE_COUNTERS.setdefault(sid, {})
    payload = _coerce_summary_counts(summary)
    counters[bucket] = payload
    return payload


def _aggregates_enabled() -> bool:
    return _SUPPRESS_ACCOUNT_STEPS and _ONLY_AGGREGATES


def _aggregate_state(sid: str, stage: str) -> dict[str, int]:
    state = _STAGE_AGGREGATES.setdefault(sid, {})
    return state.setdefault(stage, {})


def _aggregate_prune(stage_state: dict[str, int], allowed_keys: Iterable[str]) -> None:
    allowed = set(allowed_keys)
    for key in list(stage_state):
        if key not in allowed:
            stage_state.pop(key, None)


def _aggregate_set_nonnegative(stage_state: dict[str, int], key: str, value: Any) -> bool:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return False
    if number < 0:
        number = 0
    stage_state[key] = number
    return True


def _aggregate_value(stage_state: Mapping[str, Any], key: str) -> Optional[int]:
    value = stage_state.get(key)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _write_stage_aggregate(sid: str, stage: str) -> None:
    if not _aggregates_enabled():
        return
    state = _STAGE_AGGREGATES.get(sid, {})
    stage_state = state.get(stage)
    if not stage_state:
        return
    summary_payload: dict[str, int]
    if stage == "validation":
        _aggregate_prune(stage_state, _VALIDATION_AGGREGATE_KEYS)
        total_value = _aggregate_value(stage_state, "packs_total")
        completed_value = _aggregate_value(stage_state, "packs_completed")
        pending_value = _aggregate_value(stage_state, "packs_pending")
        if total_value is None and completed_value is None and pending_value is None:
            return
        total = max(total_value or 0, 0)
        completed = max(completed_value or 0, 0)
        if total and completed > total:
            completed = total
        if pending_value is None:
            pending = max(total - completed, 0)
        else:
            pending = max(pending_value, 0)
            if total and pending > total:
                pending = total
            if total and pending < total - completed:
                pending = total - completed
        stage_state["packs_total"] = total
        stage_state["packs_completed"] = completed
        stage_state["packs_pending"] = pending
        summary_payload = {
            "packs_total": total,
            "packs_completed": completed,
            "packs_pending": pending,
        }
    elif stage == "review":
        _aggregate_prune(stage_state, _REVIEW_AGGREGATE_KEYS)
        received_value = _aggregate_value(stage_state, "answers_received")
        required_value = _aggregate_value(stage_state, "answers_required")
        if received_value is None and required_value is None:
            return
        received = max(received_value or 0, 0)
        required = max(required_value or 0, 0)
        if required and received > required:
            received = required
        stage_state["answers_received"] = received
        stage_state["answers_required"] = required
        summary_payload = {
            "answers_received": received,
            "answers_required": required,
        }
    else:
        summary_payload = {key: stage_state[key] for key in sorted(stage_state)}
    steps_update_aggregate(sid, stage, summary_payload)


def _clear_stage_aggregate(sid: str, stage: str) -> None:
    state = _STAGE_AGGREGATES.get(sid)
    if not state:
        return
    state.pop(stage, None)
    if not state:
        _STAGE_AGGREGATES.pop(sid, None)


def _emit_summary_step(
    sid: str, stage: str, step: str, *, summary: Mapping[str, int]
) -> None:
    metrics = {str(key): value for key, value in summary.items()}
    out_payload = {"summary": dict(metrics)} if metrics else None
    runflow_step(
        sid,
        stage,
        step,
        status="success",
        metrics=metrics or None,
        out=out_payload,
    )


def _clear_stage_counters(sid: str, *buckets: str) -> None:
    state = _STAGE_COUNTERS.get(sid)
    if not state:
        return
    for bucket in buckets:
        state.pop(bucket, None)
    if not state:
        _STAGE_COUNTERS.pop(sid, None)


def _reset_step_counters(sid: str, stage: str) -> None:
    if not _STEP_CALL_COUNTS:
        return
    keys_to_delete = [key for key in _STEP_CALL_COUNTS if key[0] == sid and key[1] == stage]
    for key in keys_to_delete:
        del _STEP_CALL_COUNTS[key]


def _watchdog_interval_seconds() -> float:
    if _UMBRELLA_BARRIERS_WATCHDOG_INTERVAL_MS <= 0:
        return 0.0
    return _UMBRELLA_BARRIERS_WATCHDOG_INTERVAL_MS / 1000.0


def _ensure_watchdog_loop() -> Optional[asyncio.AbstractEventLoop]:
    interval = _watchdog_interval_seconds()
    if interval <= 0 or not _UMBRELLA_BARRIERS_ENABLED:
        return None

    global _WATCHDOG_LOOP, _WATCHDOG_THREAD

    loop = _WATCHDOG_LOOP
    if loop is not None and loop.is_running():
        return loop

    loop = asyncio.new_event_loop()
    _WATCHDOG_LOOP = loop

    def _runner() -> None:
        global _WATCHDOG_LOOP, _WATCHDOG_THREAD
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            with _WATCHDOG_LOCK:
                if _WATCHDOG_LOOP is loop:
                    _WATCHDOG_LOOP = None
                if _WATCHDOG_THREAD is threading.current_thread():
                    _WATCHDOG_THREAD = None

    thread = threading.Thread(
        target=_runner,
        name="runflow-barriers-watchdog",
        daemon=True,
    )
    _WATCHDOG_THREAD = thread
    thread.start()
    return loop


def _launch_watchdog_if_needed(sid: str) -> None:
    if not sid or not _UMBRELLA_BARRIERS_ENABLED:
        return

    interval = _watchdog_interval_seconds()
    if interval <= 0:
        return

    with _WATCHDOG_LOCK:
        existing = _WATCHDOG_FUTURES.get(sid)
        if existing is not None and not existing.done():
            return

        loop = _ensure_watchdog_loop()
        if loop is None:
            return

        try:
            future = asyncio.run_coroutine_threadsafe(runflow_barriers_watchdog(sid), loop)
        except RuntimeError:
            loop = _ensure_watchdog_loop()
            if loop is None:
                return
            future = asyncio.run_coroutine_threadsafe(runflow_barriers_watchdog(sid), loop)

        _WATCHDOG_FUTURES[sid] = future

        def _cleanup(done: Future[Any]) -> None:
            with _WATCHDOG_LOCK:
                current = _WATCHDOG_FUTURES.get(sid)
                if current is done:
                    _WATCHDOG_FUTURES.pop(sid, None)

        future.add_done_callback(_cleanup)


async def runflow_barriers_watchdog(sid: str) -> None:
    """Periodically reconcile umbrella readiness until ``sid`` is ready."""

    if not sid or not _UMBRELLA_BARRIERS_ENABLED:
        return

    normalized_sid = str(sid).strip()
    if not normalized_sid:
        return

    interval = _watchdog_interval_seconds()
    if interval <= 0:
        return

    base_dir = RUNS_ROOT / normalized_sid
    runflow_path = base_dir / "runflow.json"

    while True:
        try:
            runflow_barriers_refresh(normalized_sid)
        except Exception:  # pragma: no cover - defensive logging
            _LOG.debug(
                "[Runflow] Watchdog refresh failed sid=%s", normalized_sid, exc_info=True
            )

        runflow_payload = _load_json_mapping(runflow_path)
        ready = False
        if isinstance(runflow_payload, Mapping):
            barriers_payload = runflow_payload.get("umbrella_barriers")
            if isinstance(barriers_payload, Mapping):
                ready = bool(barriers_payload.get("all_ready"))

        if ready:
            break

        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise

def runflow_barriers_refresh(sid: str) -> Optional[dict[str, Any]]:
    """Recalculate umbrella readiness flags for ``sid``."""

    if not _UMBRELLA_BARRIERS_ENABLED:
        return None

    normalized_sid = str(sid or "").strip()
    if not normalized_sid:
        return None

    base_path = os.path.join(str(RUNS_ROOT), normalized_sid)
    base_dir = Path(base_path)

    try:
        if not base_dir.exists() or not base_dir.is_dir():
            return None
    except OSError:
        return None

    runflow_path = base_dir / "runflow.json"
    steps_path = base_dir / "runflow_steps.json"

    runflow_payload = _load_json_mapping(runflow_path) or {}
    steps_payload = _load_json_mapping(steps_path) or {}

    runflow_stages_raw = (
        runflow_payload.get("stages")
        if isinstance(runflow_payload, Mapping)
        else None
    )
    runflow_stages = runflow_stages_raw if isinstance(runflow_stages_raw, Mapping) else {}

    steps_stages_raw = (
        steps_payload.get("stages") if isinstance(steps_payload, Mapping) else None
    )
    steps_stages = steps_stages_raw if isinstance(steps_stages_raw, Mapping) else {}

    def _coerce_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            try:
                return int(candidate)
            except ValueError:
                return None
        return None

    # ------------------------------------------------------------------
    # Validation counters
    # ------------------------------------------------------------------
    validation_index_path = base_dir / "ai_packs" / "validation" / "index.json"
    validation_index = _load_json_mapping(validation_index_path)

    validation_entries: list[Mapping[str, Any]] = []
    if isinstance(validation_index, Mapping):
        packs_value = validation_index.get("packs")
        if isinstance(packs_value, Sequence):
            validation_entries = [
                entry for entry in packs_value if isinstance(entry, Mapping)
            ]

    validation_packs_total = len(validation_entries)

    index_dir = validation_index_path.parent
    results_dir_override = None
    if isinstance(validation_index, Mapping):
        raw_results_dir = validation_index.get("results_dir")
        if isinstance(raw_results_dir, str) and raw_results_dir.strip():
            candidate = Path(raw_results_dir.strip())
            if not candidate.is_absolute():
                candidate = (index_dir / candidate).resolve()
            else:
                candidate = candidate.resolve()
            results_dir_override = candidate

    validation_results_dir = (
        results_dir_override
        if results_dir_override is not None
        else base_dir / "ai_packs" / "validation" / "results"
    )

    def _entry_result_paths(entry: Mapping[str, Any]) -> list[Path]:
        paths: list[Path] = []
        for key in (
            "result_json",
            "result_jsonl",
            "result_json_path",
            "result_jsonl_path",
            "results_path",
        ):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                candidate = Path(value.strip())
                if not candidate.is_absolute():
                    candidate = (index_dir / candidate).resolve()
                else:
                    candidate = candidate.resolve()
                paths.append(candidate)
        return paths

    validation_packs_completed = 0
    for entry in validation_entries:
        status_value = entry.get("status")
        if not isinstance(status_value, str):
            continue
        if status_value.strip().lower() != "completed":
            continue

        result_paths = _entry_result_paths(entry)
        has_result = False
        for path in result_paths:
            try:
                if path.exists() and path.is_file():
                    has_result = True
                    break
            except OSError:
                continue

        if not has_result:
            account_id_value = entry.get("account_id")
            candidate_ids: list[Any] = []
            if isinstance(account_id_value, int):
                candidate_ids.append(account_id_value)
            elif isinstance(account_id_value, str) and account_id_value.strip():
                candidate_ids.append(account_id_value.strip())

            for account_id in candidate_ids:
                for resolver in (
                    validation_result_json_filename_for_account,
                    validation_result_jsonl_filename_for_account,
                ):
                    try:
                        filename = resolver(account_id)
                    except Exception:
                        continue
                    if not isinstance(filename, str) or not filename:
                        continue
                    candidate_path = Path(filename)
                    if not candidate_path.is_absolute():
                        candidate_path = (validation_results_dir / filename).resolve()
                    else:
                        candidate_path = candidate_path.resolve()
                    try:
                        if candidate_path.exists() and candidate_path.is_file():
                            has_result = True
                            break
                    except OSError:
                        continue
                if has_result:
                    break

        if has_result:
            validation_packs_completed += 1

    validation_packs_pending = max(
        validation_packs_total - validation_packs_completed,
        0,
    )

    validation_ready = bool(
        validation_packs_total > 0 and validation_packs_completed == validation_packs_total
    )

    # ------------------------------------------------------------------
    # Review counters
    # ------------------------------------------------------------------
    answers_required: int | None = None

    runflow_frontend = (
        runflow_stages.get("frontend") if isinstance(runflow_stages, Mapping) else None
    )
    if isinstance(runflow_frontend, Mapping):
        metrics_payload = runflow_frontend.get("metrics")
        if isinstance(metrics_payload, Mapping):
            packs_count_value = _coerce_int(metrics_payload.get("packs_count"))
            if packs_count_value is not None:
                answers_required = packs_count_value
            else:
                answers_required_value = _coerce_int(
                    metrics_payload.get("answers_required")
                )
                if answers_required_value is not None:
                    answers_required = answers_required_value
        if answers_required is None:
            stage_packs = _coerce_int(runflow_frontend.get("packs_count"))
            if stage_packs is not None:
                answers_required = stage_packs

    if answers_required is None:
        steps_frontend = (
            steps_stages.get("frontend") if isinstance(steps_stages, Mapping) else None
        )
        if isinstance(steps_frontend, Mapping):
            summary_payload = steps_frontend.get("summary")
            if isinstance(summary_payload, Mapping):
                for key in ("answers_required", "packs_count", "accounts_published"):
                    value = _coerce_int(summary_payload.get(key))
                    if value is not None:
                        answers_required = value
                        break

    if answers_required is None:
        answers_required = 0

    review_responses_dir = base_dir / "frontend" / "review" / "responses"
    explanation_required = _review_explanation_required()

    answers_received = 0
    try:
        response_entries = sorted(
            path
            for path in review_responses_dir.iterdir()
            if path.is_file() and path.name.endswith(".result.json")
        )
    except OSError:
        response_entries = []

    for response_path in response_entries:
        payload = _load_json_mapping(response_path)
        if not isinstance(payload, Mapping):
            continue
        answers_payload = payload.get("answers")
        if not isinstance(answers_payload, Mapping):
            continue
        explanation = answers_payload.get("explanation")
        if explanation_required:
            if not isinstance(explanation, str) or not explanation.strip():
                continue
        elif explanation is not None and not isinstance(explanation, str):
            continue
        answers_received += 1

    answers_required = max(answers_required, 0)
    answers_received = max(answers_received, 0)

    review_ready = bool(
        answers_required > 0 and answers_received == answers_required
    )

    # ------------------------------------------------------------------
    # Merge counters
    # ------------------------------------------------------------------
    merge_index_path = base_dir / "ai_packs" / "merge" / "pairs_index.json"
    merge_index = _load_json_mapping(merge_index_path)

    merge_expected: int | None = None
    if isinstance(merge_index, Mapping):
        totals_payload = merge_index.get("totals")
        if isinstance(totals_payload, Mapping):
            scored_pairs = _coerce_int(totals_payload.get("scored_pairs"))
            if scored_pairs is not None:
                merge_expected = max(scored_pairs, 0)
        if merge_expected is None:
            direct_pairs = _coerce_int(merge_index.get("scored_pairs"))
            if direct_pairs is not None:
                merge_expected = max(direct_pairs, 0)
        if merge_expected is None:
            pairs_entries = merge_index.get("pairs")
            if isinstance(pairs_entries, Sequence):
                merge_expected = len([entry for entry in pairs_entries if isinstance(entry, Mapping)])

    merge_results_dir = base_dir / "ai_packs" / "merge" / "results"
    try:
        merge_files = [
            path
            for path in merge_results_dir.rglob("*.result.json")
            if path.is_file()
        ]
    except OSError:
        merge_files = []

    merge_completed = len(merge_files)

    if merge_expected is None:
        merge_ready = False
    else:
        merge_ready = merge_completed >= merge_expected

    all_ready = merge_ready and validation_ready and review_ready

    normalized_barriers: dict[str, Any] = {
        "merge_ready": merge_ready,
        "validation_ready": validation_ready,
        "review_ready": review_ready,
        "all_ready": all_ready,
    }

    if _document_verifier_enabled():
        normalized_barriers.setdefault("document_ready", False)

    timestamp = _utcnow_iso()
    normalized_barriers["checked_at"] = timestamp

    validation_metrics_payload = {
        "packs_total": int(validation_packs_total),
        "packs_completed": int(validation_packs_completed),
        "packs_pending": int(validation_packs_pending),
    }

    frontend_metrics_payload = {
        "answers_required": int(answers_required),
        "answers_received": int(answers_received),
    }

    def _mutate(payload: Any) -> Any:
        if not isinstance(payload, dict):
            payload_dict: dict[str, Any] = {}
        else:
            payload_dict = payload

        existing_barriers = payload_dict.get("umbrella_barriers")
        if isinstance(existing_barriers, Mapping):
            barriers_payload = dict(existing_barriers)
        else:
            barriers_payload = {}

        barriers_payload.update(normalized_barriers)
        payload_dict["umbrella_barriers"] = barriers_payload
        payload_dict["umbrella_ready"] = all_ready
        payload_dict.setdefault("sid", normalized_sid)

        stages_raw = payload_dict.get("stages")
        if isinstance(stages_raw, Mapping):
            stages_dict: dict[str, Any] = {}
            for key, value in stages_raw.items():
                if isinstance(value, Mapping):
                    stages_dict[key] = dict(value)
                else:
                    stages_dict[key] = value
        else:
            stages_dict = {}

        def _ensure_stage_entry(stage_name: str) -> dict[str, Any]:
            existing = stages_dict.get(stage_name)
            if isinstance(existing, Mapping):
                stage_entry = dict(existing)
            else:
                stage_entry = {}
            stages_dict[stage_name] = stage_entry
            return stage_entry

        def _update_metrics(stage_name: str, metrics: Mapping[str, int]) -> None:
            stage_entry = _ensure_stage_entry(stage_name)
            existing_metrics = stage_entry.get("metrics")
            metrics_dict = dict(existing_metrics) if isinstance(existing_metrics, Mapping) else {}
            changed = False
            for key, value in metrics.items():
                if metrics_dict.get(key) != value:
                    metrics_dict[key] = value
                    changed = True
            if not isinstance(existing_metrics, Mapping) or changed:
                stage_entry["metrics"] = metrics_dict

        _update_metrics("validation", validation_metrics_payload)
        _update_metrics("frontend", frontend_metrics_payload)

        payload_dict["stages"] = stages_dict
        payload_dict["updated_at"] = timestamp

        return payload_dict

    try:
        update_json_in_place(runflow_path, _mutate)
    except Exception:
        return None

    if _UMBRELLA_BARRIERS_LOG:
        _LOG.info(
            "[Runflow] Umbrella barriers: merge=%s validation=%s review=%s all_ready=%s",
            merge_ready,
            validation_ready,
            review_ready,
            all_ready,
        )
        event_payload: dict[str, Any] = {
            "ts": timestamp,
            "event": "barriers_reconciled",
            "merge_ready": merge_ready,
            "validation_ready": validation_ready,
            "review_ready": review_ready,
            "all_ready": all_ready,
        }
        if _document_verifier_enabled():
            event_payload.setdefault("document_ready", False)
        try:
            _append_jsonl(_events_path(normalized_sid), event_payload)
        except Exception:
            _LOG.debug(
                "[Runflow] Failed to append barriers event sid=%s", normalized_sid, exc_info=True
            )

    return normalized_barriers


def _update_umbrella_barriers(sid: str) -> None:
    runflow_barriers_refresh(sid)


def runflow_refresh_umbrella_barriers(sid: str) -> None:
    """Re-evaluate umbrella readiness for ``sid``."""

    runflow_barriers_refresh(sid)


def _should_record_step(
    sid: str, stage: str, substage: str, step: str, status: str
) -> bool:
    if status != "success":
        return True
    if _STEP_SAMPLE_EVERY <= 1:
        key = (sid, stage, substage, step)
        _STEP_CALL_COUNTS[key] += 1
        return True

    key = (sid, stage, substage, step)
    _STEP_CALL_COUNTS[key] += 1
    count = _STEP_CALL_COUNTS[key]
    return count == 1 or count % _STEP_SAMPLE_EVERY == 0


def _shorten_message(message: str, *, limit: int = 200) -> str:
    compact = " ".join(message.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "\u2026"


def _format_error_payload(exc: Exception, where: str) -> dict[str, str]:
    message = str(exc) or exc.__class__.__name__
    short = _shorten_message(message)
    return {
        "type": exc.__class__.__name__,
        "message": short,
        "where": where,
        "hint": "see runflow_events.jsonl",
    }


def runflow_start_stage(
    sid: str, stage: str, extra: Optional[Mapping[str, Any]] = None
) -> None:
    steps_enabled = _ENABLE_STEPS
    events_enabled = _ENABLE_EVENTS
    _launch_watchdog_if_needed(sid)
    if not (steps_enabled or events_enabled):
        return

    ts = _utcnow_iso()
    key = (sid, stage)
    created = False
    _reset_step_counters(sid, stage)
    if steps_enabled:
        steps_init(sid)
        created = steps_stage_start(sid, stage, started_at=ts, extra=extra)
    else:
        created = key not in _STARTED_STAGES

    if events_enabled and created:
        _append_event(sid, {"ts": ts, "stage": stage, "event": "start"})

    if created:
        _STARTED_STAGES.add(key)


def runflow_end_stage(
    sid: str,
    stage: str,
    *,
    status: str = "success",
    summary: Optional[Mapping[str, Any]] = None,
    stage_status: Optional[str] = None,
    empty_ok: bool = False,
) -> None:
    steps_enabled = _ENABLE_STEPS
    events_enabled = _ENABLE_EVENTS

    if steps_enabled or events_enabled:
        ts = _utcnow_iso()
        if steps_enabled:
            status_for_steps = stage_status or status
            steps_stage_finish(
                sid,
                stage,
                status_for_steps,
                summary,
                ended_at=ts,
                empty_ok=empty_ok,
            )

        if events_enabled:
            event: dict[str, Any] = {"ts": ts, "stage": stage, "event": "end", "status": status}
            if summary:
                event["summary"] = {str(k): v for k, v in summary.items()}
            _append_event(sid, event)

    _STARTED_STAGES.discard((sid, stage))
    _reset_step_counters(sid, stage)

    _update_umbrella_barriers(sid)

    if stage == "validation":
        _clear_stage_counters(sid, "validation_build", "validation_results")
        if _aggregates_enabled():
            _clear_stage_aggregate(sid, "validation")
    elif stage == "frontend":
        _clear_stage_counters(sid, "frontend_review")
        if _aggregates_enabled():
            _clear_stage_aggregate(sid, "review")


def runflow_event(
    sid: str,
    stage: str,
    step: str,
    *,
    status: str = "success",
    account: Optional[str] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    out: Optional[Mapping[str, Any]] = None,
    substage: Optional[str] = None,
    reason: Optional[str] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    error: Optional[Mapping[str, Any]] = None,
) -> None:
    """Emit a runflow event without recording a step entry."""

    if not _ENABLE_EVENTS:
        return

    ts = _utcnow_iso()
    substage_name = substage or "default"
    event: dict[str, Any] = {
        "ts": ts,
        "stage": stage,
        "step": step,
        "status": status,
        "substage": substage_name,
    }
    if account is not None:
        event["account"] = account
    if metrics:
        event["metrics"] = {str(k): v for k, v in metrics.items()}
    if out:
        event["out"] = {str(k): v for k, v in out.items()}
    if reason is not None:
        event["reason"] = reason
    if span_id is not None:
        event["span_id"] = span_id
    if parent_span_id is not None:
        event["parent_span_id"] = parent_span_id
    if error:
        event["error"] = {str(k): v for k, v in error.items()}
    _append_event(sid, event)


def runflow_decide_step(
    sid: str,
    stage: str,
    *,
    next_action: str,
    reason: str,
) -> None:
    """Record a compact decision step for ``stage`` if runflow is enabled."""

    steps_enabled = _ENABLE_STEPS
    events_enabled = _ENABLE_EVENTS

    if not (steps_enabled or events_enabled):
        return

    ts = _utcnow_iso()
    stage_name = str(stage)

    decision_out = {"next": str(next_action), "reason": str(reason)}

    if steps_enabled:
        steps_append(
            sid,
            stage_name,
            "runflow_decide",
            "success",
            t=ts,
            out=decision_out,
        )

    if events_enabled:
        event: dict[str, Any] = {
            "ts": ts,
            "stage": stage_name,
            "event": "decide",
            "next": str(next_action),
            "reason": str(reason),
        }
        _append_event(sid, event)


def runflow_step(
    sid: str,
    stage: str,
    step: str,
    *,
    status: str = "success",
    account: Optional[str] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    out: Optional[Mapping[str, Any]] = None,
    substage: Optional[str] = None,
    reason: Optional[str] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    error: Optional[Mapping[str, Any]] = None,
) -> None:
    steps_enabled = _ENABLE_STEPS
    events_enabled = _ENABLE_EVENTS

    if not (steps_enabled or events_enabled):
        return

    ts = _utcnow_iso()
    substage_name = substage or "default"

    event_step = step
    event_status = status

    step_for_steps: Optional[str] = step
    status_for_steps = status

    if stage == "merge":
        if event_step == "merge_scoring":
            if event_status == "start":
                step_for_steps = "merge_scoring_start"
                status_for_steps = "success"
            else:
                step_for_steps = "merge_scoring_finish"
                status_for_steps = event_status
        elif event_step == "pack_skip" and event_status == "success":
            step_for_steps = None

    record_step_success = True
    if step_for_steps is not None and status_for_steps == "success":
        record_step_success = _should_record_step(
            sid, stage, substage_name, step_for_steps, status_for_steps
        )
        if not record_step_success:
            return
    elif step_for_steps is not None and status_for_steps != "success":
        _STEP_CALL_COUNTS[(sid, stage, substage_name, step_for_steps)] += 1

    if steps_enabled and step_for_steps is not None and record_step_success:
        step_span_id = span_id if _ENABLE_SPANS else None
        step_parent_span_id = parent_span_id if _ENABLE_SPANS else None
        should_write_step = True
        if stage == "merge":
            allowed_success_steps = {
                "merge_scoring_start",
                "acctnum_normalize",
                "acctnum_match_level",
                "acctnum_pairs_summary",
                "no_merge_candidates",
                "pack_create",
                "merge_scoring_finish",
            }
            if status_for_steps == "success":
                should_write_step = step_for_steps in allowed_success_steps
            else:
                should_write_step = step_for_steps == "merge_scoring_finish"
        if (
            _SUPPRESS_ACCOUNT_STEPS
            and account is not None
            and status_for_steps == "success"
        ):
            should_write_step = False
        if (
            _aggregates_enabled()
            and stage in {"validation", "frontend"}
            and status_for_steps == "success"
            and error is None
        ):
            should_write_step = False
        if should_write_step:
            steps_append(
                sid,
                stage,
                step_for_steps,
                status_for_steps,
                t=ts,
                account=account,
                metrics=metrics,
                out=out,
                reason=reason,
                span_id=step_span_id,
                parent_span_id=step_parent_span_id,
                error=error,
            )
        elif stage == "merge":
            key = (sid, stage)
            if key not in _STARTED_STAGES:
                steps_stage_start(sid, stage, started_at=ts)
                _STARTED_STAGES.add(key)

    if events_enabled:
        event: dict[str, Any] = {
            "ts": ts,
            "stage": stage,
            "step": event_step,
            "status": event_status,
            "substage": substage_name,
        }
        if account is not None:
            event["account"] = account
        if metrics:
            event["metrics"] = {str(k): v for k, v in metrics.items()}
        if out:
            event["out"] = {str(k): v for k, v in out.items()}
        if reason is not None:
            event["reason"] = reason
        if span_id is not None:
            event["span_id"] = span_id
        if parent_span_id is not None:
            event["parent_span_id"] = parent_span_id
        if error:
            event["error"] = {str(k): v for k, v in error.items()}
        _append_event(sid, event)


def record_validation_build_summary(
    sid: str,
    *,
    eligible_accounts: Any,
    packs_built: Any,
    packs_skipped: Any,
) -> None:
    summary = {
        "eligible_accounts": eligible_accounts,
        "packs_built": packs_built,
        "packs_skipped": packs_skipped,
    }
    normalized = _store_stage_counter(sid, "validation_build", summary)
    if _aggregates_enabled():
        stage_state = _aggregate_state(sid, "validation")
        _aggregate_prune(stage_state, _VALIDATION_AGGREGATE_KEYS)
        total_updated = _aggregate_set_nonnegative(
            stage_state, "packs_total", normalized.get("eligible_accounts")
        )
        if not total_updated and "packs_total" not in stage_state:
            built_value = normalized.get("packs_built")
            skipped_value = normalized.get("packs_skipped")
            candidate_total: Optional[int]
            try:
                built_int = int(built_value)
                skipped_int = int(skipped_value)
            except (TypeError, ValueError):
                candidate_total = None
            else:
                candidate_total = built_int + skipped_int
            if candidate_total is not None:
                _aggregate_set_nonnegative(stage_state, "packs_total", candidate_total)
        _aggregate_set_nonnegative(stage_state, "packs_completed", normalized.get("packs_built"))
        stage_state.pop("packs_pending", None)
        _write_stage_aggregate(sid, "validation")
    _emit_summary_step(sid, "validation", "build_packs", summary=normalized)


def record_validation_results_summary(
    sid: str,
    *,
    results_total: Any,
    completed: Any,
    failed: Any,
    pending: Any,
) -> None:
    summary = {
        "results_total": results_total,
        "completed": completed,
        "failed": failed,
        "pending": pending,
    }
    normalized = _store_stage_counter(sid, "validation_results", summary)
    if normalized.get("pending", 0) < 0:
        normalized["pending"] = 0
    if _aggregates_enabled():
        stage_state = _aggregate_state(sid, "validation")
        _aggregate_prune(stage_state, _VALIDATION_AGGREGATE_KEYS)
        _aggregate_set_nonnegative(stage_state, "packs_total", normalized.get("results_total"))
        _aggregate_set_nonnegative(stage_state, "packs_completed", normalized.get("completed"))
        pending_provided = _aggregate_set_nonnegative(
            stage_state, "packs_pending", normalized.get("pending")
        )
        if not pending_provided:
            stage_state.pop("packs_pending", None)
        _write_stage_aggregate(sid, "validation")
    _emit_summary_step(sid, "validation", "collect_results", summary=normalized)


def record_frontend_responses_progress(
    sid: str,
    *,
    accounts_published: Any,
    answers_received: Any,
    answers_required: Any,
) -> None:
    base_dir = RUNS_ROOT / sid
    counters = _frontend_answers_counters(
        base_dir, attachments_required=_review_attachment_required()
    )

    answers_required_disk = counters.get("answers_required")
    answers_received_disk = counters.get("answers_received")

    if isinstance(answers_required_disk, int):
        accounts_value = answers_required_disk
    else:
        accounts_value = accounts_published

    summary = {
        "accounts_published": accounts_value,
        "answers_received": answers_received_disk
        if isinstance(answers_received_disk, int)
        else answers_received,
        "answers_required": answers_required_disk
        if isinstance(answers_required_disk, int)
        else answers_required,
    }
    normalized = _store_stage_counter(sid, "frontend_review", summary)
    if _aggregates_enabled():
        stage_state = _aggregate_state(sid, "review")
        _aggregate_prune(stage_state, _REVIEW_AGGREGATE_KEYS)
        _aggregate_set_nonnegative(
            stage_state, "answers_received", normalized.get("answers_received")
        )
        _aggregate_set_nonnegative(
            stage_state, "answers_required", normalized.get("answers_required")
        )
        _write_stage_aggregate(sid, "review")
    _emit_summary_step(sid, "frontend", "responses_progress", summary=normalized)


def runflow_step_dec(stage: str, step: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> Any:
            if not (_ENABLE_STEPS or _ENABLE_EVENTS):
                return fn(*args, **kwargs)

            sid: Optional[str] = None
            if "sid" in kwargs and isinstance(kwargs["sid"], str):
                sid = kwargs["sid"]
            elif args:
                candidate = args[0]
                if hasattr(candidate, "sid"):
                    sid_value = getattr(candidate, "sid")
                    if isinstance(sid_value, str):
                        sid = sid_value
                elif isinstance(candidate, Mapping):
                    sid_value = candidate.get("sid")
                    if isinstance(sid_value, str):
                        sid = sid_value

            where = f"{fn.__module__}:{getattr(fn, '__qualname__', fn.__name__)}"

            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                if sid:
                    error_payload = _format_error_payload(exc, where)
                    runflow_step(
                        sid,
                        stage,
                        step,
                        status="error",
                        error=error_payload,
                    )
                raise

            if sid:
                runflow_step(sid, stage, step, status="success")
            return result

        return inner

    return _wrap


__all__ = [
    "runflow_start_stage",
    "runflow_end_stage",
    "runflow_barriers_refresh",
    "runflow_barriers_watchdog",
    "runflow_refresh_umbrella_barriers",
    "runflow_decide_step",
    "runflow_event",
    "runflow_step",
    "runflow_step_dec",
    "steps_pair_topn",
    "record_validation_build_summary",
    "record_validation_results_summary",
    "record_frontend_responses_progress",
    "runflow_account_steps_enabled",
]
