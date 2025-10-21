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

from backend.core.io.json_io import update_json_in_place
from backend.core.runflow_steps import (
    RUNS_ROOT,
    steps_append,
    steps_init,
    steps_stage_finish,
    steps_stage_start,
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
_UMBRELLA_BARRIERS_ENABLED = _env_enabled("UMBRELLA_BARRIERS_ENABLED", True)
_UMBRELLA_BARRIERS_LOG = _env_enabled("UMBRELLA_BARRIERS_LOG", True)
_UMBRELLA_BARRIERS_WATCHDOG_INTERVAL_MS = max(
    _env_int("UMBRELLA_BARRIERS_WATCHDOG_INTERVAL_MS", 5000), 0
)


_STEP_CALL_COUNTS: dict[tuple[str, str, str, str], int] = defaultdict(int)
_STARTED_STAGES: set[tuple[str, str]] = set()
_STAGE_COUNTERS: dict[str, dict[str, dict[str, int]]] = {}
_WATCHDOG_LOOP: Optional[asyncio.AbstractEventLoop] = None
_WATCHDOG_THREAD: Optional[threading.Thread] = None
_WATCHDOG_FUTURES: dict[str, Future[Any]] = {}
_WATCHDOG_LOCK = threading.Lock()


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

    interval = _watchdog_interval_seconds()
    if interval <= 0:
        return

    while True:
        try:
            result = runflow_barriers_refresh(sid)
        except Exception:  # pragma: no cover - defensive logging
            _LOG.debug(
                "[Runflow] Watchdog refresh failed sid=%s", sid, exc_info=True
            )
            result = None

        if isinstance(result, Mapping) and bool(result.get("all_ready")):
            break

        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise

def runflow_barriers_refresh(sid: str) -> Optional[dict[str, Any]]:
    """Recalculate umbrella readiness flags for ``sid``."""

    if not _UMBRELLA_BARRIERS_ENABLED:
        return None

    run_dir = RUNS_ROOT / sid
    runflow_path = run_dir / "runflow.json"

    runflow_payload = _load_json_mapping(runflow_path) or {}
    runflow_stages_raw = runflow_payload.get("stages")
    runflow_stages = runflow_stages_raw if isinstance(runflow_stages_raw, Mapping) else {}

    steps_payload = _load_json_mapping(run_dir / "runflow_steps.json") or {}
    steps_stages_raw = steps_payload.get("stages")
    steps_stages = steps_stages_raw if isinstance(steps_stages_raw, Mapping) else {}

    def _stage_info(stage: str) -> Mapping[str, Any] | None:
        info = runflow_stages.get(stage) if isinstance(runflow_stages, Mapping) else None
        if isinstance(info, Mapping):
            return info
        info = steps_stages.get(stage) if isinstance(steps_stages, Mapping) else None
        if isinstance(info, Mapping):
            return info
        return None

    def _stage_status(stage: str) -> str:
        for container in (runflow_stages, steps_stages):
            if not isinstance(container, Mapping):
                continue
            info = container.get(stage)
            if not isinstance(info, Mapping):
                continue
            status = info.get("status")
            if isinstance(status, str):
                normalized = status.strip().lower()
                if normalized:
                    return normalized
        return ""

    validation_stage_info = _stage_info("validation")
    validation_ready = False
    try:
        from backend.runflow.decider import _validation_stage_ready  # Local import to avoid circular dependency
    except Exception:
        validation_ready = False
    else:
        try:
            validation_ready = _validation_stage_ready(run_dir, validation_stage_info)
        except Exception:
            validation_ready = False

    if _stage_status("validation") == "error":
        validation_ready = False

    review_stage_info = _stage_info("frontend")
    review_status = _stage_status("frontend")

    review_dir_canonical = (run_dir / "frontend" / "review").resolve()
    index_path = _resolve_review_path(
        run_dir,
        "FRONTEND_PACKS_INDEX",
        (review_dir_canonical / "index.json").resolve(),
        review_dir=review_dir_canonical,
    )
    responses_dir = _resolve_review_path(
        run_dir,
        "FRONTEND_PACKS_RESPONSES_DIR",
        (review_dir_canonical / "responses").resolve(),
        review_dir=review_dir_canonical,
        require_descendant=True,
    )

    index_payload = _load_json_mapping(index_path)
    required_answers = 0
    if isinstance(index_payload, Mapping):
        packs_entries = index_payload.get("packs")
        if _is_sequence(packs_entries):
            required_answers = max(required_answers, len(list(packs_entries)))
        items_entries = index_payload.get("items")
        if _is_sequence(items_entries):
            required_answers = max(required_answers, len(list(items_entries)))
        answers_required_value = index_payload.get("answers_required")
        if isinstance(answers_required_value, int):
            required_answers = max(required_answers, answers_required_value)
        elif isinstance(answers_required_value, str):
            try:
                required_answers = max(required_answers, int(answers_required_value))
            except ValueError:
                pass

    attachments_required = _review_attachment_required()
    answered_ids: set[str] = set()
    if responses_dir.exists() and responses_dir.is_dir():
        for entry in sorted(responses_dir.iterdir()):
            if not entry.is_file():
                continue
            name = entry.name
            if not (name.endswith(".result.json") or name.endswith(".result.jsonl")):
                continue
            payload = _load_response_payload(entry)
            if not isinstance(payload, Mapping):
                continue
            answers = payload.get("answers")
            if not isinstance(answers, Mapping):
                continue
            explanation = answers.get("explanation")
            if _review_explanation_required():
                if not isinstance(explanation, str) or not explanation.strip():
                    continue
            else:
                if explanation is not None and (not isinstance(explanation, str) or not explanation.strip()):
                    continue
            if attachments_required and not _has_review_attachments(answers):
                continue
            received_at = payload.get("received_at")
            if not isinstance(received_at, str) or not received_at.strip():
                continue
            account_id = payload.get("account_id")
            if isinstance(account_id, str) and account_id.strip():
                answered_ids.add(account_id.strip())
            else:
                answered_ids.add(entry.stem)

    answers_received = len(answered_ids)

    review_ready = False
    if required_answers > 0:
        review_ready = answers_received >= required_answers
    elif isinstance(review_stage_info, Mapping) and bool(review_stage_info.get("empty_ok")):
        review_ready = True

    if review_status == "error":
        review_ready = False

    merge_stage_info = _stage_info("merge")
    merge_status = _stage_status("merge")
    merge_ready = False
    if merge_status in {"success", "built", "complete", "completed", "published"}:
        merge_ready = True
    elif isinstance(merge_stage_info, Mapping) and bool(merge_stage_info.get("empty_ok")):
        merge_ready = True

    if merge_status == "error":
        merge_ready = False

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

    def _mutate(payload: Any) -> Any:
        if not isinstance(payload, dict):
            payload_dict: dict[str, Any] = {}
        else:
            payload_dict = payload

        existing_raw = payload_dict.get("umbrella_barriers")
        if isinstance(existing_raw, Mapping):
            umbrella: dict[str, Any] = dict(existing_raw)
        else:
            umbrella = {}

        umbrella.update(normalized_barriers)
        payload_dict["umbrella_barriers"] = umbrella
        payload_dict["umbrella_ready"] = all_ready
        payload_dict.setdefault("sid", sid)
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
            _append_jsonl(_events_path(sid), event_payload)
        except Exception:
            _LOG.debug(
                "[Runflow] Failed to append barriers event sid=%s", sid, exc_info=True
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
    elif stage == "frontend":
        _clear_stage_counters(sid, "frontend_review")


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
            steps_stage_start(sid, stage, started_at=ts)

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
