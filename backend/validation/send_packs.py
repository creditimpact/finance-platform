"""Send Validation AI packs to the model and persist the responses."""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from jsonschema import Draft7Validator

from backend.analytics.analytics_tracker import emit_counter
from backend.core.ai.paths import (
    validation_result_error_filename_for_account,
    validation_result_jsonl_filename_for_account,
    validation_result_summary_filename_for_account,
)
from backend.core.logic.validation_field_sets import (
    ALL_VALIDATION_FIELDS,
    ALWAYS_INVESTIGATABLE_FIELDS,
    CONDITIONAL_FIELDS,
)
from backend.validation.index_schema import (
    ValidationIndex,
    ValidationPackRecord,
)
from backend.validation.redaction import sanitize_validation_log_payload

_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_TIMEOUT = 30.0
_THROTTLE_SECONDS = 0.05
_DEFAULT_QUEUE_NAME = "validation"
_VALID_DECISIONS = {"strong", "no_case"}
_ALWAYS_INVESTIGATABLE_FIELDS = ALWAYS_INVESTIGATABLE_FIELDS
_CONDITIONAL_FIELDS = CONDITIONAL_FIELDS
_ALLOWED_FIELDS = frozenset(ALL_VALIDATION_FIELDS)
_CREDITOR_REMARK_KEYWORDS = (
    "charge off",
    "charge-off",
    "consumer dispute",
    "consumer disputes",
    "consumer states",
    "fcra",
    "fraud",
    "fraudulent",
    "repossession",
)

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["decision", "justification", "labels", "confidence"],
    "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": ["strong", "no_case"]},
        "justification": {"type": "string", "minLength": 1},
        "labels": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "citations": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "default": [],
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
}

_RESPONSE_VALIDATOR = Draft7Validator(_RESPONSE_SCHEMA)
_CONFIDENCE_THRESHOLD_ENV = "VALIDATION_AI_MIN_CONFIDENCE"
_DEFAULT_CONFIDENCE_THRESHOLD = 0.70


log = logging.getLogger(__name__)

requests: Any | None = None


def _confidence_threshold() -> float:
    raw = os.getenv(_CONFIDENCE_THRESHOLD_ENV)
    if raw is None:
        return _DEFAULT_CONFIDENCE_THRESHOLD
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        log.warning(
            "VALIDATION_AI_CONFIDENCE_PARSE_FAILED value=%s default=%s",
            raw,
            _DEFAULT_CONFIDENCE_THRESHOLD,
        )
        return _DEFAULT_CONFIDENCE_THRESHOLD
    if value < 0 or value > 1:
        log.warning(
            "VALIDATION_AI_CONFIDENCE_OUT_OF_RANGE value=%s default=%s",
            value,
            _DEFAULT_CONFIDENCE_THRESHOLD,
        )
        return _DEFAULT_CONFIDENCE_THRESHOLD
    return value


def _coerce_bool_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return False


def _coerce_int_value(value: Any, default: int = 1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _extract_bureau_records(
    pack_line: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    bureaus = pack_line.get("bureaus")
    if not isinstance(bureaus, Mapping):
        return {}
    records: dict[str, dict[str, Any]] = {}
    for bureau, value in bureaus.items():
        if isinstance(value, Mapping):
            records[str(bureau)] = {
                "raw": value.get("raw"),
                "normalized": value.get("normalized"),
            }
    return records


def _normalize_account_number_token(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        last4 = value.get("last4")
        if isinstance(last4, str) and last4.strip():
            return last4.strip()
        display = value.get("display")
        if isinstance(display, str) and display.strip():
            digits = re.findall(r"\d", display)
            if len(digits) >= 4:
                return "".join(digits[-4:])
            return display.strip().lower()
        for candidate in ("normalized", "raw", "value", "text"):
            if candidate in value:
                token = _normalize_account_number_token(value[candidate])
                if token:
                    return token
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        digits = re.findall(r"\d", text)
        if len(digits) >= 4:
            return "".join(digits[-4:])
        return text.lower()
    try:
        text = str(value).strip()
    except Exception:
        return None
    if not text:
        return None
    digits = re.findall(r"\d", text)
    if len(digits) >= 4:
        return "".join(digits[-4:])
    return text.lower()


def _normalize_text_fragment(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for candidate in ("normalized", "raw", "value", "text", "display"):
            if candidate in value:
                normalized = _normalize_text_fragment(value[candidate])
                if normalized:
                    return normalized
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return " ".join(text.split())


def _normalize_text_value(data: Mapping[str, Any]) -> str | None:
    for key in ("normalized", "raw"):
        if key in data:
            normalized = _normalize_text_fragment(data[key])
            if normalized:
                return normalized
    return None


def _normalize_account_number_value(data: Mapping[str, Any]) -> str | None:
    for key in ("normalized", "raw"):
        if key in data:
            token = _normalize_account_number_token(data[key])
            if token:
                return token
    return None


def _conditional_mismatch_metrics(
    field: str, pack_line: Mapping[str, Any]
) -> tuple[bool, int, list[str]]:
    records = _extract_bureau_records(pack_line)
    normalized_values: list[str] = []

    if field == "account_number_display":
        for record in records.values():
            token = _normalize_account_number_value(record)
            if token:
                normalized_values.append(token)
    else:
        for record in records.values():
            token = _normalize_text_value(record)
            if token:
                normalized_values.append(token)

    unique_values = sorted(set(normalized_values))
    mismatch = len(unique_values) >= 2
    corroboration = len(unique_values)
    return mismatch, corroboration, normalized_values


def _has_high_signal_creditor_remarks(values: Sequence[str]) -> bool:
    for value in values:
        if any(keyword in value for keyword in _CREDITOR_REMARK_KEYWORDS):
            return True
    return False


def _append_gate_note(rationale: str, reason: str) -> str:
    note = f"[conditional_gate:{reason}]"
    if not rationale:
        return note
    return f"{rationale} {note}"


def _append_guardrail_note(rationale: str, reason: str) -> str:
    note = f"[guardrail:{reason}]"
    if not rationale:
        return note
    return f"{rationale} {note}"


def _empty_decision_metrics() -> dict[str, dict[str, int]]:
    buckets = {"conditional": 0, "non_conditional": 0}
    return {
        "strong": dict(buckets),
        "weak": dict(buckets),
        "no_case": dict(buckets),
    }


def _enforce_conditional_gate(
    field: str,
    decision: str,
    rationale: str,
    pack_line: Mapping[str, Any],
) -> tuple[str, str, Mapping[str, Any] | None]:
    if field not in _CONDITIONAL_FIELDS:
        return decision, rationale, None
    if decision != "strong":
        return decision, rationale, None
    if not _coerce_bool_flag(pack_line.get("conditional_gate")):
        return decision, rationale, None

    min_corroboration = max(1, _coerce_int_value(pack_line.get("min_corroboration"), 1))
    mismatch, corroboration, normalized_values = _conditional_mismatch_metrics(
        field, pack_line
    )

    if field == "creditor_remarks" and mismatch:
        if not _has_high_signal_creditor_remarks(normalized_values):
            mismatch = False

    if not mismatch or corroboration < min_corroboration:
        gate_payload: dict[str, Any] = {
            "reason": "insufficient_evidence",
            "corroboration": corroboration,
            "unique_values": sorted(set(normalized_values)),
            "required_corroboration": min_corroboration,
        }
        if field == "creditor_remarks":
            gate_payload["high_signal_keywords"] = _CREDITOR_REMARK_KEYWORDS
        return (
            "no_case",
            _append_gate_note(rationale, "insufficient_evidence"),
            gate_payload,
        )

    return decision, rationale, None


class ValidationPackError(RuntimeError):
    """Raised when the Validation AI sender encounters a fatal error."""


def _ensure_requests_module() -> Any:
    """Load the ``requests`` module lazily so tests can stub it."""

    global requests
    if requests is not None:
        return requests

    try:
        requests = importlib.import_module("requests")
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive import
        raise ValidationPackError(
            "requests library is required to send validation packs"
        ) from exc

    return requests


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _truncate_response_body(body: Any, *, limit: int = 300) -> str:
    text = ""
    if isinstance(body, str):
        text = body
    elif body is None:
        text = ""
    else:
        text = str(body)
    text = text.strip()
    if len(text) > limit:
        return text[:limit]
    return text


@dataclass(slots=True)
@dataclass(frozen=True)
class _ChatCompletionResponse:
    """Wrapper around the chat completion response payload."""

    payload: Mapping[str, Any]
    status_code: int
    latency: float
    retries: int


class _ChatCompletionClient:
    """Minimal HTTP client for the OpenAI Chat Completions API."""

    def __init__(self, *, base_url: str, api_key: str, timeout: float | int):
        self.base_url = base_url.rstrip("/") or "https://api.openai.com/v1"
        self.api_key = api_key
        self.timeout: float | int = timeout

    def create(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        response_format: Mapping[str, Any],
        pack_id: str | None = None,
        on_error: Callable[[int, str], None] | None = None,
    ) -> _ChatCompletionResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        project_id = os.getenv("OPENAI_PROJECT_ID")
        if project_id:
            headers["OpenAI-Project"] = project_id
        include_beta_header = (
            isinstance(response_format, Mapping)
            and response_format.get("type") == "json_object"
        )
        if include_beta_header:
            headers["OpenAI-Beta"] = "response_format=v1"
        payload = {
            "model": model,
            "messages": list(messages),
            "response_format": dict(response_format),
        }
        request_lib = _ensure_requests_module()
        start_time = time.monotonic()
        response = request_lib.post(url, headers=headers, json=payload, timeout=self.timeout)
        latency = time.monotonic() - start_time
        status_code = getattr(response, "status_code", 0)
        try:
            body_text = response.text
        except Exception:  # pragma: no cover - defensive logging
            body_text = "<unavailable>"
        snippet = _truncate_response_body(body_text)

        def _record_error(status: int, body: str) -> None:
            if on_error is None:
                return
            try:
                on_error(status, body)
            except Exception:  # pragma: no cover - best effort logging
                log.exception(
                    "VALIDATION_HTTP_ERROR_SIDECAR_FAILED pack=%s",
                    pack_id or "<unknown>",
                )

        if not getattr(response, "ok", False):
            log.error(
                "VALIDATION_HTTP_ERROR status=%s body=%s pack=%s",
                status_code,
                snippet or "<empty>",
                pack_id or "<unknown>",
            )
            try:
                normalized_status = int(status_code)
            except (TypeError, ValueError):
                normalized_status = 0
            _record_error(normalized_status, snippet or "")
        response.raise_for_status()

        return _ChatCompletionResponse(
            payload=self._safe_json(
                response,
                pack_id=pack_id,
                snippet=snippet,
                status_code=status_code,
                on_error=_record_error,
            ),
            status_code=getattr(response, "status_code", 0),
            latency=latency,
            retries=0,
        )

    @staticmethod
    def _safe_json(
        response: Any,
        *,
        pack_id: str | None,
        snippet: str,
        status_code: Any,
        on_error: Callable[[int, str], None],
    ) -> Mapping[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            log.error("VALIDATION_EMPTY_CONTENT pack=%s", pack_id or "<unknown>")
            try:
                normalized_status = int(status_code)
            except (TypeError, ValueError):
                normalized_status = 0
            on_error(normalized_status, snippet or "")
            raise ValidationPackError("Response JSON parse failed") from exc
        if payload is None:
            log.error("VALIDATION_EMPTY_CONTENT pack=%s", pack_id or "<unknown>")
            try:
                normalized_status = int(status_code)
            except (TypeError, ValueError):
                normalized_status = 0
            on_error(normalized_status, snippet or "")
            raise ValidationPackError("Response JSON payload is empty")
        return payload


@dataclass(frozen=True)
class _ManifestView:
    """Resolved information about a validation manifest."""

    index: ValidationIndex
    log_path: Path


@dataclass(frozen=True)
class _PreflightAccount:
    """Resolved paths for a single manifest entry during validation."""

    record: ValidationPackRecord
    pack_path: Path
    result_jsonl_path: Path
    result_json_path: Path
    pack_missing: bool


@dataclass(frozen=True)
class _PreflightSummary:
    """Aggregate data produced by :meth:`ValidationPackSender._preflight`."""

    manifest_path: Path
    accounts: tuple[_PreflightAccount, ...]
    missing: int
    results_dir_created: bool
    parent_dirs_created: int


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_str(value: Any, default: str = "") -> str:
    if isinstance(value, str):
        text = value.strip()
        return text or default
    if value is None:
        return default
    return str(value)


def _index_path_from_mapping(document: Mapping[str, Any]) -> Path:
    index_path_override = document.get("__index_path__")
    if index_path_override:
        return Path(str(index_path_override)).resolve()

    base_dir_override = (
        document.get("__base_dir__")
        or document.get("__manifest_dir__")
        or document.get("__index_dir__")
    )
    if base_dir_override:
        base_dir = Path(str(base_dir_override)).resolve()
    else:
        base_dir = Path.cwd()

    filename = _coerce_str(document.get("__index_filename__"), default="index.json")
    return (base_dir / filename).resolve()


def _index_from_document(document: Mapping[str, Any], *, index_path: Path) -> ValidationIndex:
    schema_version = _coerce_int(document.get("schema_version"), default=0)
    if schema_version < 2:
        document = _convert_v1_document(document, index_path=index_path)
        schema_version = 2

    sid = _coerce_str(document.get("sid"))
    if not sid:
        raise ValidationPackError("Validation manifest is missing 'sid'")

    root = _coerce_str(document.get("root"), default=".") or "."
    packs_dir = _coerce_str(document.get("packs_dir"), default="packs") or "packs"
    results_dir = _coerce_str(document.get("results_dir"), default="results") or "results"

    raw_packs = document.get("packs")
    records: list[ValidationPackRecord] = []
    if isinstance(raw_packs, Sequence):
        for entry in raw_packs:
            if isinstance(entry, Mapping):
                records.append(ValidationPackRecord.from_mapping(entry))

    return ValidationIndex(
        index_path=index_path,
        sid=sid,
        root=root,
        packs_dir=packs_dir,
        results_dir=results_dir,
        packs=records,
        schema_version=schema_version,
    )


def _convert_v1_document(
    document: Mapping[str, Any], *, index_path: Path
) -> Mapping[str, Any]:
    """Convert a legacy v1 manifest document to the v2 schema."""

    base_dir = index_path.parent.resolve()
    sid = _coerce_str(document.get("sid"))

    raw_items = document.get("packs") or document.get("items")
    records: list[dict[str, Any]] = []
    if isinstance(raw_items, Sequence):
        for entry in raw_items:
            if not isinstance(entry, Mapping):
                continue
            record: dict[str, Any] = dict(entry)
            record["pack"] = _to_relative(
                entry.get("pack_path")
                or entry.get("pack")
                or entry.get("pack_file")
                or entry.get("pack_filename"),
                base_dir,
            )
            record["result_jsonl"] = _to_relative(
                entry.get("result_jsonl_path")
                or entry.get("result_jsonl")
                or entry.get("result_jsonl_file"),
                base_dir,
            )
            record["result_json"] = _to_relative(
                entry.get("result_path")
                or entry.get("result_summary_path")
                or entry.get("result_json"),
                base_dir,
            )
            records.append(record)

    return {
        "schema_version": 2,
        "sid": sid,
        "root": ".",
        "packs_dir": "packs",
        "results_dir": "results",
        "packs": records,
    }


def _to_relative(path_value: Any, base_dir: Path) -> str:
    """Return a POSIX relative path for ``path_value`` with ``base_dir`` as the anchor."""

    text = _coerce_str(path_value)
    if not text:
        return ""

    candidate = Path(text)
    if not candidate.is_absolute():
        return PurePosixPath(text).as_posix()

    try:
        candidate_resolved = candidate.resolve()
    except OSError:
        candidate_resolved = candidate

    base_resolved = base_dir.resolve()
    try:
        relative = candidate_resolved.relative_to(base_resolved)
    except ValueError:
        try:
            relative = Path(os.path.relpath(candidate_resolved, base_resolved))
        except (OSError, ValueError):
            return candidate_resolved.as_posix()

    return PurePosixPath(relative).as_posix()


def _resolve_log_path(index_path: Path, document: Mapping[str, Any] | None) -> Path:
    index_dir = index_path.parent.resolve()
    candidate: str | None = None

    if document:
        logs_section = document.get("logs")
        if isinstance(logs_section, Mapping):
            for key in ("send", "sender", "log", "log_path", "path"):
                value = logs_section.get(key)
                if isinstance(value, str) and value.strip():
                    candidate = value.strip()
                    break

        if candidate is None:
            for key in ("log", "log_path"):
                value = document.get(key)
                if isinstance(value, str) and value.strip():
                    candidate = value.strip()
                    break

    if candidate:
        return (index_dir / PurePosixPath(candidate)).resolve()

    return index_dir / "send.log"


def _load_manifest_view(
    manifest: Mapping[str, Any] | ValidationIndex | Path | str,
) -> _ManifestView:
    if isinstance(manifest, ValidationIndex):
        index = manifest
        log_path = _resolve_log_path(index.index_path, None)
        return _ManifestView(index=index, log_path=log_path)

    if isinstance(manifest, Mapping):
        index_path = _index_path_from_mapping(manifest)
        index = _index_from_document(manifest, index_path=index_path)
        log_path = _resolve_log_path(index_path, manifest)
        return _ManifestView(index=index, log_path=log_path)

    manifest_path = Path(manifest)
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValidationPackError(f"Validation index missing: {manifest_path}") from exc
    except OSError as exc:
        raise ValidationPackError(
            f"Unable to read validation index: {manifest_path}"
        ) from exc

    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationPackError(
            f"Validation index is not valid JSON: {manifest_path}"
        ) from exc

    if not isinstance(document, Mapping):
        raise ValidationPackError("Validation index root must be an object")

    index = _index_from_document(document, index_path=manifest_path)
    log_path = _resolve_log_path(manifest_path, document)
    return _ManifestView(index=index, log_path=log_path)


class ValidationPackSender:
    """Send validation packs and store the adjudication results."""

    def __init__(
        self,
        manifest: Mapping[str, Any] | ValidationIndex | Path | str,
        *,
        http_client: _ChatCompletionClient | None = None,
    ) -> None:
        view = _load_manifest_view(manifest)
        self._index = view.index
        self.sid = self._index.sid

        raw_model = os.getenv("AI_MODEL")
        if raw_model is None or not str(raw_model).strip():
            fallback_model = os.getenv("VALIDATION_MODEL")
            if fallback_model is not None:
                raw_model = fallback_model
        if raw_model is None:
            raw_model = _DEFAULT_MODEL
        self.model = str(raw_model).strip()
        self._client = http_client or self._build_client()
        self._throttle = _THROTTLE_SECONDS
        self._results_root: Path | None = None
        self._log_path = view.log_path
        self._confidence_threshold = _confidence_threshold()
        self._default_queue = (
            self._infer_queue_hint(self._index.packs) or _DEFAULT_QUEUE_NAME
        )

    # ------------------------------------------------------------------
    # Queue routing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _queue_from_record(record: ValidationPackRecord) -> str | None:
        extra = record.extra
        if isinstance(extra, Mapping):
            for key in ("queue", "celery_queue", "task_queue", "target_queue"):
                value = extra.get(key)
                if isinstance(value, str):
                    text = value.strip()
                    if text:
                        return text
        return None

    def _infer_queue_hint(
        self, records: Sequence[ValidationPackRecord]
    ) -> str | None:
        counts: dict[str, int] = {}
        for record in records:
            queue = self._queue_from_record(record)
            if not queue:
                continue
            counts[queue] = counts.get(queue, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda item: (item[1], item[0]))[0]

    def _queue_plan(self, accounts: Sequence[_PreflightAccount]) -> dict[str, int]:
        plan: dict[str, int] = {}
        for account in accounts:
            queue = self._queue_from_record(account.record) or self._default_queue
            plan[queue] = plan.get(queue, 0) + 1
        return plan

    def _select_queue(self, plan: Mapping[str, int]) -> str:
        if not plan:
            return self._default_queue
        if self._default_queue in plan:
            return self._default_queue
        return max(plan.items(), key=lambda item: (item[1], item[0]))[0]

    # ------------------------------------------------------------------
    # Pre-flight validation
    # ------------------------------------------------------------------
    def _preflight(self, index: ValidationIndex) -> _PreflightSummary:
        manifest_path = index.index_path
        accounts: list[_PreflightAccount] = []

        results_dir = index.results_dir_path
        results_dir_exists = results_dir.exists()
        if not results_dir_exists:
            results_dir.mkdir(parents=True, exist_ok=True)

        created_parent_dirs: set[Path] = set()

        for record in index.packs:
            pack_path = index.resolve_pack_path(record)
            result_jsonl_path = index.resolve_result_jsonl_path(record)
            result_json_path = index.resolve_result_json_path(record)

            for candidate in (result_jsonl_path.parent, result_json_path.parent):
                if candidate.exists():
                    continue
                candidate.mkdir(parents=True, exist_ok=True)
                created_parent_dirs.add(candidate.resolve())

            accounts.append(
                _PreflightAccount(
                    record=record,
                    pack_path=pack_path,
                    result_jsonl_path=result_jsonl_path,
                    result_json_path=result_json_path,
                    pack_missing=not pack_path.is_file(),
                )
            )

        missing = sum(1 for account in accounts if account.pack_missing)
        summary = _PreflightSummary(
            manifest_path=manifest_path,
            accounts=tuple(accounts),
            missing=missing,
            results_dir_created=not results_dir_exists,
            parent_dirs_created=len(created_parent_dirs),
        )
        self._print_preflight_summary(index, summary)
        return summary

    def _print_preflight_summary(
        self, index: ValidationIndex, summary: _PreflightSummary
    ) -> None:
        manifest_display = self._display_path(summary.manifest_path)
        print(f"MANIFEST: {manifest_display}")
        print(f"PACKS: {len(summary.accounts)}, missing: {summary.missing}")

        results_status = "ok"
        if summary.results_dir_created:
            results_status = "created"
        elif summary.parent_dirs_created:
            results_status = f"ok (created {summary.parent_dirs_created} dirs)"
        print(f"RESULTS DIR: {results_status}")

        missing_accounts = {
            account.record.account_id for account in summary.accounts if account.pack_missing
        }
        for account in summary.accounts:
            record = account.record
            account_id = record.account_id
            pack_display = record.pack or self._display_path(
                account.pack_path, base=index.index_dir
            )
            jsonl_display = record.result_jsonl or self._display_path(
                account.result_jsonl_path, base=index.index_dir
            )
            summary_display = record.result_json or self._display_path(
                account.result_json_path, base=index.index_dir
            )

            line = (
                f"[acc={account_id:03d}] pack={pack_display} -> "
                f"{jsonl_display}, {summary_display}  (lines={record.lines})"
            )
            if account_id in missing_accounts:
                missing_path = self._display_path(
                    account.pack_path, base=index.index_dir
                )
                line += f"  [MISSING: {missing_path}]"
            print(line)

    @staticmethod
    def _display_path(path: Path, *, base: Path | None = None) -> str:
        candidate = path.resolve()
        anchors: list[Path] = []
        if base is not None:
            anchors.append(base.resolve())
        anchors.append(Path.cwd().resolve())
        for anchor in anchors:
            try:
                relative = candidate.relative_to(anchor)
                return PurePosixPath(relative).as_posix() or "."
            except ValueError:
                try:
                    relpath = Path(os.path.relpath(candidate, anchor))
                except (OSError, ValueError):
                    continue
                return PurePosixPath(relpath).as_posix()
        return candidate.as_posix()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _log_preflight_line(self, index: ValidationIndex) -> None:
        packs_dir = index.packs_dir_path
        try:
            pack_count = sum(1 for _ in packs_dir.glob("val_acc_*.jsonl")) if packs_dir.exists() else 0
        except OSError as exc:  # pragma: no cover - defensive logging
            log.warning(
                "VALIDATION_SEND_PREFLIGHT_GLOB_FAILED sid=%s packs_dir=%s error=%s",
                self.sid,
                str(packs_dir),
                exc,
            )
            pack_count = 0

        api_key_set = bool(os.getenv("OPENAI_API_KEY"))
        base_url_set = bool(os.getenv("OPENAI_BASE_URL"))

        log.info(
            "VALIDATION_SEND_PREFLIGHT sid=%s model=%s packs_dir=%s pack_files=%s env_api_key=%s env_base_url=%s",
            self.sid,
            self.model or "<missing>",
            str(packs_dir),
            pack_count,
            "set" if api_key_set else "missing",
            "set" if base_url_set else "missing",
        )

        log.info(
            "VALIDATION_PACKS_DIR_USED sid=%s dir=%s",
            self.sid,
            str(index.packs_dir_path),
        )
        log.info(
            "VALIDATION_RESULTS_DIR_USED sid=%s dir=%s",
            self.sid,
            str(index.results_dir_path),
        )

    def send(self) -> list[dict[str, Any]]:
        """Send every pack referenced by the manifest index."""

        index = self._load_index()
        self._log_preflight_line(index)

        if not self.model:
            log.error(
                "VALIDATION_SEND_MODEL_MISSING sid=%s manifest=%s detail=%s",
                self.sid,
                str(index.index_path),
                "no model configured",
            )
            return []

        if not index.packs:
            log.warning(
                "VALIDATION_SEND_NO_PACKS sid=%s manifest=%s detail=%s",
                self.sid,
                str(index.index_path),
                "no eligible packs found",
            )
            return []

        preflight = self._preflight(index)
        self._results_root = index.results_dir_path

        log.info(
            "VALIDATION_SEND_DISCOVERY sid=%s manifest=%s packs=%s missing=%s results_dir=%s",
            self.sid,
            str(index.index_path),
            len(preflight.accounts),
            preflight.missing,
            str(index.results_dir_path),
        )

        dispatchable_accounts = [
            account for account in preflight.accounts if not account.pack_missing
        ]
        queue_plan = self._queue_plan(dispatchable_accounts)
        total_enqueued = sum(queue_plan.values())
        missing_accounts = sum(1 for account in preflight.accounts if account.pack_missing)
        default_queue = self._select_queue(queue_plan)
        queue_plan_display = ", ".join(
            f"{name}:{count}" for name, count in sorted(queue_plan.items())
        )
        if not queue_plan_display:
            queue_plan_display = "none"

        if total_enqueued:
            log.info(
                "VALIDATION_SEND_QUEUE_PLAN sid=%s queue=%s routed=%s missing=%s routes=%s",
                self.sid,
                default_queue,
                total_enqueued,
                missing_accounts,
                queue_plan_display,
            )
        else:
            log.warning(
                "VALIDATION_SEND_QUEUE_PLAN sid=%s queue=%s routed=%s missing=%s routes=%s",
                self.sid,
                default_queue,
                total_enqueued,
                missing_accounts,
                queue_plan_display,
            )

        self._log(
            "send_queue_plan",
            queue=default_queue,
            total_accounts=len(preflight.accounts),
            missing_accounts=missing_accounts,
            routed_accounts=total_enqueued,
            routes=dict(queue_plan),
        )

        results: list[dict[str, Any]] = []
        for account in preflight.accounts:
            record = account.record
            account_id = record.account_id
            try:
                normalized_account = self._normalize_account_id(account_id)
            except ValueError:
                normalized_account = None

            pack_relative = record.pack
            result_jsonl_relative = record.result_jsonl
            result_json_relative = record.result_json

            resolved_pack = account.pack_path
            result_jsonl_path = account.result_jsonl_path
            result_json_path = account.result_json_path

            account_label = (
                f"{normalized_account:03d}"
                if normalized_account is not None
                else str(account_id)
            )

            if account.pack_missing:
                missing_display = pack_relative or self._display_path(
                    resolved_pack, base=index.index_dir
                )
                error_message = f"Pack file missing: {missing_display}"
                log.warning(
                    "VALIDATION_PACK_MISSING sid=%s account_id=%s pack=%s",
                    self.sid,
                    account_label,
                    missing_display,
                )
                log.error(
                    "VALIDATION_SEND_ACCOUNT_FAILED sid=%s account_id=%s pack=%s exc_type=%s message=%s line_ids=%s",
                    self.sid,
                    account_label,
                    str(resolved_pack),
                    "FileNotFoundError",
                    error_message,
                    [],
                )
                if normalized_account is None:
                    self._log(
                        "send_account_failed",
                        account_id=str(account_id),
                        error=error_message,
                    )
                    continue

                self._log(
                    "send_account_failed",
                    account_id=f"{normalized_account:03d}",
                    error=error_message,
                )
                account_summary = self._record_account_failure(
                    normalized_account,
                    resolved_pack,
                    pack_relative,
                    result_jsonl_path,
                    result_jsonl_relative,
                    result_json_path,
                    result_json_relative,
                    error_message,
                )
                results.append(account_summary)
                continue

            log.info(
                "PROCESSING account_id=%s pack=%s result_json=%s",
                account_label,
                pack_relative
                or self._display_path(resolved_pack, base=index.index_dir),
                result_json_relative
                or self._display_path(result_json_path, base=index.index_dir),
            )

            try:
                account_summary = self._process_account(
                    account_id,
                    normalized_account,
                    resolved_pack,
                    pack_relative,
                    result_jsonl_path,
                    result_jsonl_relative,
                    result_json_path,
                    result_json_relative,
                )
            except ValidationPackError as exc:
                failed_line_ids = getattr(exc, "line_ids", []) or []
                log.error(
                    "VALIDATION_SEND_ACCOUNT_FAILED sid=%s account_id=%s pack=%s exc_type=%s message=%s line_ids=%s",
                    self.sid,
                    account_label,
                    str(resolved_pack),
                    type(exc).__name__,
                    exc,
                    failed_line_ids,
                )
                if normalized_account is None:
                    self._log(
                        "send_account_failed",
                        account_id=str(account_id),
                        error=str(exc),
                    )
                    continue
                self._log(
                    "send_account_failed",
                    account_id=f"{normalized_account:03d}",
                    error=str(exc),
                )
                account_summary = self._record_account_failure(
                    normalized_account,
                    resolved_pack,
                    pack_relative,
                    result_jsonl_path,
                    result_jsonl_relative,
                    result_json_path,
                    result_json_relative,
                    str(exc),
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                failed_line_ids = getattr(exc, "line_ids", []) or []
                log.error(
                    "VALIDATION_SEND_ACCOUNT_FAILED sid=%s account_id=%s pack=%s exc_type=%s message=%s line_ids=%s",
                    self.sid,
                    account_label,
                    str(resolved_pack),
                    type(exc).__name__,
                    exc,
                    failed_line_ids,
                )
                log.exception(
                    "VALIDATION_PACK_ACCOUNT_UNEXPECTED sid=%s account=%s line_ids=%s",
                    self.sid,
                    account_label,
                    failed_line_ids,
                )
                if normalized_account is None:
                    self._log(
                        "send_account_failed",
                        account_id=str(account_id),
                        error=str(exc),
                    )
                    continue
                self._log(
                    "send_account_failed",
                    account_id=f"{normalized_account:03d}",
                    error=str(exc),
                )
                account_summary = self._record_account_failure(
                    normalized_account,
                    resolved_pack,
                    pack_relative,
                    result_jsonl_path,
                    result_jsonl_relative,
                    result_json_path,
                    result_json_relative,
                    str(exc),
                )
            results.append(account_summary)
        return results

    # ------------------------------------------------------------------
    # Account processing
    # ------------------------------------------------------------------
    def _process_account(
        self,
        account_id: Any,
        normalized_account: int | None,
        pack_path: Path,
        pack_display: str,
        result_jsonl_path: Path,
        result_jsonl_display: str,
        result_summary_path: Path,
        result_summary_display: str,
    ) -> dict[str, Any]:
        account_int = normalized_account
        if account_int is None:
            raise ValidationPackError(f"Account id is not numeric: {account_id!r}")

        pack_identifier = f"acc_{account_int:03d}"
        error_filename = validation_result_error_filename_for_account(account_int)
        error_path = result_summary_path.with_name(error_filename)
        self._clear_error_sidecar(error_path, pack_id=pack_identifier)

        try:
            pack_lines = list(
                self._iter_pack_lines(pack_path, display_path=pack_display)
            )
        except ValidationPackError:
            self._log(
                "send_account_start",
                account_id=f"{account_int:03d}",
                pack=pack_display,
                pack_absolute=str(pack_path),
                lines=0,
            )
            raise

        log.info(
            "VALIDATION_SEND_ACCOUNT_START sid=%s account_id=%03d pack=%s lines=%s results=%s",
            self.sid,
            account_int,
            str(pack_path),
            len(pack_lines),
            str(result_summary_path),
        )

        result_lines: list[dict[str, Any]] = []
        errors: list[str] = []
        total_fields = len(pack_lines)
        fields_sent = 0
        conditional_sent = 0
        decision_metrics = _empty_decision_metrics()
        failed_line_ids: list[str] = []
        current_line_id: str | None = None
        start_time = time.monotonic()

        self._log(
            "send_account_start",
            account_id=f"{account_int:03d}",
            pack=pack_display,
            pack_absolute=str(pack_path),
            lines=len(pack_lines),
        )

        try:
            for idx, pack_line in enumerate(pack_lines, start=1):
                field_name = self._coerce_field_name(pack_line, idx)
                current_line_id = self._coerce_identifier(
                    account_int, idx, pack_line.get("id")
                )
                if not self._is_allowed_field(field_name):
                    self._log(
                        "send_line_skipped",
                        account_id=f"{account_int:03d}",
                        line_number=idx,
                        field=field_name,
                        reason="field_not_allowed",
                    )
                    continue
                try:
                    response = self._call_model(
                        pack_line,
                        account_id=account_int,
                        account_label=f"{account_int:03d}",
                        line_number=idx,
                        line_id=current_line_id,
                        pack_id=pack_identifier,
                        error_path=error_path,
                    )
                except Exception as exc:
                    error_message = (
                        "AI request failed for acc "
                        f"{account_int:03d} pack={pack_display} -> "
                        f"{result_jsonl_display}, {result_summary_display}: {exc}"
                    )
                    errors.append(error_message)
                    log.error(
                        "VALIDATION_SEND_MODEL_ERROR sid=%s account_id=%03d line_id=%s "
                        "exc_type=%s message=%s",
                        self.sid,
                        account_int,
                        current_line_id,
                        type(exc).__name__,
                        exc,
                    )
                    response = self._fallback_response(error_message)
                line_result, metadata = self._build_result_line(
                    account_int, idx, pack_line, response
                )
                result_lines.append(line_result)
                fields_sent += 1
                is_conditional = bool(metadata.get("conditional"))
                bucket = "conditional" if is_conditional else "non_conditional"
                if is_conditional:
                    conditional_sent += 1

                gate_info = metadata.get("gate_info") or None
                if gate_info:
                    decision_metrics["weak"][bucket] += 1
                    gate_log: dict[str, Any] = {
                        "account_id": f"{account_int:03d}",
                        "line_number": idx,
                        "field": metadata.get("field"),
                        "reason": gate_info.get("reason"),
                        "original_decision": metadata.get("original_decision"),
                        "final_decision": metadata.get("final_decision"),
                        "corroboration": gate_info.get("corroboration"),
                        "required_corroboration": gate_info.get("required_corroboration"),
                    }
                    unique_values = gate_info.get("unique_values")
                    if unique_values is not None:
                        gate_log["unique_values"] = unique_values
                    if "high_signal_keywords" in gate_info:
                        gate_log["high_signal_keywords"] = gate_info["high_signal_keywords"]
                    self._log("send_conditional_gate_downgrade", **gate_log)
                else:
                    final_decision = metadata.get("final_decision", "no_case")
                    decision_metrics.setdefault(
                        final_decision, {"conditional": 0, "non_conditional": 0}
                    )
                    decision_metrics[final_decision][bucket] += 1
                time.sleep(self._throttle)
        except ValidationPackError as exc:
            if current_line_id:
                failed_line_ids.append(current_line_id)
            if not getattr(exc, "line_ids", None) and failed_line_ids:
                setattr(exc, "line_ids", list(dict.fromkeys(failed_line_ids)))
            raise
        except Exception as exc:
            if current_line_id:
                failed_line_ids.append(current_line_id)
            if failed_line_ids and not getattr(exc, "line_ids", None):
                setattr(exc, "line_ids", list(dict.fromkeys(failed_line_ids)))
            raise

        status = "error" if errors else "done"
        error_message = "; ".join(errors) if errors else None
        metrics_payload = {
            "total_fields": total_fields,
            "fields_sent": fields_sent,
            "fields_skipped": max(total_fields - fields_sent, 0),
            "conditional_fields_sent": conditional_sent,
            "decision_counts": decision_metrics,
        }
        jsonl_path, summary_path = self._write_results(
            account_int,
            result_lines,
            status=status,
            error=error_message,
            jsonl_path=result_jsonl_path,
            jsonl_display=result_jsonl_display,
            summary_path=result_summary_path,
            summary_display=result_summary_display,
        )

        summary_payload: dict[str, Any] = {
            "sid": self.sid,
            "account_id": account_int,
            "pack_path": str(pack_path),
            "pack_manifest_path": pack_display,
            "results_path": str(summary_path),
            "results_manifest_path": result_summary_display,
            "jsonl_path": str(jsonl_path),
            "jsonl_manifest_path": result_jsonl_display,
            "status": status,
            "model": self.model,
            "request_lines": len(pack_lines),
            "results": result_lines,
            "completed_at": _utc_now(),
        }
        if error_message:
            summary_payload["error"] = error_message
        summary_payload["metrics"] = metrics_payload

        self._log(
            "send_account_done",
            account_id=f"{account_int:03d}",
            status=status,
            errors=len(errors),
            results=len(result_lines),
        )
        self._log(
            "send_account_metrics",
            account_id=f"{account_int:03d}",
            **metrics_payload,
        )
        duration = time.monotonic() - start_time
        log.info(
            "VALIDATION_SEND_ACCOUNT_END sid=%s account_id=%03d status=%s results=%s duration=%.3fs",
            self.sid,
            account_int,
            status,
            len(result_lines),
            duration,
        )
        return summary_payload

    def _record_account_failure(
        self,
        account_id: int,
        pack_path: Path,
        pack_display: str,
        result_jsonl_path: Path,
        result_jsonl_display: str,
        result_summary_path: Path,
        result_summary_display: str,
        error: str,
    ) -> dict[str, Any]:
        jsonl_path, summary_path = self._write_results(
            account_id,
            [],
            status="error",
            error=error,
            jsonl_path=result_jsonl_path,
            jsonl_display=result_jsonl_display,
            summary_path=result_summary_path,
            summary_display=result_summary_display,
        )
        summary_payload = {
            "sid": self.sid,
            "account_id": account_id,
            "pack_path": str(pack_path),
            "pack_manifest_path": pack_display,
            "results_path": str(summary_path),
            "results_manifest_path": result_summary_display,
            "jsonl_path": str(jsonl_path),
            "jsonl_manifest_path": result_jsonl_display,
            "status": "error",
            "model": self.model,
            "request_lines": 0,
            "results": [],
            "completed_at": _utc_now(),
            "error": error,
        }
        metrics_payload = {
            "total_fields": 0,
            "fields_sent": 0,
            "fields_skipped": 0,
            "conditional_fields_sent": 0,
            "decision_counts": _empty_decision_metrics(),
        }
        summary_payload["metrics"] = metrics_payload
        self._log(
            "send_account_done",
            account_id=f"{account_id:03d}",
            status="error",
            errors=1,
            results=0,
        )
        self._log(
            "send_account_metrics",
            account_id=f"{account_id:03d}",
            **metrics_payload,
        )
        return summary_payload

    def _call_model(
        self,
        pack_line: Mapping[str, Any],
        *,
        account_id: int,
        account_label: str,
        line_number: int,
        line_id: str,
        pack_id: str,
        error_path: Path,
    ) -> Mapping[str, Any]:
        prompt = pack_line.get("prompt")
        if not isinstance(prompt, Mapping):
            raise ValidationPackError("Pack line missing prompt")

        system_prompt = str(prompt.get("system") or "")
        user_payload = prompt.get("user")
        try:
            user_message = json.dumps(user_payload, ensure_ascii=False, sort_keys=True)
        except Exception as exc:
            raise ValidationPackError(f"Unable to serialise user payload: {exc}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        def _record_sidecar(status: int, body: str) -> None:
            try:
                normalized_status = int(status)
            except (TypeError, ValueError):
                normalized_status = 0
            payload = {
                "status": normalized_status,
                "body": body,
                "pack_id": pack_id,
            }
            self._write_error_sidecar(error_path, payload)

        response = self._client.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            pack_id=pack_id,
            on_error=_record_sidecar,
        )
        if hasattr(response, "payload"):
            payload = response.payload  # type: ignore[assignment]
            status_code = getattr(response, "status_code", 0)
            latency = getattr(response, "latency", 0.0)
            retries = getattr(response, "retries", 0)
        elif isinstance(response, Mapping):
            payload = response
            status_code = int(getattr(response, "status_code", 0))
            latency_raw = getattr(response, "latency", 0.0)
            try:
                latency = float(latency_raw)
            except (TypeError, ValueError):
                latency = 0.0
            retries = int(getattr(response, "retries", 0))
        else:
            raise ValidationPackError(
                f"Model client returned unsupported response type: {type(response)!r}"
            )
        log.info(
            "VALIDATION_SEND_MODEL_CALL sid=%s account_id=%s line_id=%s line_number=%s status=%s latency=%.3fs retries=%s",
            self.sid,
            account_label,
            line_id,
            line_number,
            status_code,
            latency,
            retries,
        )

        choices = payload.get("choices")
        if not isinstance(choices, Sequence) or not choices:
            log.error("VALIDATION_EMPTY_CONTENT pack=%s", pack_id)
            serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            _record_sidecar(int(status_code), serialized)
            raise ValidationPackError("Model response missing choices")
        message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            log.error("VALIDATION_EMPTY_CONTENT pack=%s", pack_id)
            serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            _record_sidecar(int(status_code), serialized)
            raise ValidationPackError("Model response missing content")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            log.error("VALIDATION_EMPTY_CONTENT pack=%s", pack_id)
            _record_sidecar(int(status_code), content)
            raise ValidationPackError(f"Model response is not valid JSON: {exc}")
        if not isinstance(parsed, Mapping):
            raise ValidationPackError("Model response is not an object")
        return parsed

    # ------------------------------------------------------------------
    # Guardrails
    # ------------------------------------------------------------------
    def _validate_response_payload(
        self, response: Mapping[str, Any]
    ) -> tuple[dict[str, Any] | None, list[str]]:
        if not isinstance(response, Mapping):
            return None, ["response_not_mapping"]

        errors = [error.message for error in _RESPONSE_VALIDATOR.iter_errors(response)]
        if errors:
            return None, errors

        justification = self._normalize_justification(response.get("justification"))
        labels = self._normalize_labels(response.get("labels"))
        if not justification:
            return None, ["empty_justification"]
        if not labels:
            return None, ["empty_labels"]

        normalized = {
            "decision": self._normalize_decision(response.get("decision")),
            "justification": justification,
            "citations": self._normalize_citations(response.get("citations")),
            "confidence": self._normalize_confidence(response.get("confidence")),
            "labels": labels,
        }

        return normalized, []

    # ------------------------------------------------------------------
    # Result construction & persistence
    # ------------------------------------------------------------------
    def _build_result_line(
        self,
        account_id: int,
        line_number: int,
        pack_line: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        line_id = self._coerce_identifier(account_id, line_number, pack_line.get("id"))
        field = self._coerce_field_name(pack_line, line_number)
        normalized_response, schema_errors = self._validate_response_payload(response)

        guardrail_info: dict[str, Any] | None = None
        if normalized_response is None:
            emit_counter("validation.ai.response_invalid")
            log.warning(
                "VALIDATION_AI_RESPONSE_INVALID field=%s errors=%s",
                field,
                schema_errors,
            )
            original_decision = self._normalize_decision(response.get("decision"))
            decision = "no_case"
            fallback_text = self._normalize_justification(response.get("rationale"))
            rationale = _append_guardrail_note(fallback_text, "invalid_response")
            citations: list[str] = []
            confidence: float | None = None
            labels: list[str] = []
            guardrail_info = {"reason": "invalid_response", "errors": schema_errors}
        else:
            decision = normalized_response["decision"]
            original_decision = decision
            rationale = normalized_response["justification"]
            citations = normalized_response["citations"]
            confidence = normalized_response["confidence"]
            labels = normalized_response["labels"]

            if (
                decision == "strong"
                and confidence is not None
                and confidence < self._confidence_threshold
            ):
                emit_counter("validation.ai.response_low_confidence")
                log.warning(
                    "VALIDATION_AI_LOW_CONFIDENCE field=%s confidence=%.6f threshold=%.6f",
                    field,
                    confidence,
                    self._confidence_threshold,
                )
                guardrail_info = {
                    "reason": "low_confidence",
                    "confidence": confidence,
                    "threshold": self._confidence_threshold,
                }
                decision = "no_case"
                rationale = _append_guardrail_note(rationale, "low_confidence")

        decision, rationale, gate_info = _enforce_conditional_gate(
            field, decision, rationale, pack_line
        )

        result: dict[str, Any] = {
            "id": line_id,
            "account_id": account_id,
            "field": field,
            "decision": decision,
            "rationale": rationale,
            "citations": citations,
        }
        if confidence is not None:
            result["confidence"] = confidence
        if normalized_response is not None and labels:
            result["labels"] = labels

        metadata: dict[str, Any] = {
            "field": field,
            "final_decision": decision,
            "original_decision": original_decision,
            "conditional": field in _CONDITIONAL_FIELDS,
            "gate_info": gate_info,
        }
        if normalized_response is not None and labels:
            metadata["labels"] = labels
        if guardrail_info is not None:
            metadata["guardrail"] = guardrail_info
            log_payload: dict[str, Any] = {
                "account_id": f"{account_id:03d}",
                "line_number": line_number,
                "field": field,
                "reason": guardrail_info.get("reason"),
            }
            if "confidence" in guardrail_info:
                log_payload["confidence"] = guardrail_info["confidence"]
            if "threshold" in guardrail_info:
                log_payload["threshold"] = guardrail_info["threshold"]
            if "errors" in guardrail_info:
                log_payload["errors"] = guardrail_info["errors"]
            self._log("send_guardrail_triggered", **log_payload)
        return result, metadata

    def _write_results(
        self,
        account_id: int,
        result_lines: Sequence[Mapping[str, Any]],
        *,
        status: str = "done",
        error: str | None = None,
        jsonl_path: Path | None = None,
        jsonl_display: str | None = None,
        summary_path: Path | None = None,
        summary_display: str | None = None,
    ) -> tuple[Path, Path]:
        results_root = self._results_root or self._index.results_dir_path
        results_root.mkdir(parents=True, exist_ok=True)

        if jsonl_path is None:
            jsonl_path = (
                results_root
                / validation_result_jsonl_filename_for_account(account_id)
            )

        if summary_path is None:
            summary_path = (
                results_root
                / validation_result_summary_filename_for_account(account_id)
            )

        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        jsonl_lines = [
            json.dumps(line, ensure_ascii=False, sort_keys=True) for line in result_lines
        ]
        jsonl_payload = "\n".join(jsonl_lines)
        if jsonl_payload:
            jsonl_payload += "\n"
        jsonl_path.write_text(jsonl_payload, encoding="utf-8")

        summary_payload: dict[str, Any] = {
            "sid": self.sid,
            "account_id": account_id,
            "status": status,
            "model": self.model,
            "request_lines": len(result_lines),
            "completed_at": _utc_now(),
            "results": list(result_lines),
        }
        if error:
            summary_payload["error"] = error
        summary_path.write_text(
            json.dumps(summary_payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._log(
            "send_account_results",
            account_id=f"{account_id:03d}",
            jsonl=jsonl_display or jsonl_path.name,
            jsonl_absolute=str(jsonl_path.resolve()),
            summary=summary_display or summary_path.name,
            summary_absolute=str(summary_path.resolve()),
            results=len(result_lines),
            status=status,
        )
        log.info(
            "VALIDATION_SEND_RESULTS_WRITTEN sid=%s account_id=%03d jsonl=%s summary=%s decisions=%s status=%s",
            self.sid,
            account_id,
            str(jsonl_path),
            str(summary_path),
            len(result_lines),
            status,
        )
        return jsonl_path, summary_path

    def _write_error_sidecar(self, path: Path, payload: Mapping[str, Any]) -> None:
        data: dict[str, Any] = dict(payload)
        body_value = data.get("body", "")
        if not isinstance(body_value, str):
            body_value = "" if body_value is None else str(body_value)
        data["body"] = _truncate_response_body(body_value)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            log.exception(
                "VALIDATION_ERROR_SIDECAR_WRITE_FAILED sid=%s pack=%s path=%s error=%s",
                self.sid,
                data.get("pack_id", "<unknown>"),
                str(path),
                exc,
            )

    def _clear_error_sidecar(self, path: Path, *, pack_id: str) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:  # pragma: no cover - defensive logging
            log.warning(
                "VALIDATION_ERROR_SIDECAR_CLEANUP_FAILED sid=%s pack=%s path=%s error=%s",
                self.sid,
                pack_id,
                str(path),
                exc,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _iter_pack_lines(
        self, pack_path: Path, *, display_path: str | None = None
    ) -> Iterable[Mapping[str, Any]]:
        display = display_path or str(pack_path)
        try:
            text = pack_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ValidationPackError(f"Pack file missing: {display}") from exc
        except OSError as exc:
            raise ValidationPackError(f"Unable to read pack file: {display}") from exc

        lines: list[Mapping[str, Any]] = []
        for idx, raw in enumerate(text.splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValidationPackError(
                    f"Invalid JSON in pack line {idx} of {display}: {exc}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise ValidationPackError(
                    f"Pack line {idx} of {display} is not an object"
                )
            lines.append(payload)
        return lines

    def _build_client(self) -> _ChatCompletionClient:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValidationPackError("OPENAI_API_KEY is required to send validation packs")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        timeout = _env_float("AI_REQUEST_TIMEOUT", _DEFAULT_TIMEOUT)
        return _ChatCompletionClient(base_url=base_url, api_key=api_key, timeout=timeout)

    @staticmethod
    def _normalize_account_id(account_id: Any) -> int:
        if account_id is None:
            raise ValueError("account_id is required")
        try:
            return int(str(account_id))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"account_id must be numeric: {account_id!r}") from exc

    def _fallback_response(self, message: str) -> Mapping[str, Any]:
        return {
            "decision": "no_case",
            "rationale": f"AI error: {message}",
            "citations": [],
        }

    def _normalize_decision(self, value: Any) -> str:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in _VALID_DECISIONS:
                return lowered
        return "no_case"

    @staticmethod
    def _normalize_justification(value: Any) -> str:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        return ""

    @staticmethod
    def _normalize_labels(value: Any) -> list[str]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return []
        labels: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    labels.append(text)
        return labels

    @staticmethod
    def _normalize_citations(value: Any) -> list[str]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return []
        citations: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                citations.append(item.strip())
        return citations

    @staticmethod
    def _normalize_confidence(value: Any) -> float | None:
        if value is None:
            return None
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None
        if confidence < 0 or confidence > 1:
            return None
        return round(confidence, 6)

    def _coerce_identifier(self, account_id: int, line_number: int, candidate: Any) -> str:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        return f"acc_{account_id:03d}__line_{line_number:03d}"

    def _coerce_field_name(
        self, pack_line: Mapping[str, Any], line_number: int
    ) -> str:
        field_key = pack_line.get("field_key")
        if isinstance(field_key, str) and field_key.strip():
            return field_key.strip()
        field = pack_line.get("field")
        if isinstance(field, str) and field.strip():
            return field.strip()
        return f"line_{line_number:03d}"

    def _is_allowed_field(self, field: str) -> bool:
        if field in _ALLOWED_FIELDS:
            return True

        canonical = field.strip().lower().replace(" ", "_")
        if canonical in _ALLOWED_FIELDS:
            return True

        log.debug("ALLOWING_UNKNOWN_FIELD field=%s canonical=%s", field, canonical)
        return True

    def _load_index(self) -> ValidationIndex:
        return self._index

    def _log(self, event: str, **payload: Any) -> None:
        log_path = self._log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry: MutableMapping[str, Any] = {
            "timestamp": _utc_now(),
            "sid": self.sid,
            "event": event,
        }
        entry.update(sanitize_validation_log_payload(payload))
        line = json.dumps(entry, ensure_ascii=False, sort_keys=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def send_validation_packs(
    manifest: Mapping[str, Any] | ValidationIndex | Path | str,
) -> list[dict[str, Any]]:
    """Send all validation packs referenced by ``manifest``."""

    sender = ValidationPackSender(manifest)
    return sender.send()


__all__ = ["send_validation_packs", "ValidationPackSender", "ValidationPackError"]
