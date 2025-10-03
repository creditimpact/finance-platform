"""Send Validation AI packs to the model and persist the responses."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import requests

from backend.core.ai.paths import (
    validation_result_jsonl_filename_for_account,
    validation_result_summary_filename_for_account,
)
from backend.validation.index_schema import (
    ValidationIndex,
    ValidationPackRecord,
)

_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_TIMEOUT = 30.0
_THROTTLE_SECONDS = 0.05
_VALID_DECISIONS = {"strong", "no_case"}
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


def _enforce_conditional_gate(
    field: str,
    decision: str,
    rationale: str,
    pack_line: Mapping[str, Any],
) -> tuple[str, str]:
    if decision != "strong":
        return decision, rationale
    if not _coerce_bool_flag(pack_line.get("conditional_gate")):
        return decision, rationale

    min_corroboration = max(1, _coerce_int_value(pack_line.get("min_corroboration"), 1))
    mismatch, corroboration, normalized_values = _conditional_mismatch_metrics(
        field, pack_line
    )

    if field == "creditor_remarks" and mismatch:
        if not _has_high_signal_creditor_remarks(normalized_values):
            mismatch = False

    if not mismatch or corroboration < min_corroboration:
        return "no_case", _append_gate_note(rationale, "insufficient_evidence")

    return decision, rationale


class ValidationPackError(RuntimeError):
    """Raised when the Validation AI sender encounters a fatal error."""


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


@dataclass(slots=True)
class _ChatCompletionClient:
    """Minimal HTTP client for the OpenAI Chat Completions API."""

    base_url: str
    api_key: str
    timeout: float | None

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/") or "https://api.openai.com/v1"

    def create(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        response_format: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": list(messages),
            "response_format": dict(response_format),
        }
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()


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
        self.model = os.getenv("AI_MODEL", _DEFAULT_MODEL)
        self._client = http_client or self._build_client()
        self._throttle = _THROTTLE_SECONDS
        self._results_root: Path | None = None
        self._log_path = view.log_path

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
    def send(self) -> list[dict[str, Any]]:
        """Send every pack referenced by the manifest index."""

        index = self._load_index()
        preflight = self._preflight(index)
        self._results_root = index.results_dir_path

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

        result_lines: list[dict[str, Any]] = []
        errors: list[str] = []

        self._log(
            "send_account_start",
            account_id=f"{account_int:03d}",
            pack=pack_display,
            pack_absolute=str(pack_path),
            lines=len(pack_lines),
        )

        for idx, pack_line in enumerate(pack_lines, start=1):
            try:
                response = self._call_model(pack_line)
            except Exception as exc:
                error_message = (
                    "AI request failed for acc "
                    f"{account_int:03d} pack={pack_display} -> "
                    f"{result_jsonl_display}, {result_summary_display}: {exc}"
                )
                errors.append(error_message)
                response = self._fallback_response(error_message)
            result_lines.append(self._build_result_line(account_int, idx, pack_line, response))
            time.sleep(self._throttle)

        status = "error" if errors else "done"
        error_message = "; ".join(errors) if errors else None
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

        self._log(
            "send_account_done",
            account_id=f"{account_int:03d}",
            status=status,
            errors=len(errors),
            results=len(result_lines),
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
        self._log(
            "send_account_done",
            account_id=f"{account_id:03d}",
            status="error",
            errors=1,
            results=0,
        )
        return summary_payload

    def _call_model(self, pack_line: Mapping[str, Any]) -> Mapping[str, Any]:
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
        response = self._client.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
        )

        choices = response.get("choices")
        if not isinstance(choices, Sequence) or not choices:
            raise ValidationPackError("Model response missing choices")
        message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise ValidationPackError("Model response missing content")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValidationPackError(f"Model response is not valid JSON: {exc}")
        if not isinstance(parsed, Mapping):
            raise ValidationPackError("Model response is not an object")
        return parsed

    # ------------------------------------------------------------------
    # Result construction & persistence
    # ------------------------------------------------------------------
    def _build_result_line(
        self,
        account_id: int,
        line_number: int,
        pack_line: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        line_id = self._coerce_identifier(account_id, line_number, pack_line.get("id"))
        field = self._coerce_field_name(pack_line, line_number)
        decision = self._normalize_decision(response.get("decision"))
        rationale = self._normalize_rationale(response.get("rationale"))
        citations = self._normalize_citations(response.get("citations"))
        confidence = self._normalize_confidence(response.get("confidence"))

        decision, rationale = _enforce_conditional_gate(
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
        return result

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
        return jsonl_path, summary_path

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
    def _normalize_rationale(value: Any) -> str:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        return ""

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
        field = pack_line.get("field")
        if isinstance(field, str) and field.strip():
            return field.strip()
        field_key = pack_line.get("field_key")
        if isinstance(field_key, str) and field_key.strip():
            return field_key.strip()
        return f"line_{line_number:03d}"

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
        entry.update(payload)
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
