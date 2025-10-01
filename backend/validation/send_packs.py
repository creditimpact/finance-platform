"""Send Validation AI packs to the model and persist the responses."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import requests

from backend.core.ai.paths import (
    validation_result_jsonl_filename_for_account,
    validation_result_summary_filename_for_account,
)

from .build_packs import load_manifest_from_source, resolve_manifest_paths

_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_TIMEOUT = 30.0
_THROTTLE_SECONDS = 0.05
_VALID_DECISIONS = {"strong", "no_case"}


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


class ValidationPackSender:
    """Send validation packs and store the adjudication results."""

    def __init__(
        self,
        manifest: Mapping[str, Any],
        *,
        http_client: _ChatCompletionClient | None = None,
    ) -> None:
        self.paths = resolve_manifest_paths(manifest)
        self.sid = self.paths.sid
        self.model = os.getenv("AI_MODEL", _DEFAULT_MODEL)
        self._client = http_client or self._build_client()
        self._throttle = _THROTTLE_SECONDS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def send(self) -> list[dict[str, Any]]:
        """Send every pack referenced by the manifest index."""

        index_payload = self._load_index()
        items = index_payload.get("items")
        if not isinstance(items, Sequence):
            return []

        results: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            account_id = item.get("account_id")
            pack_path = item.get("pack")
            if pack_path is None:
                continue
            try:
                normalized_account = self._normalize_account_id(account_id)
            except ValueError:
                normalized_account = None
            resolved_pack = Path(str(pack_path))
            if not resolved_pack.is_absolute():
                resolved_pack = (self.paths.packs_dir / resolved_pack).resolve()
            try:
                account_summary = self._process_account(
                    account_id,
                    normalized_account,
                    resolved_pack,
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
                    account_id=normalized_account,
                    error=str(exc),
                )
                account_summary = self._record_account_failure(
                    normalized_account, resolved_pack, str(exc)
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
    ) -> dict[str, Any]:
        account_int = normalized_account
        if account_int is None:
            raise ValidationPackError(f"Account id is not numeric: {account_id!r}")

        try:
            pack_lines = list(self._iter_pack_lines(pack_path))
        except ValidationPackError:
            self._log(
                "send_account_start",
                account_id=account_int,
                pack=str(pack_path),
                lines=0,
            )
            raise

        result_lines: list[dict[str, Any]] = []
        errors: list[str] = []

        self._log(
            "send_account_start",
            account_id=account_int,
            pack=str(pack_path),
            lines=len(pack_lines),
        )

        for idx, pack_line in enumerate(pack_lines, start=1):
            try:
                response = self._call_model(pack_line)
            except Exception as exc:
                errors.append(str(exc))
                response = self._fallback_response(str(exc))
            result_lines.append(self._build_result_line(account_int, idx, pack_line, response))
            time.sleep(self._throttle)

        status = "error" if errors else "done"
        error_message = "; ".join(errors) if errors else None
        jsonl_path, summary_path = self._write_results(
            account_int, result_lines, status=status, error=error_message
        )

        summary_payload: dict[str, Any] = {
            "sid": self.sid,
            "account_id": account_int,
            "pack_path": str(pack_path),
            "results_path": str(summary_path),
            "jsonl_path": str(jsonl_path),
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
            account_id=account_int,
            status=status,
            errors=len(errors),
        )
        return summary_payload

    def _record_account_failure(
        self, account_id: int, pack_path: Path, error: str
    ) -> dict[str, Any]:
        jsonl_path, summary_path = self._write_results(
            account_id, [], status="error", error=error
        )
        summary_payload = {
            "sid": self.sid,
            "account_id": account_id,
            "pack_path": str(pack_path),
            "results_path": str(summary_path),
            "jsonl_path": str(jsonl_path),
            "status": "error",
            "model": self.model,
            "request_lines": 0,
            "results": [],
            "completed_at": _utc_now(),
            "error": error,
        }
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
    ) -> tuple[Path, Path]:
        results_dir = self.paths.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = results_dir / validation_result_jsonl_filename_for_account(account_id)
        summary_path = (
            results_dir / validation_result_summary_filename_for_account(account_id)
        )

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
        return jsonl_path, summary_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _iter_pack_lines(self, pack_path: Path) -> Iterable[Mapping[str, Any]]:
        try:
            text = pack_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ValidationPackError(f"Pack file missing: {pack_path}") from exc
        except OSError as exc:
            raise ValidationPackError(f"Unable to read pack file: {pack_path}") from exc

        lines: list[Mapping[str, Any]] = []
        for idx, raw in enumerate(text.splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValidationPackError(
                    f"Invalid JSON in pack line {idx} of {pack_path}: {exc}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise ValidationPackError(
                    f"Pack line {idx} of {pack_path} is not an object"
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

    def _load_index(self) -> Mapping[str, Any]:
        index_path = self.paths.index_path
        try:
            text = index_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ValidationPackError(f"Validation index missing: {index_path}") from exc
        except OSError as exc:
            raise ValidationPackError(f"Unable to read validation index: {index_path}") from exc
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValidationPackError(f"Validation index is not valid JSON: {index_path}") from exc
        if not isinstance(document, Mapping):
            raise ValidationPackError("Validation index root must be an object")
        return document

    def _log(self, event: str, **payload: Any) -> None:
        log_path = self.paths.log_path
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
    manifest: Mapping[str, Any] | Path | str,
) -> list[dict[str, Any]]:
    """Send all validation packs referenced by ``manifest``."""

    manifest_data = load_manifest_from_source(manifest)
    sender = ValidationPackSender(manifest_data)
    return sender.send()


__all__ = ["send_validation_packs", "ValidationPackSender", "ValidationPackError"]
