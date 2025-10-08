"""Validation AI pack payload builder."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from backend.ai.manifest import ensure_validation_section
from backend.ai.validation_index import (
    ValidationIndexEntry,
    ValidationPackIndexWriter,
)
from backend.analytics.analytics_tracker import emit_counter

from backend.core.ai.paths import (
    validation_base_dir,
    validation_index_path,
    validation_pack_filename_for_account,
    validation_packs_dir,
    validation_result_jsonl_filename_for_account,
    validation_result_filename_for_account,
    validation_results_dir,
    validation_logs_path,
)
from backend.pipeline.runs import RunManifest, persist_manifest
from backend.validation.redaction import sanitize_validation_payload
from backend.core.ai.eligibility_policy import (
    canonicalize_history,
    canonicalize_scalar,
)
from backend.core.ai.report_compare import (
    classify_reporting_pattern,
    compute_reason_flags,
)
from backend.core.logic.validation_field_sets import (
    ALL_VALIDATION_FIELD_SET,
    ALL_VALIDATION_FIELDS,
    ALWAYS_INVESTIGATABLE_FIELDS,
    CONDITIONAL_FIELDS,
)

log = logging.getLogger(__name__)

_PACKS_ENABLED_ENV = "VALIDATION_PACKS_ENABLED"
_PACKS_PER_FIELD_ENV = "VALIDATION_PACKS_PER_FIELD"
_PACK_MAX_SIZE_ENV = "VALIDATION_PACK_MAX_SIZE_KB"
_AUTO_SEND_ENV_VARS: tuple[str, ...] = (
    "ENABLE_VALIDATION_SENDER",
    "AUTO_VALIDATION_SEND",
    "VALIDATION_SEND_ON_BUILD",
)
_AUTOSEND_RECHECK_MIN_DELAY = 1.0
_AUTOSEND_RECHECK_MAX_DELAY = 2.0

_LOCKS_DIRNAME = ".locks"
_MERGE_INFLIGHT_LOCK_FILENAME = "merge_inflight.lock"

_BUREAUS = ("transunion", "experian", "equifax")
_SYSTEM_PROMPT = (
    "Project: credit-analyzer\n"
    "Module: Validation / AI Adjudication\n\n"
    "The validation system detects discrepancies between credit bureaus’ reported values for each field (Equifax, Experian, TransUnion).\n"
    "It doesn’t understand language or legal reasoning — that’s why findings are sent to the AI adjudicator.\n\n"
    "Your role is to determine whether the bureau statements conflict in a way that creates usable evidence for a consumer dispute.\n"
    "You are not deciding which bureau is correct; you are deciding whether the discrepancy is strong enough that a bureau would be obligated or reasonably expected to investigate or correct it if disputed.\n"
    "Assume the consumer asserts the version that is most favorable to them.\n"
    "The detection platform already confirmed the numerical/text mismatch — focus only on the linguistic and legal significance.\n\n"
    "Each pack represents one field (e.g., account_type, payment_status, etc.) and includes the normalized/raw values reported by each bureau.\n"
    "Evaluate these values exactly as written and produce a JSON decision that aligns with the expected schema."
)

_ACCOUNT_TYPE_GENERIC_HINT = (
    "For account_type with C4 generic-vs-specific wording, default is not material unless category changes (e.g., revolving vs installment)."
)

_PROMPT_USER_TEMPLATE = (
    "Evaluate the following tri-bureau field finding and answer the core question: Is this discrepancy strong enough that a credit bureau would be obligated or reasonably expected to investigate or correct it if disputed?\n"
    "The detection system already confirmed the mismatch — concentrate on legal/linguistic significance, not detection mechanics.\n"
    "Assume the consumer claims the more favorable version is correct.\n\n"
    "Instructions:\n"
    "1) Choose one decision: strong_actionable | supportive_needs_companion | neutral_context_only | no_case.\n"
    "   Base your choice on the legal and material significance of the bureau statements, not on detection mechanics.\n"
    "2) Populate the 'checks' object:\n"
    "   - materiality: true/false\n"
    "   - supports_consumer: true/false (true when the mismatch favors the consumer’s asserted, more beneficial version)\n"
    "   - doc_requirements_met: true/false (use finding.documents; assume true when the list is non-empty)\n"
    "   - mismatch_code: copy finding.reason_code exactly (e.g., \"C4_TWO_MATCH_ONE_DIFF\")\n"
    "3) Provide a rationale (≤120 words) that explains whether the bureau narratives conflict in a legally meaningful way. Reference the mismatch code and rely only on the provided text.\n"
    "4) List citations for the exact bureau:value pairs you used (e.g., \"equifax: paid charge-off\").\n\n"
    "Constraints:\n"
    "- Use only the provided normalized/raw values; do not speculate beyond the pack.\n"
    "- Do not simply restate that a mismatch exists—focus on its legal impact.\n"
    "- Minor wording differences that keep the same meaning (e.g., \"real estate mortgage\" vs \"conventional real estate mortgage\" or \"auto loan\" vs \"vehicle loan\") usually stay neutral_context_only.\n"
    "- Clear categorical conflicts (e.g., \"secured loan\" vs \"unsecured loan\", \"charged-off\" vs \"paid as agreed\") should be treated as strong_actionable.\n"
    "- Differences like \"mortgage\" vs \"HELOC\" can be supportive_needs_companion when they suggest a different product but need corroboration.\n"
    "- Output strictly valid JSON matching the expected_output schema.\n\n"
    "Finding (verbatim JSON):\n"
    "<finding blob here>\n\n"
    "Return JSON only.\n"
    "If you cannot produce a valid object, return:\n"
    '{"decision":"no_case","rationale":"schema_mismatch","citations":["system:none"],"checks":{"materiality":false,"supports_consumer":false,"doc_requirements_met":false,"mismatch_code":"unknown"}}'
)

_EXPECTED_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["decision", "rationale", "citations", "checks"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": [
                "strong_actionable",
                "supportive_needs_companion",
                "neutral_context_only",
                "no_case",
            ],
        },
        "rationale": {"type": "string", "maxLength": 700},
        "citations": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "checks": {
            "type": "object",
            "required": [
                "materiality",
                "supports_consumer",
                "doc_requirements_met",
                "mismatch_code",
            ],
            "properties": {
                "materiality": {"type": "boolean"},
                "supports_consumer": {"type": "boolean"},
                "doc_requirements_met": {"type": "boolean"},
                "mismatch_code": {"type": "string"},
            },
        },
    },
}

_FIELD_CATEGORY_MAP: dict[str, str] = {
    # Open / Identification
    "date_opened": "open_ident",
    "closed_date": "open_ident",
    "account_type": "open_ident",
    "creditor_type": "open_ident",
    # Terms
    "high_balance": "terms",
    "credit_limit": "terms",
    "term_length": "terms",
    "payment_amount": "terms",
    "payment_frequency": "terms",
    # Activity
    "balance_owed": "activity",
    "last_payment": "activity",
    "past_due_amount": "activity",
    "date_of_last_activity": "activity",
    # Status / Reporting
    "account_status": "status",
    "payment_status": "status",
    "date_reported": "status",
    "account_rating": "status",
    # Histories
    "two_year_payment_history": "history",
}

_missing_fields = ALL_VALIDATION_FIELD_SET - set(_FIELD_CATEGORY_MAP)
if _missing_fields:
    missing = ", ".join(sorted(_missing_fields))
    raise RuntimeError(f"Missing validation categories for fields: {missing}")

_extra_fields = set(_FIELD_CATEGORY_MAP) - ALL_VALIDATION_FIELD_SET
if _extra_fields:
    extra = ", ".join(sorted(_extra_fields))
    raise RuntimeError(f"Unexpected validation fields in category map: {extra}")

_ALWAYS_INVESTIGATABLE_FIELDS: dict[str, str] = {
    field: _FIELD_CATEGORY_MAP[field] for field in ALWAYS_INVESTIGATABLE_FIELDS
}

_CONDITIONAL_FIELDS: dict[str, str] = {
    field: _FIELD_CATEGORY_MAP[field] for field in CONDITIONAL_FIELDS
}

_ALLOWED_FIELD_CATEGORIES: dict[str, str] = {
    field: _FIELD_CATEGORY_MAP[field] for field in ALL_VALIDATION_FIELDS
}

_ALLOWED_FIELDS: frozenset[str] = frozenset(ALL_VALIDATION_FIELDS)
_ALLOWED_CATEGORIES: frozenset[str] = frozenset(
    _ALLOWED_FIELD_CATEGORIES.values()
)

AI_FIELDS: frozenset[str] = frozenset(
    {
        "account_type",
        "creditor_type",
        "account_rating",
    }
)
FALLBACK_FIELDS: frozenset[str] = frozenset({"two_year_payment_history"})
EXCLUDED_FIELDS: frozenset[str] = frozenset(
    {"seven_year_history", "account_number_display"}
)

_PACK_ELIGIBLE_FIELDS: frozenset[str] = frozenset(AI_FIELDS | FALLBACK_FIELDS)

_TRUE_STRINGS: frozenset[str] = frozenset({"1", "true", "yes", "y", "on"})
_FALSE_STRINGS: frozenset[str] = frozenset({"0", "false", "no", "n", "off", ""})


def _normalize_flag(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
    return None


def _is_mismatch(requirement: Mapping[str, Any]) -> bool:
    return _normalize_flag(requirement.get("is_mismatch")) is True


def _history_2y_allowed() -> bool:
    """Return ``True`` if two-year history fallback packs are enabled."""

    return os.getenv("VALIDATION_ALLOW_HISTORY_2Y_AI", "1") == "1"


def _reasons_enabled() -> bool:
    raw = os.getenv("VALIDATION_REASON_ENABLED")
    if raw is None:
        return False

    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return False


@dataclass(frozen=True)
class PackLine:
    """Single validation pack line ready to be serialized.

    A pack line is a fully hydrated JSON-serialisable mapping that encodes
    everything the adjudication model needs in order to make a validation
    decision for one weak field.  The ``payload`` contains:

    * ``id`` – stable identifier of the ``account`` / ``field`` pair
      (``acc_<ACCID>__<FIELDKEY>``).
    * ``sid`` / ``account_id`` / ``account_key`` – run and account metadata so
      downstream tooling can trace lineage.
    * ``field`` / ``field_key`` / ``category`` / ``documents`` /
      ``min_days`` / ``strength`` – the raw requirement description copied from
      ``summary.json``.
    * ``bureaus`` – the raw and normalised bureau values for each supported
      agency.
    * ``context`` – auxiliary consistency data (consensus, disagreeing or
      missing bureaus, history snippets, requirement notes, etc.).
    * ``prompt`` – the actual request that will be sent to the model.  It
      contains the system prompt plus the user payload above, keeping the pack
      self-descriptive.
    * ``expected_output`` – schema describing the desired JSON response.  The
      model must emit ``decision`` (``strong`` or ``no_case``), ``rationale``,
      and ``citations``.  Models *may* also include a ``confidence`` field
      between ``0`` and ``1``; callers should treat it as optional.

    The builder keeps this schema mirrored with ``docs/ai_packs/validation`` so
    future contributors can safely extend the pack format.
    """

    payload: Mapping[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.payload, ensure_ascii=False, sort_keys=True)


@dataclass
class _PackSizeStats:
    count: int = 0
    total_bytes: int = 0
    max_bytes: int = 0

    def observe(self, size_bytes: int) -> None:
        self.count += 1
        self.total_bytes += max(size_bytes, 0)
        if size_bytes > self.max_bytes:
            self.max_bytes = size_bytes

    def average_bytes(self) -> float:
        if not self.count:
            return 0.0
        return self.total_bytes / self.count

    def average_kb(self) -> float:
        return self.average_bytes() / 1024

    def max_kb(self) -> float:
        return self.max_bytes / 1024

    def to_payload(self) -> Mapping[str, float | int]:
        return {
            "count": self.count,
            "avg_bytes": self.average_bytes(),
            "avg_kb": self.average_kb(),
            "max_bytes": self.max_bytes,
            "max_kb": self.max_kb(),
        }


class ValidationPackWriter:
    """Build consolidated validation packs for a run."""

    def __init__(
        self,
        sid: str,
        *,
        runs_root: Path | str | None = None,
        per_field: bool = False,
    ) -> None:
        self.sid = str(sid)
        self._runs_root = Path(runs_root) if runs_root is not None else Path("runs")
        self._accounts_root = self._runs_root / self.sid / "cases" / "accounts"
        self._packs_dir = validation_packs_dir(
            self.sid, runs_root=self._runs_root, create=True
        )
        self._results_dir = validation_results_dir(
            self.sid, runs_root=self._runs_root, create=True
        )
        self._log_path = validation_logs_path(
            self.sid, runs_root=self._runs_root, create=True
        )
        index_path = validation_index_path(
            self.sid, runs_root=self._runs_root, create=True
        )
        self._index_writer = ValidationPackIndexWriter(
            sid=self.sid,
            index_path=index_path,
            packs_dir=self._packs_dir,
            results_dir=self._results_dir,
        )
        self._per_field = per_field
        self._pack_max_size_kb = _pack_max_size_kb()
        self._pack_max_size_bytes = (
            int(self._pack_max_size_kb * 1024) if self._pack_max_size_kb is not None else None
        )
        self._field_counts: Counter[str] = Counter()
        self._size_stats = _PackSizeStats()
        self._last_pack_written: bool = False
        self._last_pack_had_findings: bool = False
        self._has_written_any_pack: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def write_all_packs(self) -> dict[int, list[PackLine]]:
        """Build packs for every account under ``cases/accounts``."""

        results: dict[int, list[PackLine]] = {}
        for account_id in self._discover_account_ids():
            try:
                results[account_id] = self.write_pack_for_account(account_id)
            except Exception:
                pack_path: Path | None = None
                account_label = account_id
                try:
                    normalized_id = self._normalize_account_id(account_id)
                    pack_path = (
                        self._packs_dir
                        / validation_pack_filename_for_account(normalized_id)
                    )
                    account_label = (
                        f"{normalized_id:03d}" if isinstance(normalized_id, int) else normalized_id
                    )
                except Exception:
                    pass
                log.exception(
                    "VALIDATION_PACK_WRITE_FAILED sid=%s account_id=%s pack=%s",
                    self.sid,
                    account_label,
                    pack_path,
                )
        return results

    def write_pack_for_account(self, account_id: int | str) -> list[PackLine]:
        """Build and persist the pack for ``account_id``."""

        self._last_pack_written = False
        normalized_id = self._normalize_account_id(account_id)
        summary = self._load_summary(normalized_id)
        pack_lines = self._build_pack_lines_from_summary(normalized_id, summary)
        if not pack_lines:
            account_label = (
                f"{normalized_id:03d}" if isinstance(normalized_id, int) else str(normalized_id)
            )
            if self._last_pack_had_findings:
                log.info(
                    "validation pack skipped: no eligible lines (sid=%s account=%s)",
                    self.sid,
                    account_label,
                )
            else:
                log.info(
                    "validation pack skipped: no findings (sid=%s account=%s)",
                    self.sid,
                    account_label,
                )
            return []

        serialized, size_bytes = self._serialize_pack_lines(pack_lines)
        pack_path = self._packs_dir / validation_pack_filename_for_account(normalized_id)

        if (
            pack_lines
            and self._pack_max_size_bytes is not None
            and size_bytes > self._pack_max_size_bytes
        ):
            self._handle_blocked_pack(normalized_id, pack_lines, summary, size_bytes)
            return []

        self._write_pack_file(pack_path, serialized)
        self._update_index(normalized_id, pack_path, pack_lines, summary=summary)
        self._append_log_entry(
            normalized_id,
            pack_lines,
            summary,
            pack_size_bytes=size_bytes,
        )
        self._last_pack_written = True
        self._has_written_any_pack = True
        return pack_lines

    def last_pack_was_written(self) -> bool:
        """Return ``True`` if the latest ``write_pack_for_account`` produced a pack."""

        return self._last_pack_written

    def has_written_any_pack(self) -> bool:
        """Return ``True`` if any pack has been written during this writer's lifecycle."""

        return self._has_written_any_pack

    def build_pack_lines(self, account_id: int) -> list[PackLine]:
        """Return the pack lines for ``account_id`` without writing them."""

        summary = self._load_summary(account_id)
        return self._build_pack_lines_from_summary(account_id, summary)

    def _build_pack_lines_from_summary(
        self, account_id: int, summary: Mapping[str, Any] | None
    ) -> list[PackLine]:
        self._last_pack_had_findings = False
        if not summary:
            return []

        validation_block = self._extract_validation_block(summary)
        if not validation_block:
            return []

        findings = validation_block["findings"]
        if not findings:
            return []

        self._last_pack_had_findings = True

        send_to_ai_map = validation_block.get("send_to_ai", {})
        consistency_map = validation_block.get("field_consistency", {})

        bureaus_cache: dict[str, Mapping[str, Any]] | None = None

        def _load_bureaus_if_needed() -> Mapping[str, Mapping[str, Any]]:
            nonlocal bureaus_cache
            if bureaus_cache is None:
                bureaus_cache = self._load_bureaus(account_id)
            return bureaus_cache

        pack_lines: list[PackLine] = []
        seen_pack_keys: set[str] = set()
        for requirement in findings:
            if not isinstance(requirement, Mapping):
                continue

            canonical_field = self._canonical_field_name(requirement.get("field"))
            if canonical_field is None:
                continue

            if canonical_field not in _PACK_ELIGIBLE_FIELDS:
                continue

            if requirement.get("is_missing") is True:
                continue

            if isinstance(consistency_map, Mapping):
                consistency_entry = consistency_map.get(canonical_field)
            else:
                consistency_entry = None

            bureaus_payload: Mapping[str, Mapping[str, Any]] = _load_bureaus_if_needed()

            requirement_payload = dict(requirement)
            existing_bureau_values = None
            if "bureau_values" in requirement_payload:
                existing_bureau_values = requirement_payload.get("bureau_values")
            if bureaus_payload:
                requirement_payload["bureau_values"] = self._build_bureau_values(
                    canonical_field,
                    bureaus_payload,
                    consistency_entry,
                    existing_requirement_values=existing_bureau_values,
                )

            context_payload = self._build_context(consistency_entry)
            if context_payload:
                requirement_payload["context"] = context_payload

            pack_key = self._build_pack_key(
                account_id,
                canonical_field,
                requirement_payload,
            )
            if pack_key in seen_pack_keys:
                continue
            seen_pack_keys.add(pack_key)

            send_flag = self._resolve_send_flag(
                requirement,
                canonical_field,
                send_to_ai_map=send_to_ai_map,
            )

            if send_flag is not True:
                continue

            if not self._should_send_to_ai(
                requirement,
                canonical_field,
                send_flag=send_flag,
                send_to_ai_map=send_to_ai_map,
            ):
                continue

            line = build_line(
                sid=self.sid,
                account_id=account_id,
                field=canonical_field,
                finding=requirement_payload,
                fallback_bureaus_loader=_load_bureaus_if_needed,
            )
            if line is not None:
                payload = dict(line)
                payload["send_to_ai"] = True
                pack_lines.append(PackLine(payload))

        return pack_lines

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------
    def _write_pack_file(self, path: Path, serialized: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding="utf-8")

    @staticmethod
    def _serialize_pack_lines(
        lines: Sequence[PackLine],
    ) -> tuple[str, int]:
        if not lines:
            return "", 0

        serialized_lines: list[str] = []
        for line in lines:
            if isinstance(line, PackLine):
                serialized_lines.append(line.to_json())
            elif isinstance(line, Mapping):  # pragma: no cover - defensive
                serialized_lines.append(
                    json.dumps(line, ensure_ascii=False, sort_keys=True)
                )
            else:  # pragma: no cover - defensive
                serialized_lines.append(json.dumps(line, ensure_ascii=False))

        serialized = "\n".join(serialized_lines) + "\n"
        size_bytes = len(serialized.encode("utf-8"))
        return serialized, size_bytes

    def _handle_blocked_pack(
        self,
        account_id: int,
        lines: Sequence[PackLine],
        summary: Mapping[str, Any] | None,
        size_bytes: int,
    ) -> None:
        account_label = f"{account_id:03d}" if isinstance(account_id, int) else str(account_id)
        log.warning(
            "VALIDATION_PACK_SIZE_BLOCKED sid=%s account_id=%s size_bytes=%d max_kb=%s",
            self.sid,
            account_label,
            size_bytes,
            self._pack_max_size_kb,
        )

        total_fields = self._count_total_requirements(summary)
        entry = {
            "timestamp": _utc_now(),
            "account_index": int(account_id),
            "weak_count": len(lines),
            "fields_built": 0,
            "total_fields": total_fields,
            "conditional_fields_built": 0,
            "statuses": ["pack_blocked_max_size"],
            "mode": "per_field" if self._per_field else "per_account",
            "pack_size_bytes": size_bytes,
            "pack_size_kb": _bytes_to_kb(size_bytes),
            "pack_size_limit_kb": self._pack_max_size_kb,
            "fields_emitted": [
                field
                for field in (
                    self._extract_line_field(line)
                    for line in lines
                )
                if field
            ],
            "cumulative_field_counts": dict(self._field_counts),
            "cumulative_size": self._size_stats.to_payload(),
        }
        _append_validation_log_entry(self._log_path, entry)

    def _update_index(
        self,
        account_id: int,
        pack_path: Path,
        lines: Sequence[PackLine],
        *,
        summary: Mapping[str, Any] | None = None,
    ) -> None:
        if not lines:
            return

        try:
            exists = pack_path.exists()
        except OSError:
            exists = False

        if not exists or not pack_path.is_file():
            log.warning(
                "VALIDATION_INDEX_SKIP_MISSING_PACK sid=%s account_id=%03d path=%s",
                self.sid,
                account_id,
                pack_path,
            )
            return

        try:
            size = pack_path.stat().st_size
        except OSError:
            size = 0

        if size <= 0:
            log.warning(
                "VALIDATION_INDEX_SKIP_EMPTY_PACK sid=%s account_id=%03d path=%s",
                self.sid,
                account_id,
                pack_path,
            )
            return
        weak_fields: list[str] = []
        for line in lines:
            if not isinstance(line, PackLine):
                continue
            payload = line.payload
            if not isinstance(payload, Mapping):
                continue
            field_key = str(payload.get("field_key") or "").strip()
            if not field_key:
                raw_field = payload.get("field")
                field_key = str(raw_field).strip() if raw_field is not None else ""
            if not field_key:
                continue
            weak_fields.append(field_key)

        if summary is None:
            summary = self._load_summary(account_id)
        source_hash = self._build_source_hash(summary, lines)

        entry = ValidationIndexEntry(
            account_id=account_id,
            pack_path=pack_path.resolve(),
            result_jsonl_path=None,
            result_json_path=None,
            weak_fields=tuple(weak_fields),
            line_count=len(lines),
            status="built",
            source_hash=source_hash,
        )
        self._index_writer.upsert(entry)

    def _append_log_entry(
        self,
        account_id: int,
        lines: Sequence[PackLine],
        summary: Mapping[str, Any] | None,
        *,
        pack_size_bytes: int,
    ) -> None:
        statuses = self._derive_statuses(summary, lines)
        total_fields = self._count_total_requirements(summary)
        conditional_fields_built = sum(
            1
            for line in lines
            if (field := self._extract_line_field(line)) and field in _CONDITIONAL_FIELDS
        )
        entry = {
            "timestamp": _utc_now(),
            "account_index": int(account_id),
            "weak_count": len(lines),
            "fields_built": len(lines),
            "total_fields": total_fields,
            "conditional_fields_built": conditional_fields_built,
            "statuses": statuses,
            "mode": "per_field" if self._per_field else "per_account",
            "pack_size_bytes": pack_size_bytes,
            "pack_size_kb": _bytes_to_kb(pack_size_bytes),
            "pack_size_limit_kb": self._pack_max_size_kb,
        }

        fields_emitted = [
            field
            for field in (
                self._extract_line_field(line)
                for line in lines
            )
            if field
        ]
        entry["fields_emitted"] = fields_emitted

        if fields_emitted:
            for field in fields_emitted:
                self._field_counts[field] += 1
            self._size_stats.observe(pack_size_bytes)

        entry["cumulative_field_counts"] = dict(self._field_counts)
        entry["cumulative_size"] = self._size_stats.to_payload()
        _append_validation_log_entry(self._log_path, entry)

    def _count_total_requirements(
        self, summary: Mapping[str, Any] | None
    ) -> int:
        if not summary:
            return 0
        validation_block = self._extract_validation_block(summary)
        if not validation_block:
            return 0
        findings = validation_block.get("findings")
        if not isinstance(findings, Sequence):
            return 0
        return sum(1 for requirement in findings if isinstance(requirement, Mapping))

    @staticmethod
    def _extract_line_field(line: PackLine) -> str | None:
        payload: Mapping[str, Any] | None
        if isinstance(line, PackLine):
            payload = line.payload
        else:
            payload = line  # type: ignore[assignment]
        if not isinstance(payload, Mapping):
            return None
        field_key = payload.get("field_key")
        if isinstance(field_key, str) and field_key.strip():
            return field_key.strip()
        field = payload.get("field")
        if isinstance(field, str) and field.strip():
            return field.strip()
        return None

    def _derive_statuses(
        self, summary: Mapping[str, Any] | None, lines: Sequence[PackLine]
    ) -> list[str]:
        if lines:
            return ["pack_written"]

        statuses: list[str] = []
        if summary is None:
            statuses.append("summary_missing")
            statuses.append("no_weak_items")
            return statuses

        validation_block = self._extract_validation_block(summary)
        if not validation_block:
            statuses.append("no_validation_requirements")
            statuses.append("no_weak_items")
            return statuses

        requirements = validation_block.get("findings") or []
        send_to_ai_map = validation_block.get("send_to_ai", {})
        has_ai_needed = False
        for requirement in requirements:
            if not isinstance(requirement, Mapping):
                continue
            strength = self._normalize_strength(requirement.get("strength"))
            if strength == "strong":
                continue
            canonical_field = self._canonical_field_name(requirement.get("field"))
            if canonical_field is None:
                continue
            if canonical_field not in _PACK_ELIGIBLE_FIELDS:
                continue
            if not self._should_send_to_ai(
                requirement,
                canonical_field,
                send_to_ai_map=send_to_ai_map,
            ):
                continue
            has_ai_needed = True
            break

        statuses.append("no_weak_items")
        if not has_ai_needed:
            statuses.insert(0, "no_ai_needed")
        return statuses

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _build_reason_metadata(
        self,
        account_id: int,
        field_name: str,
        bureau_values: Mapping[str, Mapping[str, Any]],
    ) -> tuple[Mapping[str, Any] | None, bool]:
        raw_values: dict[str, Any] = {}
        for bureau in _BUREAUS:
            bureau_data = bureau_values.get(bureau, {})
            raw_values[bureau] = bureau_data.get("raw")

        try:
            pattern = classify_reporting_pattern(raw_values)
        except Exception:  # pragma: no cover - defensive
            log.exception(
                "VALIDATION_REASON_CLASSIFY_FAILED field=%s", field_name
            )
            pattern = "unknown"

        if field_name in {"two_year_payment_history", "seven_year_history"}:
            canonicalizer = canonicalize_history
        else:
            canonicalizer = canonicalize_scalar

        canonical_values: dict[str, Any] = {}
        for bureau in _BUREAUS:
            try:
                canonical_values[bureau] = canonicalizer(raw_values.get(bureau))
            except Exception:  # pragma: no cover - defensive
                log.exception(
                    "VALIDATION_REASON_CANONICALIZE_FAILED field=%s bureau=%s",
                    field_name,
                    bureau,
                )
                canonical_values[bureau] = None

        flags = compute_reason_flags(field_name, pattern, match_matrix={})

        missing_bureaus = [
            bureau for bureau in _BUREAUS if canonical_values.get(bureau) is None
        ]
        present_bureaus = [
            bureau for bureau in _BUREAUS if canonical_values.get(bureau) is not None
        ]

        reason_payload = {
            "schema": 1,
            "pattern": pattern,
            "missing": flags.get("missing", False),
            "mismatch": flags.get("mismatch", False),
            "both": flags.get("both", False),
            "eligible": flags.get("eligible", False),
            "coverage": {
                "missing_bureaus": missing_bureaus,
                "present_bureaus": present_bureaus,
            },
            "values": canonical_values,
        }

        ai_needed = field_name in _CONDITIONAL_FIELDS and bool(flags.get("eligible"))

        if _reasons_enabled():
            self._record_reason_observability(
                account_id,
                field_name,
                pattern,
                flags,
                ai_needed,
            )
        return reason_payload, ai_needed

    def _record_reason_observability(
        self,
        account_id: int,
        field_name: str,
        pattern: str,
        flags: Mapping[str, Any],
        ai_needed: bool,
    ) -> None:
        """Log and emit metrics describing the escalation rationale."""

        missing = bool(flags.get("missing", False))
        mismatch = bool(flags.get("mismatch", False))
        eligible = bool(flags.get("eligible", False))

        metric_pattern = pattern if isinstance(pattern, str) and pattern else "unknown"

        log.info(
            "VALIDATION_ESCALATION_REASON sid=%s account_id=%s field=%s pattern=%s "
            "missing=%s mismatch=%s eligible=%s",
            self.sid,
            account_id,
            field_name,
            metric_pattern,
            missing,
            mismatch,
            eligible,
        )

        emit_counter(f"validation.pattern.{metric_pattern}")
        emit_counter(f"validation.eligible.{str(eligible).lower()}")
        emit_counter(f"validation.ai_needed.{str(ai_needed).lower()}")

    def _build_context(
        self, consistency: Mapping[str, Any] | None
    ) -> Mapping[str, Any]:
        if not isinstance(consistency, Mapping):
            return {}

        context: dict[str, Any] = {}
        consensus = self._coerce_optional_str(consistency.get("consensus"))
        if consensus:
            context["consensus"] = consensus

        disagreeing = self._normalize_string_list(
            consistency.get("disagreeing_bureaus")
        )
        if disagreeing:
            context["disagreeing_bureaus"] = disagreeing

        missing = self._normalize_string_list(consistency.get("missing_bureaus"))
        if missing:
            context["missing_bureaus"] = missing

        history = consistency.get("history")
        if isinstance(history, Mapping):
            context["history"] = self._normalize_history(history)

        return context

    def _build_bureau_values(
        self,
        field: str,
        bureaus_data: Mapping[str, Mapping[str, Any]],
        consistency: Mapping[str, Any] | None,
        *,
        existing_requirement_values: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Mapping[str, Any]]:
        raw_map = {}
        normalized_map = {}
        if isinstance(consistency, Mapping):
            raw_values = consistency.get("raw")
            if isinstance(raw_values, Mapping):
                raw_map = raw_values
            normalized_values = consistency.get("normalized")
            if isinstance(normalized_values, Mapping):
                normalized_map = normalized_values

        values: dict[str, dict[str, Any]] = {}
        existing_map = (
            existing_requirement_values
            if isinstance(existing_requirement_values, Mapping)
            else {}
        )
        for bureau in _BUREAUS:
            bureau_data = bureaus_data.get(bureau, {})
            raw_value = self._extract_value(raw_map.get(bureau))
            if raw_value is None:
                raw_value = self._extract_value(bureau_data.get(field))

            normalized_value = self._extract_value(normalized_map.get(bureau))

            existing_entry = existing_map.get(bureau)
            if isinstance(existing_entry, Mapping):
                if raw_value is None and "raw" in existing_entry:
                    raw_value = existing_entry.get("raw")
                if normalized_value is None and "normalized" in existing_entry:
                    normalized_value = existing_entry.get("normalized")

            values[bureau] = {
                "raw": raw_value,
                "normalized": normalized_value,
            }

        return values

    def _build_pack_key(
        self,
        account_id: int,
        canonical_field: str,
        requirement: Mapping[str, Any],
    ) -> str:
        payload = {
            "account_id": int(account_id),
            "field": canonical_field,
            "requirement": _json_clone(requirement),
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------
    def _discover_account_ids(self) -> Iterable[int]:
        if not self._accounts_root.is_dir():
            return []

        ids: list[int] = []
        for child in sorted(self._accounts_root.iterdir()):
            if not child.is_dir():
                continue
            try:
                account_id = int(child.name)
            except (TypeError, ValueError):
                log.debug("Skipping non-numeric account directory: %s", child)
                continue
            ids.append(account_id)
        return ids

    def _load_summary(self, account_id: int) -> Mapping[str, Any] | None:
        path = self._accounts_root / str(account_id) / "summary.json"
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError:
            log.warning("VALIDATION_SUMMARY_READ_FAILED path=%s", path, exc_info=True)
            return None

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            log.warning("VALIDATION_SUMMARY_INVALID_JSON path=%s", path, exc_info=True)
            return None

        return payload if isinstance(payload, Mapping) else None

    def _load_bureaus(self, account_id: int) -> Mapping[str, Mapping[str, Any]]:
        path = self._accounts_root / str(account_id) / "bureaus.json"
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError:
            log.warning("VALIDATION_BUREAUS_READ_FAILED path=%s", path, exc_info=True)
            return {}

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            log.warning("VALIDATION_BUREAUS_INVALID_JSON path=%s", path, exc_info=True)
            return {}

        if not isinstance(payload, Mapping):
            return {}

        normalized: dict[str, dict[str, Any]] = {}
        for bureau, values in payload.items():
            if not isinstance(values, Mapping):
                continue
            normalized[bureau.strip().lower()] = {
                str(key): val for key, val in values.items()
            }
        return normalized

    # ------------------------------------------------------------------
    # Extractors
    # ------------------------------------------------------------------
    def _extract_validation_block(
        self, summary: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        block = summary.get("validation_requirements")
        if not isinstance(block, Mapping):
            return None

        raw_findings = block.get("findings")
        if isinstance(raw_findings, Sequence):
            findings_list = [
                entry for entry in raw_findings if isinstance(entry, Mapping)
            ]
        else:
            return None

        consistency = block.get("field_consistency")
        consistency_map = (
            consistency if isinstance(consistency, Mapping) else {}
        )

        send_to_ai_entries: Sequence[Any] | None
        if isinstance(raw_findings, Sequence):
            send_to_ai_entries = raw_findings
        else:
            send_to_ai_entries = None
        send_to_ai_map: dict[str, bool] = {}
        if isinstance(send_to_ai_entries, Sequence):
            for entry in send_to_ai_entries:
                if not isinstance(entry, Mapping):
                    continue
                field_name = entry.get("field")
                canonical = self._canonical_field_name(field_name)
                if canonical is None:
                    continue
                send_flag = bool(entry.get("send_to_ai"))
                send_to_ai_map[canonical] = send_flag

                if isinstance(field_name, str):
                    alias = field_name.strip().lower()
                elif field_name is not None:
                    alias = str(field_name).strip().lower()
                else:
                    alias = ""

                if alias and alias not in send_to_ai_map:
                    send_to_ai_map[alias] = send_flag

        return {
            "findings": findings_list,
            "field_consistency": consistency_map,
            "send_to_ai": send_to_ai_map,
        }

    def _resolve_send_flag(
        self,
        requirement: Mapping[str, Any],
        canonical_field: str,
        *,
        send_to_ai_map: Mapping[str, Any] | None = None,
    ) -> bool | None:
        lookup_keys: list[str] = [canonical_field]
        raw_field = requirement.get("field")
        if isinstance(raw_field, str):
            alias = raw_field.strip().lower()
        elif raw_field is not None:
            alias = str(raw_field).strip().lower()
        else:
            alias = ""

        if alias and alias not in lookup_keys:
            lookup_keys.append(alias)

        send_flag = _normalize_flag(requirement.get("send_to_ai"))

        if send_flag is None and isinstance(send_to_ai_map, Mapping):
            for key in lookup_keys:
                if key in send_to_ai_map:
                    send_flag = _normalize_flag(send_to_ai_map[key])
                    if send_flag is not None:
                        break

        return send_flag

    def _should_send_to_ai(
        self,
        requirement: Mapping[str, Any],
        canonical_field: str,
        *,
        send_flag: bool | None = None,
        send_to_ai_map: Mapping[str, Any] | None = None,
    ) -> bool:
        """Determine whether ``requirement`` should be routed to AI."""

        if canonical_field in EXCLUDED_FIELDS:
            return False

        if canonical_field not in _PACK_ELIGIBLE_FIELDS:
            return False

        if _normalize_flag(requirement.get("is_missing")) is True:
            return False

        if not _is_mismatch(requirement):
            return False

        if send_flag is None:
            send_flag = self._resolve_send_flag(
                requirement,
                canonical_field,
                send_to_ai_map=send_to_ai_map,
            )

        if send_flag is None:
            return False

        if send_flag is not True:
            return False

        if canonical_field in FALLBACK_FIELDS:
            return _history_2y_allowed()

        if canonical_field not in AI_FIELDS:
            return False

        return True

    @staticmethod
    def _normalize_account_id(account_id: int | str) -> int:
        if isinstance(account_id, int):
            return account_id
        return int(str(account_id).strip())

    @staticmethod
    def _canonical_field_name(field: Any) -> str | None:
        if field is None:
            return None

        if isinstance(field, str):
            text = field.strip()
        else:
            text = str(field).strip()

        if not text:
            return None

        normalized = text.lower()
        if normalized not in _ALLOWED_FIELDS:
            return None

        return normalized

    def _build_source_hash(
        self,
        summary: Mapping[str, Any] | None,
        lines: Sequence[PackLine],
    ) -> str:
        findings: list[Any] = []
        field_consistency: dict[str, Any] = {}
        canonical_fields: dict[str, set[str]] = {}

        if isinstance(summary, Mapping):
            validation_block = self._extract_validation_block(summary) or {}
            raw_findings: Sequence[Any] | None = validation_block.get("findings")
            if not isinstance(raw_findings, Sequence):
                raw_findings = None
            send_to_ai_map = validation_block.get("send_to_ai", {})
            if isinstance(raw_findings, Sequence):
                for entry in raw_findings:
                    if not isinstance(entry, Mapping):
                        continue
                    canonical_field = self._canonical_field_name(entry.get("field"))
                    if canonical_field is None:
                        continue
                    if canonical_field not in _PACK_ELIGIBLE_FIELDS:
                        continue
                    if not self._should_send_to_ai(
                        entry,
                        canonical_field,
                        send_to_ai_map=send_to_ai_map,
                    ):
                        continue

                    cloned = _json_clone(entry)
                    if isinstance(cloned, Mapping):
                        try:
                            cloned["field"] = canonical_field
                        except Exception:  # pragma: no cover - defensive
                            pass
                        expected_category = _ALLOWED_FIELD_CATEGORIES[canonical_field]
                        try:
                            cloned["category"] = expected_category
                        except Exception:  # pragma: no cover - defensive
                            pass
                    findings.append(cloned)

                    raw_field = entry.get("field")
                    aliases: set[str] = canonical_fields.setdefault(
                        canonical_field, set()
                    )
                    if isinstance(raw_field, str):
                        candidate = raw_field.strip()
                    elif raw_field is not None:
                        candidate = str(raw_field).strip()
                    else:
                        candidate = ""
                    if candidate and candidate != canonical_field:
                        aliases.add(candidate)
            raw_consistency = validation_block.get("field_consistency")
            if isinstance(raw_consistency, Mapping):
                for field in sorted(canonical_fields):
                    value = raw_consistency.get(field)
                    if value is None:
                        for alias in canonical_fields[field]:
                            value = raw_consistency.get(alias)
                            if value is not None:
                                break
                    field_consistency[field] = (
                        _json_clone(value) if value is not None else None
                    )

        payload = {
            "findings": findings,
            "field_consistency": field_consistency,
            "pack_lines": [line.to_json() for line in lines],
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_strength(strength: Any) -> str:
        if isinstance(strength, str):
            normalized = strength.strip().lower()
            if normalized in {"weak", "soft"}:
                return "weak"
            if normalized:
                return normalized
        return "weak"

    @staticmethod
    def _coerce_optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_optional_str(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return str(value)

    @staticmethod
    def _coerce_optional_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if value is None:
            return None
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return bool(value)
        return None

    @staticmethod
    def _normalize_string_list(value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
            return []
        normalized: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized

    @staticmethod
    def _field_key(field: str) -> str:
        key = re.sub(r"[^a-z0-9]+", "_", field.strip().lower())
        return key.strip("_") or "field"

    @staticmethod
    def _extract_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            for candidate in ("raw", "normalized", "value", "text"):
                if candidate in value:
                    return value[candidate]
            return dict(value)
        return value

    @staticmethod
    def _normalize_history(history: Mapping[str, Any]) -> Mapping[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in history.items():
            try:
                normalized[str(key)] = value
            except Exception:  # pragma: no cover - defensive
                continue
        return normalized


_ACCOUNT_TYPE_STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "without",
}


def _normalize_account_type_text(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _tokenize_account_type_value(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value))


def _is_generic_vs_specific_account_type_mismatch(
    finding: Mapping[str, Any]
) -> bool:
    if not isinstance(finding, Mapping):
        return False

    if finding.get("reason_code") != "C4_TWO_MATCH_ONE_DIFF":
        return False

    bureau_values = finding.get("bureau_values")
    if not isinstance(bureau_values, Mapping):
        return False

    normalized_strings: list[str] = []
    for entry in bureau_values.values():
        value: Any | None = None
        if isinstance(entry, Mapping):
            raw_value = entry.get("normalized")
            if not isinstance(raw_value, str) or not raw_value.strip():
                raw_value = entry.get("raw")
            if isinstance(raw_value, str) and raw_value.strip():
                value = raw_value
        elif isinstance(entry, str):  # pragma: no cover - defensive
            value = entry

        if isinstance(value, str):
            normalized_value = _normalize_account_type_text(value)
            if normalized_value:
                normalized_strings.append(normalized_value)

    if len(normalized_strings) < 2:
        return False

    counts = Counter(normalized_strings)
    if not counts:
        return False

    majority_value, majority_count = counts.most_common(1)[0]
    if majority_count < 2:
        return False

    minority_candidates = [value for value in counts if value != majority_value]
    if len(minority_candidates) != 1:
        return False

    minority_value = minority_candidates[0]
    if counts[minority_value] != 1:
        return False

    majority_tokens = _tokenize_account_type_value(majority_value)
    minority_tokens = _tokenize_account_type_value(minority_value)
    if not majority_tokens or not minority_tokens:
        return False

    common_tokens = majority_tokens & minority_tokens
    meaningful_common = {token for token in common_tokens if token not in _ACCOUNT_TYPE_STOPWORDS}
    if not meaningful_common:
        return False

    if majority_tokens.issubset(minority_tokens):
        extra_tokens = minority_tokens - majority_tokens
    elif minority_tokens.issubset(majority_tokens):
        extra_tokens = majority_tokens - minority_tokens
    else:
        return False

    meaningful_extra = {token for token in extra_tokens if token not in _ACCOUNT_TYPE_STOPWORDS}
    if not meaningful_extra:
        return False

    if len(meaningful_extra) > 3:
        return False

    return True


def build_line(
    *,
    sid: str,
    account_id: int,
    field: str,
    finding: Mapping[str, Any],
    fallback_bureaus_loader: Callable[[], Mapping[str, Mapping[str, Any]]] | None = None,
) -> Mapping[str, Any] | None:
    if not isinstance(finding, Mapping):
        return None

    if not isinstance(field, str):
        field_name = str(field)
    else:
        field_name = field

    field_name = field_name.strip()
    if not field_name:
        return None

    account_key = f"{account_id:03d}" if isinstance(account_id, int) else str(account_id)
    field_key = ValidationPackWriter._field_key(field_name)

    finding_payload = _json_clone(finding)
    finding_json = json.dumps(finding_payload, ensure_ascii=False, sort_keys=True)
    prompt_user = _PROMPT_USER_TEMPLATE.replace("<finding blob here>", finding_json)

    payload: dict[str, Any] = {
        "id": f"acc_{account_key}__{field_key}",
        "sid": sid,
        "account_id": account_id,
        "field": field_name,
        "finding": finding_payload,
        "finding_json": finding_json,
        "prompt": {
            "system": _SYSTEM_PROMPT,
            "user": prompt_user,
        },
        "expected_output": _json_clone(_EXPECTED_OUTPUT_SCHEMA),
    }

    if field_name == "account_type" and _is_generic_vs_specific_account_type_mismatch(
        finding_payload
    ):
        payload["prompt"]["system"] = (
            f"{payload['prompt']['system']}\n{_ACCOUNT_TYPE_GENERIC_HINT}"
        )

    return sanitize_validation_payload(payload)


def _json_clone(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_clone(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_clone(entry) for entry in value]
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _bytes_to_kb(size_bytes: int) -> float:
    return size_bytes / 1024


def _append_validation_log_entry(path: Path, entry: Mapping[str, Any]) -> None:
    try:
        serialized = json.dumps(entry, sort_keys=True, ensure_ascii=False)
    except TypeError:
        log.exception("VALIDATION_LOG_SERIALIZE_FAILED path=%s", path)
        return

    try:
        existing = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = ""
    except OSError:
        log.warning("VALIDATION_LOG_READ_FAILED path=%s", path, exc_info=True)
        existing = ""

    if existing and not existing.endswith("\n"):
        existing += "\n"

    new_contents = (existing + serialized + "\n") if existing else (serialized + "\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")

    try:
        temp_path.write_text(new_contents, encoding="utf-8")
        temp_path.replace(path)
    except OSError:
        log.warning("VALIDATION_LOG_WRITE_FAILED path=%s", path, exc_info=True)
        with suppress(FileNotFoundError, OSError):
            temp_path.unlink(missing_ok=True)


_WRITER_CACHE: dict[tuple[str, Path, bool], "ValidationPackWriter"] = {}
_WRITER_CACHE_LOCK = threading.Lock()


def _resolve_runs_root_from_artifacts(
    sid: str, *paths: Path | str | None
) -> Path:
    for raw in paths:
        if raw is None:
            continue
        try:
            candidate = Path(raw).resolve()
        except Exception:
            continue
        for parent in candidate.parents:
            if parent.name == sid:
                return parent.parent.resolve()
    return Path("runs").resolve()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "on", "y"}:
            return True
        if lowered in {"0", "false", "no", "off", "n"}:
            return False
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return bool(raw)
    return default


def _packs_enabled() -> bool:
    return _env_flag(_PACKS_ENABLED_ENV, True)


def _packs_per_field_enabled() -> bool:
    return _env_flag(_PACKS_PER_FIELD_ENV, False)


def _auto_send_enabled() -> bool:
    """Return ``True`` when every auto-send toggle is explicitly enabled."""

    return all(_env_flag(name, False) for name in _AUTO_SEND_ENV_VARS)


def _pack_max_size_kb() -> float | None:
    raw = os.getenv(_PACK_MAX_SIZE_ENV)
    if raw is None:
        return None

    if isinstance(raw, str):
        raw_value = raw.strip()
    else:
        raw_value = str(raw)

    if not raw_value:
        return None

    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        log.warning(
            "VALIDATION_PACK_MAX_SIZE_INVALID env_value=%r", raw,
        )
        return None

    if value <= 0:
        return None

    return value


def _merge_inflight_lock_path(runs_root: Path, sid: str) -> Path:
    return runs_root / sid / _LOCKS_DIRNAME / _MERGE_INFLIGHT_LOCK_FILENAME


def _merge_stage_sent(runs_root: Path, sid: str) -> bool:
    manifest_path = runs_root / sid / "manifest.json"
    manifest = RunManifest(manifest_path)
    try:
        manifest.load()
    except FileNotFoundError:
        return False
    except Exception:  # pragma: no cover - defensive logging
        log.debug(
            "VALIDATION_MERGE_STATUS_LOAD_FAILED sid=%s path=%s",
            sid,
            manifest_path,
            exc_info=True,
        )
        return False
    stage_status = manifest.get_ai_stage_status("merge")
    return bool(stage_status.get("sent"))


def _wait_for_merge_completion(
    sid: str, runs_root: Path, *, poll_interval: float = 0.5
) -> None:
    """Block until the merge inflight lock is cleared for ``sid``."""

    lock_path = _merge_inflight_lock_path(runs_root, sid)
    wait_reasons: set[str] = set()

    while True:
        lock_exists = lock_path.exists()
        merge_sent = _merge_stage_sent(runs_root, sid)
        if not lock_exists and merge_sent:
            break
        reason = "lock" if lock_exists else "status"
        if reason not in wait_reasons:
            log.info("VALIDATION_WAITING_FOR_MERGE sid=%s reason=%s", sid, reason)
            wait_reasons.add(reason)
        time.sleep(max(poll_interval, 0.05))

    if wait_reasons:
        log.info("VALIDATION_RESUMED_AFTER_MERGE sid=%s", sid)


def _wait_for_index_materialized(
    index_path: Path, *, attempts: int = 5, delay: float = 0.5
) -> bool:
    """Return ``True`` when ``index_path`` exists and has a non-zero size."""

    total_attempts = max(1, attempts)

    for attempt in range(1, total_attempts + 1):
        try:
            if index_path.exists() and index_path.stat().st_size > 0:
                return True
        except OSError:
            # Transient filesystem issues should retry until attempts are exhausted.
            pass

        if attempt < total_attempts:
            time.sleep(max(delay, 0.0))

    return False


def _schedule_validation_recheck(sid: str, runs_root: Path, stage: str) -> None:
    delay = random.uniform(_AUTOSEND_RECHECK_MIN_DELAY, _AUTOSEND_RECHECK_MAX_DELAY)

    def _runner() -> None:
        time.sleep(delay)
        try:
            log.info(
                "VALIDATION_AUTOSEND_RECHECK sid=%s stage=%s delay=%.2f",
                sid,
                stage,
                delay,
            )
            _maybe_send_validation_packs(
                sid,
                runs_root,
                stage=stage,
                recheck=True,
            )
        except Exception:  # pragma: no cover - defensive logging
            log.exception(
                "VALIDATION_AUTOSEND_RECHECK_FAILED sid=%s stage=%s",
                sid,
                stage,
            )

    thread = threading.Thread(
        target=_runner,
        name=f"validation-autosend-recheck-{sid}",
        daemon=True,
    )
    thread.start()


def _maybe_send_validation_packs(
    sid: str,
    runs_root: Path,
    *,
    stage: str = "validation",
    recheck: bool = False,
) -> None:
    ensure_validation_section(sid, runs_root=runs_root)

    if not _auto_send_enabled():
        log.info("VALIDATION_AUTOSEND_SKIPPED sid=%s reason=env_disabled", sid)
        return

    from backend.validation.send_packs import send_validation_packs

    index_path = validation_index_path(sid, runs_root=runs_root, create=True)

    if not _wait_for_index_materialized(index_path):
        log.info(
            "VALIDATION_AUTOSEND_SKIPPED sid=%s reason=index_unavailable path=%s",
            sid,
            index_path,
        )
        return

    log.info(
        "VALIDATION_AUTOSEND_TRIGGERED sid=%s stage=%s path=%s",
        sid,
        stage,
        index_path,
    )
    try:
        send_validation_packs(index_path, stage=stage)
    except TypeError as exc:
        if "stage" not in str(exc):
            raise
        send_validation_packs(index_path)

    if not recheck:
        _schedule_validation_recheck(sid, runs_root, stage)


def _get_writer(sid: str, runs_root: Path | str) -> ValidationPackWriter:
    resolved_root = Path(runs_root).resolve()
    per_field = _packs_per_field_enabled()
    key = (str(sid), resolved_root, per_field)
    with _WRITER_CACHE_LOCK:
        writer = _WRITER_CACHE.get(key)
        if writer is None:
            writer = ValidationPackWriter(sid, runs_root=resolved_root, per_field=per_field)
            _WRITER_CACHE[key] = writer
    return writer


def _update_manifest_for_run(sid: str, runs_root: Path | str) -> None:
    runs_root_path = Path(runs_root).resolve()
    base_dir = validation_base_dir(sid, runs_root=runs_root_path, create=True)
    packs_dir = validation_packs_dir(sid, runs_root=runs_root_path, create=True)
    results_dir = validation_results_dir(sid, runs_root=runs_root_path, create=True)
    index_path = validation_index_path(sid, runs_root=runs_root_path, create=True)
    log_path = validation_logs_path(sid, runs_root=runs_root_path, create=True)

    manifest_path = runs_root_path / sid / "manifest.json"
    manifest = RunManifest.load_or_create(manifest_path, sid)
    manifest.upsert_validation_packs_dir(
        base_dir,
        packs_dir=packs_dir,
        results_dir=results_dir,
        index_file=index_path,
        log_file=log_path,
    )
    persist_manifest(manifest)


def build_validation_pack_for_account(
    sid: str,
    account_id: int | str,
    summary_path: Path | str,
    bureaus_path: Path | str,
) -> list[PackLine]:
    """Build and persist the validation pack for ``account_id`` within ``sid``."""

    if not _packs_enabled():
        log.info(
            "VALIDATION_PACKS_DISABLED sid=%s account=%s reason=env_toggle",
            sid,
            account_id,
        )
        return []

    runs_root = _resolve_runs_root_from_artifacts(sid, summary_path, bureaus_path)
    writer = _get_writer(sid, runs_root)
    lines = writer.write_pack_for_account(account_id)
    if writer.last_pack_was_written():
        _update_manifest_for_run(sid, runs_root)
    return lines


def build_validation_packs_for_run(
    sid: str, *, runs_root: Path | str | None = None
) -> dict[int, list[PackLine]]:
    """Build validation packs for every account of ``sid``."""

    runs_root_path = (
        Path(runs_root).resolve() if runs_root is not None else Path("runs").resolve()
    )
    ensure_validation_section(sid, runs_root=runs_root_path)

    if not _packs_enabled():
        log.info(
            "VALIDATION_PACKS_DISABLED sid=%s reason=env_toggle", sid,
        )
        return {}

    _wait_for_merge_completion(sid, runs_root_path)

    writer = _get_writer(sid, runs_root_path)
    results = writer.write_all_packs()
    if any(result for result in results.values()):
        _update_manifest_for_run(sid, runs_root_path)
    return results
