"""Validation AI pack payload builder."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from backend.core.ai.paths import (
    validation_pack_filename_for_account,
    validation_packs_dir,
)

log = logging.getLogger(__name__)

_BUREAUS = ("transunion", "experian", "equifax")
_SYSTEM_PROMPT = (
    "You are an adjudication assistant reviewing credit report discrepancies. "
    "Evaluate the provided bureau data and decide if the consumer has a strong claim. "
    "Respond with a JSON object that matches the expected output schema."
)
_EXPECTED_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["decision", "rationale", "citations"],
    "properties": {
        "decision": {"type": "string", "enum": ["strong", "no_case"]},
        "rationale": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


@dataclass(frozen=True)
class PackLine:
    """Single validation pack line ready to be serialized."""

    payload: Mapping[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.payload, ensure_ascii=False, sort_keys=True)


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
        self._per_field = per_field

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def write_all_packs(self) -> dict[int, list[PackLine]]:
        """Build packs for every account under ``cases/accounts``."""

        results: dict[int, list[PackLine]] = {}
        for account_id in self._discover_account_ids():
            results[account_id] = self.write_pack_for_account(account_id)
        return results

    def write_pack_for_account(self, account_id: int | str) -> list[PackLine]:
        """Build and persist the pack for ``account_id``."""

        normalized_id = self._normalize_account_id(account_id)
        pack_lines = self.build_pack_lines(normalized_id)
        pack_path = self._packs_dir / validation_pack_filename_for_account(normalized_id)
        self._write_pack_file(pack_path, pack_lines)
        return pack_lines

    def build_pack_lines(self, account_id: int) -> list[PackLine]:
        """Return the pack lines for ``account_id`` without writing them."""

        summary = self._load_summary(account_id)
        if not summary:
            return []

        validation_block = self._extract_validation_block(summary)
        if not validation_block:
            return []

        requirements = validation_block["requirements"]
        if not requirements:
            return []

        consistency_map = validation_block["field_consistency"]
        bureaus_data = self._load_bureaus(account_id)

        pack_lines: list[PackLine] = []
        for requirement in requirements:
            line = self._build_line(
                account_id,
                requirement,
                bureaus_data,
                consistency_map.get(requirement.get("field")),
            )
            if line is not None:
                pack_lines.append(PackLine(line))

        return pack_lines

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------
    def _write_pack_file(self, path: Path, lines: Sequence[PackLine]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not lines:
            path.write_text("", encoding="utf-8")
            return

        serialized = "\n".join(line.to_json() for line in lines) + "\n"
        path.write_text(serialized, encoding="utf-8")

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    def _build_line(
        self,
        account_id: int,
        requirement: Mapping[str, Any],
        bureaus_data: Mapping[str, Mapping[str, Any]],
        consistency: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        if not isinstance(requirement, Mapping):
            return None

        if not requirement.get("ai_needed"):
            return None

        field = requirement.get("field")
        if field is None:
            return None

        strength = self._normalize_strength(requirement.get("strength"))
        if strength == "strong":
            return None

        field_name = str(field)
        field_key = self._field_key(field_name)
        account_key = f"{account_id:03d}"

        documents = self._normalize_string_list(requirement.get("documents"))
        category = self._coerce_optional_str(requirement.get("category"))
        min_days = self._coerce_optional_int(requirement.get("min_days"))

        context = self._build_context(consistency)
        bureau_values = self._build_bureau_values(
            field_name, bureaus_data, consistency
        )

        prompt_payload = {
            "system": _SYSTEM_PROMPT,
            "user": {
                "sid": self.sid,
                "account_id": account_id,
                "account_key": account_key,
                "field": field_name,
                "field_key": field_key,
                "category": category,
                "documents": documents,
                "bureaus": bureau_values,
                "context": context,
            },
            "guidance": (
                "Return a JSON object with a decision of either 'strong' or 'no_case', "
                "along with rationale and any supporting citations."
            ),
        }

        payload: dict[str, Any] = {
            "id": f"acc_{account_key}__{field_key}",
            "sid": self.sid,
            "account_id": account_id,
            "account_key": account_key,
            "field": field_name,
            "field_key": field_key,
            "category": category,
            "documents": documents,
            "min_days": min_days,
            "strength": strength,
            "bureaus": bureau_values,
            "context": context,
            "prompt": prompt_payload,
            "expected_output": _EXPECTED_OUTPUT_SCHEMA,
        }

        extra_context = requirement.get("notes") or requirement.get("reason")
        if extra_context:
            payload.setdefault("context", {})["requirement_note"] = str(
                extra_context
            )

        return payload

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
        for bureau in _BUREAUS:
            bureau_data = bureaus_data.get(bureau, {})
            raw_value = self._extract_value(raw_map.get(bureau))
            if raw_value is None:
                raw_value = self._extract_value(bureau_data.get(field))

            normalized_value = self._extract_value(normalized_map.get(bureau))

            values[bureau] = {
                "raw": raw_value,
                "normalized": normalized_value,
            }

        return values

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

        requirements = block.get("requirements")
        if not isinstance(requirements, Sequence):
            return None

        consistency = block.get("field_consistency")
        consistency_map = (
            consistency if isinstance(consistency, Mapping) else {}
        )

        return {
            "requirements": list(requirements),
            "field_consistency": consistency_map,
        }

    @staticmethod
    def _normalize_account_id(account_id: int | str) -> int:
        if isinstance(account_id, int):
            return account_id
        return int(str(account_id).strip())

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
