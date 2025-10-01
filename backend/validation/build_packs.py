"""Utilities for constructing Validation AI packs from prepared cases."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from backend.core.ai.paths import validation_pack_filename_for_account


_SYSTEM_PROMPT = (
    "You are an adjudication assistant reviewing credit report discrepancies. "
    "Evaluate the provided bureau data and decide if the consumer has a strong claim. "
    "Respond with a JSON object that matches the expected output schema."
)
_GUIDANCE_TEXT = (
    "Return a JSON object with a decision of either 'strong' or 'no_case', "
    "along with rationale and any supporting citations."
)
_EXPECTED_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["decision", "rationale", "citations"],
    "properties": {
        "decision": {"type": "string", "enum": ["strong", "no_case"]},
        "rationale": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
}
_BUREAUS = ("transunion", "experian", "equifax")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


@dataclass(frozen=True)
class ManifestPaths:
    """Resolved locations for validation pack inputs and outputs."""

    sid: str
    accounts_dir: Path
    packs_dir: Path
    results_dir: Path
    index_path: Path
    log_path: Path


class ValidationPackBuilder:
    """Build per-account Validation AI pack payloads."""

    def __init__(self, manifest: Mapping[str, Any]) -> None:
        self.paths = self._resolve_manifest_paths(manifest)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self) -> list[dict[str, Any]]:
        """Generate packs for every account referenced in the manifest."""

        items: list[dict[str, Any]] = []
        for account_id, account_dir in self._iter_accounts():
            payloads = self._build_account_pack(account_id, account_dir)
            pack_path = self._write_pack(account_id, payloads)
            items.append(
                {
                    "account_id": account_id,
                    "pack": str(pack_path.resolve()),
                }
            )
        self._write_index(items)
        return items

    # ------------------------------------------------------------------
    # Account pack construction
    # ------------------------------------------------------------------
    def _iter_accounts(self) -> Iterable[tuple[int, Path]]:
        accounts_dir = self.paths.accounts_dir
        if not accounts_dir.is_dir():
            return []

        for child in sorted(accounts_dir.iterdir()):
            if not child.is_dir():
                continue
            try:
                account_id = int(child.name)
            except (TypeError, ValueError):
                continue
            yield account_id, child

    def _build_account_pack(self, account_id: int, account_dir: Path) -> list[dict[str, Any]]:
        summary = self._read_json(account_dir / "summary.json")
        if not isinstance(summary, Mapping):
            return []

        validation_block = summary.get("validation_requirements")
        if not isinstance(validation_block, Mapping):
            return []

        requirements = validation_block.get("requirements")
        if not isinstance(requirements, Sequence):
            return []

        field_consistency = validation_block.get("field_consistency")
        if not isinstance(field_consistency, Mapping):
            field_consistency = {}

        bureaus = self._read_json(account_dir / "bureaus.json")
        bureaus_map: Mapping[str, Mapping[str, Any]]
        if isinstance(bureaus, Mapping):
            normalized_bureaus: dict[str, dict[str, Any]] = {}
            for name, value in bureaus.items():
                if not isinstance(value, Mapping):
                    continue
                bureau_key = str(name).strip().lower()
                normalized_bureaus[bureau_key] = {
                    str(key): val for key, val in value.items()
                }
            bureaus_map = normalized_bureaus
        else:
            bureaus_map = {}

        payloads: list[dict[str, Any]] = []
        for requirement in requirements:
            if not isinstance(requirement, Mapping):
                continue

            normalized_strength = self._normalize_strength(requirement.get("strength"))
            include = bool(requirement.get("ai_needed")) or normalized_strength == "weak"
            if not include:
                continue

            field = requirement.get("field")
            if not field:
                continue

            line = self._build_line(
                account_id,
                requirement,
                normalized_strength,
                bureaus_map,
                field_consistency.get(str(field)),
            )
            if line is not None:
                payloads.append(line)

        return payloads

    def _build_line(
        self,
        account_id: int,
        requirement: Mapping[str, Any],
        strength: str,
        bureaus: Mapping[str, Mapping[str, Any]],
        consistency: object,
    ) -> dict[str, Any] | None:
        field = requirement.get("field")
        if not isinstance(field, str) or not field.strip():
            return None
        field_name = field.strip()
        field_key = self._field_key(field_name)
        account_key = f"{account_id:03d}"

        documents = self._normalize_string_list(requirement.get("documents"))
        category = self._coerce_optional_str(requirement.get("category"))
        min_days = self._coerce_optional_int(requirement.get("min_days"))

        context = self._build_context(consistency)
        extra_context = requirement.get("notes") or requirement.get("reason")
        if extra_context:
            context.setdefault("requirement_note", str(extra_context))

        bureau_values = self._build_bureau_values(field_name, bureaus, consistency)

        prompt_user = {
            "sid": self.paths.sid,
            "account_id": account_id,
            "account_key": account_key,
            "field": field_name,
            "field_key": field_key,
            "category": category,
            "documents": documents,
            "bureaus": bureau_values,
            "context": context,
        }

        payload: dict[str, Any] = {
            "sid": self.paths.sid,
            "account_id": account_id,
            "account_key": account_key,
            "id": f"acc_{account_key}__{field_key}",
            "field": field_name,
            "field_key": field_key,
            "category": category,
            "documents": documents,
            "min_days": min_days,
            "strength": strength,
            "bureaus": bureau_values,
            "context": context,
            "expected_output": _EXPECTED_OUTPUT_SCHEMA,
            "prompt": {
                "system": _SYSTEM_PROMPT,
                "guidance": _GUIDANCE_TEXT,
                "user": prompt_user,
            },
        }

        return payload

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def _write_pack(self, account_id: int, payloads: Sequence[Mapping[str, Any]]) -> Path:
        packs_dir = self.paths.packs_dir
        packs_dir.mkdir(parents=True, exist_ok=True)
        pack_path = packs_dir / validation_pack_filename_for_account(account_id)

        if not payloads:
            pack_path.write_text("", encoding="utf-8")
        else:
            lines = [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in payloads]
            pack_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        self._log(
            "pack_created",
            account_id=f"{account_id:03d}",
            pack=str(pack_path.resolve()),
            fields=len(payloads),
        )
        return pack_path

    def _write_index(self, items: Sequence[Mapping[str, Any]]) -> None:
        self.paths.results_dir.mkdir(parents=True, exist_ok=True)

        index_payload = {
            "sid": self.paths.sid,
            "packs_dir": str(self.paths.packs_dir.resolve()),
            "results_dir": str(self.paths.results_dir.resolve()),
            "items": list(items),
        }
        index_path = self.paths.index_path
        index_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(index_payload, ensure_ascii=False, indent=2)
        index_path.write_text(serialized + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_manifest_paths(manifest: Mapping[str, Any]) -> ManifestPaths:
        sid = str(manifest.get("sid") or "").strip()
        if not sid:
            raise ValueError("Manifest is missing 'sid'")

        base_dirs = manifest.get("base_dirs")
        if not isinstance(base_dirs, Mapping):
            raise ValueError("Manifest is missing 'base_dirs'")

        accounts_dir_raw = base_dirs.get("cases_accounts_dir")
        if not accounts_dir_raw:
            raise ValueError("Manifest missing base_dirs.cases_accounts_dir")
        accounts_dir = Path(str(accounts_dir_raw)).resolve()

        ai_section = manifest.get("ai")
        if not isinstance(ai_section, Mapping):
            raise ValueError("Manifest missing 'ai' section")

        packs_section = ai_section.get("packs")
        if not isinstance(packs_section, Mapping):
            raise ValueError("Manifest missing ai.packs")

        validation_section = packs_section.get("validation")
        if not isinstance(validation_section, Mapping):
            raise ValueError("Manifest missing ai.packs.validation")

        packs_dir_raw = validation_section.get("packs_dir") or validation_section.get("packs")
        results_dir_raw = validation_section.get("results_dir") or validation_section.get("results")
        index_path_raw = validation_section.get("index")
        log_path_raw = (
            validation_section.get("logs")
            or validation_section.get("log")
            or validation_section.get("log_file")
        )

        if not packs_dir_raw:
            raise ValueError("Manifest missing ai.packs.validation.packs_dir")
        if not results_dir_raw:
            raise ValueError("Manifest missing ai.packs.validation.results_dir")
        if not index_path_raw:
            raise ValueError("Manifest missing ai.packs.validation.index")
        if not log_path_raw:
            raise ValueError("Manifest missing ai.packs.validation.logs")

        packs_dir = Path(str(packs_dir_raw))
        results_dir = Path(str(results_dir_raw))
        index_path = Path(str(index_path_raw))
        log_path = Path(str(log_path_raw))

        return ManifestPaths(
            sid=sid,
            accounts_dir=accounts_dir,
            packs_dir=packs_dir,
            results_dir=results_dir,
            index_path=index_path,
            log_path=log_path,
        )

    @staticmethod
    def _normalize_strength(strength: Any) -> str:
        if isinstance(strength, str):
            normalized = strength.strip().lower()
            if normalized in {"weak", "soft"}:
                return "weak"
            if normalized:
                return normalized
        return "unknown"

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
        try:
            text = str(value)
        except Exception:
            return None
        return text.strip() or None

    @staticmethod
    def _normalize_string_list(value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
            return []
        result: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                result.append(text)
        return result

    @staticmethod
    def _field_key(field: str) -> str:
        key = re.sub(r"[^a-z0-9]+", "_", field.strip().lower())
        return key.strip("_") or "field"

    @staticmethod
    def _build_context(consistency: object) -> dict[str, Any]:
        if not isinstance(consistency, Mapping):
            return {}

        context: dict[str, Any] = {}
        consensus = ValidationPackBuilder._coerce_optional_str(consistency.get("consensus"))
        if consensus:
            context["consensus"] = consensus

        disagreeing = ValidationPackBuilder._normalize_string_list(
            consistency.get("disagreeing_bureaus")
        )
        if disagreeing:
            context["disagreeing_bureaus"] = disagreeing

        missing = ValidationPackBuilder._normalize_string_list(
            consistency.get("missing_bureaus")
        )
        if missing:
            context["missing_bureaus"] = missing

        history = consistency.get("history")
        if isinstance(history, Mapping):
            context["history"] = ValidationPackBuilder._normalize_history(history)

        return context

    @staticmethod
    def _normalize_history(history: Mapping[str, Any]) -> Mapping[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in history.items():
            try:
                normalized[str(key)] = value
            except Exception:
                continue
        return normalized

    @staticmethod
    def _build_bureau_values(
        field: str,
        bureaus: Mapping[str, Mapping[str, Any]],
        consistency: object,
    ) -> dict[str, dict[str, Any]]:
        raw_map: Mapping[str, Any] = {}
        normalized_map: Mapping[str, Any] = {}
        if isinstance(consistency, Mapping):
            raw_values = consistency.get("raw")
            if isinstance(raw_values, Mapping):
                raw_map = raw_values
            normalized_values = consistency.get("normalized")
            if isinstance(normalized_values, Mapping):
                normalized_map = normalized_values

        values: dict[str, dict[str, Any]] = {}
        for bureau in _BUREAUS:
            bureau_data = bureaus.get(bureau, {})
            if not isinstance(bureau_data, Mapping):
                bureau_data = {}

            raw_value = ValidationPackBuilder._extract_value(raw_map.get(bureau))
            if raw_value is None:
                raw_value = ValidationPackBuilder._extract_value(bureau_data.get(field))

            normalized_value = ValidationPackBuilder._extract_value(normalized_map.get(bureau))

            values[bureau] = {
                "raw": raw_value,
                "normalized": normalized_value,
            }

        return values

    @staticmethod
    def _extract_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            for candidate in ("raw", "normalized", "value", "text"):
                if candidate in value:
                    return value[candidate]
            return dict(value)
        return value

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        if not text.strip():
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log(self, event: str, **payload: Any) -> None:
        log_path = self.paths.log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": _utc_now(),
            "sid": self.paths.sid,
            "event": event,
        }
        record.update(payload)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def load_manifest_from_source(
    manifest: Mapping[str, Any] | Path | str,
) -> Mapping[str, Any]:
    """Return a manifest mapping from ``manifest`` regardless of input type."""
    if isinstance(manifest, Mapping):
        return manifest

    manifest_path = Path(manifest)
    manifest_text = manifest_path.read_text(encoding="utf-8")
    data = json.loads(manifest_text)
    if not isinstance(data, Mapping):
        raise TypeError("Manifest root must be a mapping")
    return data


def build_validation_packs(
    manifest: Mapping[str, Any] | Path | str,
) -> list[dict[str, Any]]:
    """Build Validation AI packs for every account defined by ``manifest``."""

    manifest_data = load_manifest_from_source(manifest)
    builder = ValidationPackBuilder(manifest_data)
    return builder.build()


def resolve_manifest_paths(manifest: Mapping[str, Any]) -> ManifestPaths:
    """Return the resolved :class:`ManifestPaths` from ``manifest``."""

    return ValidationPackBuilder._resolve_manifest_paths(manifest)


__all__ = [
    "build_validation_packs",
    "load_manifest_from_source",
    "resolve_manifest_paths",
    "ValidationPackBuilder",
    "ManifestPaths",
]
