"""Per-account polarity analysis based on bureau data."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping

from backend.core.io.json_io import _atomic_write_json
from backend.core.io.tags import upsert_tag
from backend.core.logic.polarity import classify_field_value, load_polarity_config

logger = logging.getLogger(__name__)

_BUREAU_KEYS: tuple[str, ...] = ("transunion", "experian", "equifax")


def _load_bureaus(account_path: Path, sid: str) -> Dict[str, Any]:
    bureaus_path = account_path / "bureaus.json"
    try:
        raw_text = bureaus_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning(
            "POLARITY_BUREAUS_MISSING sid=%s path=%s",
            sid,
            bureaus_path,
        )
        return {}
    except OSError:
        logger.exception("POLARITY_BUREAUS_READ_FAILED path=%s", bureaus_path)
        return {}

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.exception("POLARITY_BUREAUS_PARSE_FAILED path=%s", bureaus_path)
        return {}

    if not isinstance(data, Mapping):
        logger.warning("POLARITY_BUREAUS_INVALID root_type=%s", type(data).__name__)
        return {}

    return dict(data)


def _load_summary(summary_path: Path) -> Dict[str, Any]:
    try:
        raw_text = summary_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:
        logger.exception("POLARITY_SUMMARY_READ_FAILED path=%s", summary_path)
        return {}

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.exception("POLARITY_SUMMARY_PARSE_FAILED path=%s", summary_path)
        return {}

    if not isinstance(data, Mapping):
        logger.warning("POLARITY_SUMMARY_INVALID root_type=%s", type(data).__name__)
        return {}

    return dict(data)


def _should_write_probes() -> bool:
    flag = os.getenv("WRITE_POLARITY_PROBES")
    if not flag:
        return False
    return flag.strip().lower() in {"1", "true", "yes", "on"}


def _maybe_write_probe_tags(account_dir: Path, payload: Mapping[str, Any]) -> None:
    if not _should_write_probes():
        return

    tag_payload = {"source": "intra_polarity"}
    tag_payload.update(payload)
    upsert_tag(account_dir, tag_payload, unique_keys=("kind", "field", "bureau"))


def _extract_configured_fields() -> tuple[str, ...]:
    config = load_polarity_config()
    fields_cfg = config.get("fields") if isinstance(config, Mapping) else None
    if not isinstance(fields_cfg, Mapping):
        return ()
    return tuple(str(field) for field in fields_cfg.keys())


def analyze_account_polarity(sid: str, account_dir: "os.PathLike[str]") -> Dict[str, Any]:
    """Analyze polarity for bureau fields and persist results."""

    account_path = Path(account_dir)
    bureaus_data = _load_bureaus(account_path, sid)
    fields = _extract_configured_fields()

    polarity_block: Dict[str, Dict[str, Dict[str, str]]] = {}

    for field in fields:
        field_results: Dict[str, Dict[str, str]] = {}
        for bureau_key in _BUREAU_KEYS:
            bureau_values = bureaus_data.get(bureau_key)
            if not isinstance(bureau_values, Mapping):
                continue
            if field not in bureau_values:
                continue
            classification = classify_field_value(field, bureau_values.get(field))
            polarity = str(classification.get("polarity", "unknown"))
            severity = str(classification.get("severity", "low"))
            field_results[bureau_key] = {
                "polarity": polarity,
                "severity": severity,
            }
            _maybe_write_probe_tags(
                account_path,
                {
                    "kind": "polarity_probe",
                    "field": field,
                    "bureau": bureau_key,
                    "polarity": polarity,
                    "severity": severity,
                },
            )
        if field_results:
            polarity_block[field] = field_results

    summary_path = account_path / "summary.json"
    summary_data = _load_summary(summary_path)
    if summary_data.get("polarity_check") != polarity_block:
        summary_data["polarity_check"] = polarity_block
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(summary_path, summary_data)

    return polarity_block


__all__ = ["analyze_account_polarity"]
