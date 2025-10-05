"""Validation summary orchestration helpers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from backend.core.logic.validation_requirements import (
    build_validation_requirements_for_account,
)
from backend.validation.build_packs import (
    load_manifest_from_source,
    resolve_manifest_paths,
)

log = logging.getLogger(__name__)


@dataclass
class AccountContext:
    """Lightweight wrapper describing a single validation account."""

    sid: str
    runs_root: Path
    index: int | str
    account_key: str
    account_id: str
    account_dir: Path
    summary_path: Path
    bureaus_path: Path
    cached_findings: list[Mapping[str, Any]] | None = field(default=None)


SummaryBuilder = Callable[[str | Path], Mapping[str, Any]]
PackBuilder = Callable[[str, int | str, Path, Path], Sequence[Any]]
SendCallback = Callable[[AccountContext, Sequence[Mapping[str, Any]], Sequence[Any] | None], None]


@dataclass
class ValidationPipelineConfig:
    """Runtime configuration for the validation summary pipeline."""

    build_packs: bool = True
    summary_builder: SummaryBuilder | None = None
    pack_builder: PackBuilder | None = None
    send_callback: SendCallback | None = None


def iterate_accounts(manifest: Mapping[str, Any] | Path | str) -> Iterator[AccountContext]:
    """Yield :class:`AccountContext` objects for every account in ``manifest``."""

    _, paths, runs_root = _prepare_manifest(manifest)
    yield from _iter_account_dirs(paths.sid, runs_root, paths.accounts_dir)


def write_summary_for_account(
    acc_ctx: AccountContext,
    *,
    cfg: ValidationPipelineConfig | None = None,
    runs_root: Path | str | None = None,
    sid: str | None = None,
) -> Mapping[str, Any]:
    """Build and persist validation requirements for ``acc_ctx``."""

    if runs_root is not None:
        acc_ctx.runs_root = Path(runs_root)
    if sid is not None:
        acc_ctx.sid = str(sid)

    config = cfg or ValidationPipelineConfig()
    builder = config.summary_builder or _default_summary_builder

    result = builder(acc_ctx.account_dir)

    block = result.get("validation_requirements") if isinstance(result, Mapping) else None
    findings = _sanitize_findings(block.get("findings")) if isinstance(block, Mapping) else []
    acc_ctx.cached_findings = findings

    return result


def load_findings_from_summary(
    runs_root: Path | str,
    sid: str,
    account_key: int | str,
) -> list[Mapping[str, Any]]:
    """Return cached findings for ``account_key`` from summary.json."""

    summary_path = (
        Path(runs_root)
        / str(sid)
        / "cases"
        / "accounts"
        / str(account_key)
        / "summary.json"
    )

    try:
        raw_text = summary_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError:
        log.debug("SUMMARY_READ_FAILED path=%s", summary_path, exc_info=True)
        return []

    try:
        summary = json.loads(raw_text)
    except Exception:
        log.debug("SUMMARY_PARSE_FAILED path=%s", summary_path, exc_info=True)
        return []

    if not isinstance(summary, Mapping):
        return []

    validation_block = summary.get("validation_requirements")
    if not isinstance(validation_block, Mapping):
        return []

    findings = validation_block.get("findings")
    return _sanitize_findings(findings)


def build_and_queue_packs(
    acc_ctx: AccountContext,
    *,
    findings: Sequence[Mapping[str, Any]] | None,
    cfg: ValidationPipelineConfig | None = None,
) -> Sequence[Any]:
    """Build validation packs for ``acc_ctx`` and optionally enqueue send."""

    config = cfg or ValidationPipelineConfig()
    if not config.build_packs:
        return []

    builder = config.pack_builder or _default_pack_builder
    try:
        pack_lines = builder(
            acc_ctx.sid,
            acc_ctx.index,
            acc_ctx.summary_path,
            acc_ctx.bureaus_path,
        )
    except Exception:  # pragma: no cover - defensive pack builder guard
        log.exception(
            "VALIDATION_PACK_BUILD_FAILED sid=%s account_id=%s",
            acc_ctx.sid,
            acc_ctx.account_id,
        )
        return []

    callback = config.send_callback
    if callback is not None:
        try:
            callback(acc_ctx, list(findings or []), pack_lines)
        except Exception:  # pragma: no cover - defensive queue handling
            log.exception(
                "VALIDATION_PACK_QUEUE_FAILED sid=%s account_id=%s",
                acc_ctx.sid,
                acc_ctx.account_id,
            )

    return pack_lines


def run_validation_summary_pipeline(
    manifest: Mapping[str, Any] | Path | str,
    *,
    cfg: ValidationPipelineConfig | None = None,
) -> dict[str, Any]:
    """Build validation summaries (and packs) for every account in ``manifest``."""

    config = cfg or ValidationPipelineConfig()
    _, paths, runs_root = _prepare_manifest(manifest)

    stats = {
        "sid": paths.sid,
        "total_accounts": 0,
        "summaries_written": 0,
        "packs_built": 0,
        "skipped_accounts": 0,
        "errors": 0,
    }

    for acc_ctx in _iter_account_dirs(paths.sid, runs_root, paths.accounts_dir):
        stats["total_accounts"] += 1
        try:
            result = write_summary_for_account(acc_ctx, cfg=config)
            status = str(result.get("status") or "") if isinstance(result, Mapping) else ""
            if status != "ok":
                stats["skipped_accounts"] += 1
                continue

            stats["summaries_written"] += 1
            findings = acc_ctx.cached_findings
            if findings is None:
                findings = load_findings_from_summary(
                    acc_ctx.runs_root, acc_ctx.sid, acc_ctx.account_key
                )
                acc_ctx.cached_findings = findings

            if config.build_packs and _should_queue_pack(findings):
                pack_lines = build_and_queue_packs(
                    acc_ctx, findings=findings, cfg=config
                )
                if pack_lines:
                    stats["packs_built"] += 1
        except Exception:  # pragma: no cover - defensive pipeline guard
            stats["errors"] += 1
            log.exception(
                "ACCOUNT FAILED, continuing. account_id=%s",
                acc_ctx.account_id,
            )
            continue

    return stats


def _prepare_manifest(
    manifest: Mapping[str, Any] | Path | str,
):
    manifest_data = load_manifest_from_source(manifest)
    paths = resolve_manifest_paths(manifest_data)
    runs_root = _infer_runs_root(paths.accounts_dir, paths.sid)
    return manifest_data, paths, runs_root


def _iter_account_dirs(
    sid: str, runs_root: Path, accounts_dir: Path
) -> Iterable[AccountContext]:
    if not accounts_dir.is_dir():
        return []

    for account_dir in sorted(accounts_dir.iterdir(), key=_account_sort_key):
        if not account_dir.is_dir():
            continue

        account_key = account_dir.name
        coerced_index = _coerce_account_index(account_key)
        account_id = _resolve_account_id(account_dir, account_key)

        yield AccountContext(
            sid=sid,
            runs_root=runs_root,
            index=coerced_index if coerced_index is not None else account_key,
            account_key=account_key,
            account_id=account_id,
            account_dir=account_dir,
            summary_path=account_dir / "summary.json",
            bureaus_path=account_dir / "bureaus.json",
        )


def _account_sort_key(path: Path) -> tuple[int, Any]:
    name = path.name
    try:
        return (0, int(name))
    except (TypeError, ValueError):
        return (1, name)


def _coerce_account_index(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _resolve_account_id(account_dir: Path, default: str) -> str:
    meta_path = account_dir / "meta.json"
    try:
        raw_text = meta_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return str(default)
    except OSError:
        log.debug("META_READ_FAILED path=%s", meta_path, exc_info=True)
        return str(default)

    try:
        meta = json.loads(raw_text)
    except Exception:
        log.debug("META_PARSE_FAILED path=%s", meta_path, exc_info=True)
        return str(default)

    if not isinstance(meta, Mapping):
        return str(default)

    for key in ("account_id", "accountId", "account", "id"):
        value = meta.get(key)
        if value:
            return str(value)

    idx_value = meta.get("account_index") or meta.get("index")
    if idx_value:
        return str(idx_value)

    return str(default)


def _sanitize_findings(findings: Any) -> list[Mapping[str, Any]]:
    if not isinstance(findings, Sequence) or isinstance(findings, (str, bytes, bytearray)):
        return []

    sanitized: list[Mapping[str, Any]] = []
    for entry in findings:
        if isinstance(entry, Mapping):
            sanitized.append(dict(entry))
    return sanitized


def _should_queue_pack(findings: Sequence[Mapping[str, Any]] | None) -> bool:
    if not findings:
        return False
    return any(bool(entry.get("send_to_ai")) for entry in findings if isinstance(entry, Mapping))


def _infer_runs_root(accounts_dir: Path, sid: str) -> Path:
    resolved = accounts_dir.resolve()
    for parent in resolved.parents:
        if parent.name == sid:
            return parent.parent.resolve()
    try:
        return resolved.parents[2].resolve()
    except IndexError:
        return resolved.parent.resolve()


def _default_summary_builder(account_dir: str | Path) -> Mapping[str, Any]:
    return build_validation_requirements_for_account(account_dir, build_pack=False)


def _default_pack_builder(
    sid: str, account_key: int | str, summary_path: Path, bureaus_path: Path
) -> Sequence[Any]:
    from backend.ai.validation_builder import build_validation_pack_for_account

    return build_validation_pack_for_account(sid, account_key, summary_path, bureaus_path)


__all__ = [
    "AccountContext",
    "ValidationPipelineConfig",
    "build_and_queue_packs",
    "iterate_accounts",
    "load_findings_from_summary",
    "run_validation_summary_pipeline",
    "write_summary_for_account",
]

