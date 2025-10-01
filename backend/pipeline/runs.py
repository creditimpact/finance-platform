from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone
from typing import Mapping
import json, os, glob, shutil, time

from backend.core.ai.paths import (
    MergePaths,
    ensure_merge_paths,
    merge_paths_from_any,
)

RUNS_ROOT_ENV = "RUNS_ROOT"                 # optional override
MANIFEST_ENV  = "REPORT_MANIFEST_PATH"      # explicit manifest path

def _runs_root() -> Path:
    rr = os.getenv(RUNS_ROOT_ENV)
    return Path(rr) if rr else Path("runs")

RUNS_ROOT = _runs_root()

def _utc_now():
    # timezone-aware UTC to avoid deprecation; normalize suffix to 'Z'
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def safe_replace(src: str | Path, dst: str | Path, retries: int = 5, delay: float = 0.1) -> None:
    src_path = str(src)
    dst_path = str(dst)
    for i in range(retries):
        try:
            os.replace(src_path, dst_path)
            return
        except PermissionError:
            if i == retries - 1:
                raise
            time.sleep(delay)

@dataclass
class RunManifest:
    path: Path
    data: dict = field(default_factory=dict)

    # -------- creation / loading ----------
    @staticmethod
    def for_sid(sid: str) -> "RunManifest":
        base = _runs_root() / sid
        base.mkdir(parents=True, exist_ok=True)
        m = RunManifest(base / "manifest.json")
        return m._load_or_create(sid)

    @staticmethod
    def from_env_or_latest() -> "RunManifest":
        p = os.getenv(MANIFEST_ENV)
        if p:
            return RunManifest(Path(p)).load()
        cands = glob.glob(str(_runs_root() / "*/manifest.json"))
        if not cands:
            raise FileNotFoundError("No manifests found under runs/*/manifest.json")
        newest = max(map(Path, cands), key=lambda x: x.stat().st_mtime)
        return RunManifest(newest).load()

    @staticmethod
    def load_or_create(path: Path, sid: str | None = None) -> "RunManifest":
        """Load an existing manifest at ``path`` or create a new one."""

        manifest = RunManifest(path)
        if path.exists():
            return manifest.load()

        effective_sid = sid or path.parent.name
        if not effective_sid:
            raise ValueError("sid is required to create a new manifest")

        manifest.path.parent.mkdir(parents=True, exist_ok=True)
        return manifest._load_or_create(effective_sid)

    def _load_or_create(self, sid: str) -> "RunManifest":
        if self.path.exists():
            return self.load()
        self.data = {
            "sid": sid,
            "created_at": _utc_now(),
            "status": "in_progress",
            "base_dirs": {
                "uploads_dir": None,
                "traces_dir": None,
                "cases_dir": None,
                "exports_dir": None,
                "logs_dir": None,
            },
            "ai": {
                "packs": {
                    "base": None,
                    "dir": None,
                    "packs": None,
                    "packs_dir": None,
                    "results": None,
                    "results_dir": None,
                    "index": None,
                    "pairs": 0,
                    "last_built_at": None,
                    "logs": None,
                    "validation": {
                        "base": None,
                        "dir": None,
                        "packs": None,
                        "packs_dir": None,
                        "results": None,
                        "results_dir": None,
                        "index": None,
                        "last_built_at": None,
                        "logs": None,
                    },
                },
                "validation": {
                    "base": None,
                    "dir": None,
                    "accounts": None,
                    "accounts_dir": None,
                    "last_prepared_at": None,
                },
                "status": {
                    "enqueued": False,
                    "built": False,
                    "sent": False,
                    "compacted": False,
                    "skipped_reason": None,
                },
            },
            "artifacts": {
                "uploads": {},
                "traces": {"accounts_table": {}},
                "cases": {},
                "exports": {},
                "logs": {},
            },
            "env_snapshot": {}
        }
        self._update_index(sid)
        (self.path.parent.parent / "current.txt").write_text(sid, encoding="utf-8")
        return self.save()

    def load(self) -> "RunManifest":
        with self.path.open("r", encoding="utf-8") as fh:
            self.data = json.load(fh)
        return self

    def save(self) -> "RunManifest":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        self._upgrade_ai_packs_structure()
        self._mirror_ai_to_legacy_artifacts()
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self.data, fh, ensure_ascii=False, indent=2)
        safe_replace(tmp, self.path)
        return self

    def _upgrade_ai_packs_structure(self) -> None:
        ai_section = self.data.get("ai")
        if not isinstance(ai_section, dict):
            return

        packs_section = ai_section.get("packs")
        if not isinstance(packs_section, dict):
            return

        base_value = packs_section.get("base")
        packs_value = packs_section.get("packs")
        results_value = packs_section.get("results")
        dir_value = packs_section.get("dir")

        needs_upgrade = not (base_value and packs_value and results_value)

        if not needs_upgrade and isinstance(dir_value, str):
            normalized = dir_value.rstrip("/\\")
            if normalized.endswith("ai_packs"):
                needs_upgrade = True

        if not needs_upgrade:
            return

        run_root = self.path.parent
        runs_root = run_root.parent
        canonical_paths = ensure_merge_paths(runs_root, self.sid, create=False)

        merge_paths: MergePaths | None = None
        candidates = [
            packs_section.get("base"),
            packs_section.get("packs"),
            packs_section.get("results"),
            packs_section.get("dir"),
            packs_section.get("packs_dir"),
            packs_section.get("results_dir"),
        ]

        for candidate in candidates:
            if not candidate:
                continue
            try:
                merge_paths = merge_paths_from_any(Path(candidate), create=False)
                break
            except ValueError:
                continue

        if merge_paths is None:
            merge_paths = canonical_paths

        existing_logs = packs_section.get("logs")
        prefer_existing_index = bool(packs_section.get("index"))
        self._apply_merge_paths_to_packs(
            packs_section,
            merge_paths,
            prefer_existing_index=prefer_existing_index,
        )

        if existing_logs:
            packs_section["logs"] = existing_logs

    def _mirror_ai_to_legacy_artifacts(self) -> None:
        ai_section = self.data.get("ai")
        if not isinstance(ai_section, dict):
            return

        artifacts = self.data.setdefault("artifacts", {})
        artifacts.pop("ai_packs", None)

        legacy_ai = artifacts.setdefault("ai", {})

        packs = ai_section.get("packs")
        if isinstance(packs, dict):
            legacy_packs = legacy_ai.setdefault("packs", {})
            for key in (
                "base",
                "dir",
                "packs",
                "packs_dir",
                "results",
                "results_dir",
                "index",
                "pairs",
                "last_built_at",
                "logs",
            ):
                if key in packs:
                    legacy_packs[key] = packs[key]

        status = ai_section.get("status")
        if isinstance(status, dict):
            legacy_status = legacy_ai.setdefault("status", {})
            for key, value in status.items():
                legacy_status[key] = value

    def _update_index(self, sid: str) -> None:
        idx = _runs_root() / "index.json"
        rec = {"sid": sid, "created_at": _utc_now()}
        if idx.exists():
            try:
                obj = json.loads(idx.read_text(encoding="utf-8"))
            except Exception:
                obj = {"runs": []}
        else:
            obj = {"runs": []}
        obj.setdefault("runs", []).append(rec)
        tmp = idx.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        tmp.replace(idx)

    # -------- API ----------
    def snapshot_env(self, keys: list[str]) -> "RunManifest":
        for k in keys:
            v = os.getenv(k)
            if v is not None:
                self.data["env_snapshot"][k] = v
        return self.save()

    def set_base_dir(self, label: str, path: Path) -> "RunManifest":
        resolved = Path(path).resolve()
        self.data.setdefault("base_dirs", {})[label] = str(resolved)
        return self.save()

    def set_artifact(self, group: str, name: str, path: str | Path) -> "RunManifest":
        """Record an artifact path under ``artifacts.<group>.<name>``.

        Parameters
        ----------
        group:
            Dotted path indicating the artifact grouping (for example
            ``"traces.accounts_table"``).  Missing intermediate dictionaries are
            created automatically.
        name:
            Artifact identifier within the group.
        path:
            Filesystem location to store.  The path is normalized to an absolute
            string representation.
        """

        resolved_path = str(Path(path).resolve())
        cursor: dict[str, object] = self.data.setdefault("artifacts", {})
        for part in str(group).split("."):
            if not part:
                continue
            next_cursor = cursor.setdefault(part, {})
            if not isinstance(next_cursor, dict):
                raise TypeError(
                    f"Cannot assign artifact into non-mapping at group '{group}'"
                )
            cursor = next_cursor
        cursor[str(name)] = resolved_path
        return self.save()

    def _ensure_ai_section(self) -> tuple[dict[str, object], dict[str, object]]:
        ai = self.data.setdefault("ai", {})
        packs = ai.setdefault(
            "packs",
            {
                "base": None,
                "dir": None,
                "packs": None,
                "packs_dir": None,
                "results": None,
                "results_dir": None,
                "index": None,
                "pairs": 0,
                "last_built_at": None,
                "logs": None,
            },
        )
        validation_section = packs.get("validation")
        if not isinstance(validation_section, dict):
            validation_section = {
                "base": None,
                "dir": None,
                "packs": None,
                "packs_dir": None,
                "results": None,
                "results_dir": None,
                "index": None,
                "last_built_at": None,
                "logs": None,
            }
            packs["validation"] = validation_section
        else:
            for key in (
                "base",
                "dir",
                "packs",
                "packs_dir",
                "results",
                "results_dir",
                "index",
                "last_built_at",
                "logs",
            ):
                validation_section.setdefault(key, None)
        ai.setdefault(
            "validation",
            {
                "base": None,
                "dir": None,
                "accounts": None,
                "accounts_dir": None,
                "last_prepared_at": None,
            },
        )
        status = ai.setdefault(
            "status",
            {
                "enqueued": False,
                "built": False,
                "sent": False,
                "compacted": False,
                "skipped_reason": None,
            },
        )
        return packs, status

    def _ensure_ai_validation_pack_section(self) -> dict[str, object]:
        packs, _ = self._ensure_ai_section()
        validation = packs.setdefault(
            "validation",
            {
                "base": None,
                "dir": None,
                "packs": None,
                "packs_dir": None,
                "results": None,
                "results_dir": None,
                "index": None,
                "last_built_at": None,
                "logs": None,
            },
        )
        if not isinstance(validation, dict):
            raise TypeError("ai.packs.validation must be a mapping")
        return validation

    def _apply_merge_paths_to_packs(
        self,
        packs: dict[str, object],
        merge_paths: MergePaths,
        *,
        prefer_existing_index: bool = False,
    ) -> None:
        base_str = str(merge_paths.base)
        packs_str = str(merge_paths.packs_dir)
        results_str = str(merge_paths.results_dir)

        packs["base"] = base_str
        packs["dir"] = base_str
        packs["packs"] = packs_str
        packs["packs_dir"] = packs_str
        packs["results"] = results_str
        packs["results_dir"] = results_str
        packs["logs"] = str(merge_paths.log_file)

        if not (prefer_existing_index and packs.get("index")):
            packs["index"] = str(merge_paths.index_file)

    def _ensure_validation_section(self) -> dict[str, object]:
        ai = self.data.setdefault("ai", {})
        validation = ai.setdefault(
            "validation",
            {
                "base": None,
                "dir": None,
                "accounts": None,
                "accounts_dir": None,
                "last_prepared_at": None,
            },
        )
        return validation

    def upsert_validation_packs_dir(
        self,
        base_dir: Path,
        *,
        packs_dir: Path | None = None,
        results_dir: Path | None = None,
        index_file: Path | None = None,
        log_file: Path | None = None,
        account_dir: Path | None = None,
    ) -> "RunManifest":
        validation = self._ensure_validation_section()
        resolved = Path(base_dir).resolve()
        resolved_str = str(resolved)
        timestamp = _utc_now()

        validation["base"] = resolved_str
        validation["dir"] = resolved_str
        validation["accounts"] = resolved_str
        validation["accounts_dir"] = resolved_str
        validation["last_prepared_at"] = timestamp

        packs_validation = self._ensure_ai_validation_pack_section()
        packs_validation["base"] = resolved_str

        pack_path: Path
        if packs_dir is not None:
            pack_path = Path(packs_dir).resolve()
        elif account_dir is not None:
            pack_path = Path(account_dir).resolve()
        else:
            pack_path = (resolved / "packs").resolve()
        packs_validation["dir"] = resolved_str
        packs_validation["packs"] = str(pack_path)
        packs_validation["packs_dir"] = str(pack_path)

        if results_dir is not None:
            results_path = Path(results_dir).resolve()
        elif account_dir is not None:
            results_path = (Path(account_dir).resolve() / "results").resolve()
        else:
            results_path = (resolved / "results").resolve()
        packs_validation["results"] = str(results_path)
        packs_validation["results_dir"] = str(results_path)

        index_path = (
            Path(index_file).resolve(strict=False)
            if index_file is not None
            else (resolved / "index.json").resolve(strict=False)
        )
        packs_validation["index"] = str(index_path)

        log_path = (
            Path(log_file).resolve(strict=False)
            if log_file is not None
            else (resolved / "logs.txt").resolve(strict=False)
        )
        packs_validation["logs"] = str(log_path)
        packs_validation["last_built_at"] = timestamp

        return self.save()

    def upsert_ai_packs_dir(self, packs_dir: Path) -> "RunManifest":
        packs, _ = self._ensure_ai_section()
        merge_paths = merge_paths_from_any(packs_dir, create=False)
        self._apply_merge_paths_to_packs(packs, merge_paths, prefer_existing_index=True)
        return self

    def set_ai_enqueued(self) -> "RunManifest":
        _, status = self._ensure_ai_section()
        status["enqueued"] = True
        status["skipped_reason"] = None
        return self.save()

    def set_ai_built(self, packs_dir: Path, pairs: int) -> "RunManifest":
        packs, status = self._ensure_ai_section()
        merge_paths = merge_paths_from_any(packs_dir, create=False)
        self._apply_merge_paths_to_packs(packs, merge_paths)
        packs["pairs"] = int(pairs)
        packs["last_built_at"] = _utc_now()
        status["built"] = True
        status["skipped_reason"] = None
        return self.save()

    def set_ai_sent(self) -> "RunManifest":
        _, status = self._ensure_ai_section()
        status["sent"] = True
        return self.save()

    def set_ai_compacted(self) -> "RunManifest":
        _, status = self._ensure_ai_section()
        status["compacted"] = True
        return self.save()

    def set_ai_skipped(self, reason: str) -> "RunManifest":
        _, status = self._ensure_ai_section()
        status["skipped_reason"] = str(reason)
        status["built"] = False
        status["sent"] = False
        status["compacted"] = False
        return self.save()

    def update_ai_packs(
        self,
        *,
        dir: Path | str | None = None,
        index: Path | str | None = None,
        logs: Path | str | None = None,
        pairs: int | None = None,
        last_built_at: str | None = None,
    ) -> "RunManifest":
        packs, _ = self._ensure_ai_section()

        merge_paths: MergePaths | None = None
        if dir is not None:
            dir_path = Path(dir).resolve()
            try:
                merge_paths = merge_paths_from_any(dir_path, create=False)
            except ValueError:
                packs["legacy_dir"] = str(dir_path)
                packs["legacy_packs_dir"] = str(dir_path)
            else:
                self._apply_merge_paths_to_packs(
                    packs,
                    merge_paths,
                    prefer_existing_index=index is None,
                )
                packs.pop("legacy_dir", None)
                packs.pop("legacy_packs_dir", None)

        if index is not None:
            packs["index"] = str(Path(index).resolve())

        if logs is not None:
            packs["logs"] = str(Path(logs).resolve())
        elif merge_paths is not None:
            packs["logs"] = str(merge_paths.log_file)

        if pairs is not None:
            packs["pairs"] = int(pairs)

        if last_built_at is not None:
            packs["last_built_at"] = str(last_built_at)

        return self.save()

    def get_ai_merge_paths(self) -> dict[str, Path | None]:
        """Return the resolved merge AI pack locations for this manifest.

        The returned mapping always includes canonical ``merge`` paths rooted at
        ``runs/<sid>/ai_packs/merge``.  When the manifest still references the
        legacy flat ``ai_packs`` directory, ``legacy_dir`` points to that base so
        callers can perform read-only operations without migrating data.  The
        ``index_file`` and ``log_file`` entries prefer existing files regardless
        of layout.
        """

        run_root = self.path.parent
        runs_root = run_root.parent
        canonical_paths = ensure_merge_paths(runs_root, self.sid, create=False)

        ai_section = self.data.get("ai")
        packs_section: dict[str, object] = {}
        if isinstance(ai_section, dict):
            packs_value = ai_section.get("packs")
            if isinstance(packs_value, dict):
                packs_section = packs_value

        base_value = packs_section.get("base")
        packs_value = packs_section.get("packs")
        results_value = packs_section.get("results")
        dir_value = packs_section.get("dir")
        packs_dir_value = packs_section.get("packs_dir")
        results_dir_value = packs_section.get("results_dir")
        index_value = packs_section.get("index")
        logs_value = packs_section.get("logs")
        legacy_dir_value = packs_section.get("legacy_dir")
        legacy_packs_dir_value = packs_section.get("legacy_packs_dir")

        merge_paths = canonical_paths
        legacy_dir: Path | None = None
        legacy_packs_dir: Path | None = None

        def _apply_candidate(value: object) -> None:
            nonlocal merge_paths, legacy_dir, legacy_packs_dir
            if not value:
                return
            try:
                candidate_path = Path(value)
            except (TypeError, ValueError):
                return
            try:
                merge_paths = merge_paths_from_any(candidate_path, create=False)
            except ValueError:
                resolved = candidate_path.resolve()
                if legacy_dir is None:
                    legacy_dir = resolved
                    legacy_packs_dir = resolved

        for candidate in (
            base_value,
            packs_value,
            results_value,
            dir_value,
            packs_dir_value,
            results_dir_value,
        ):
            _apply_candidate(candidate)
            if merge_paths is not canonical_paths:
                break

        base_path = merge_paths.base
        packs_dir = merge_paths.packs_dir
        results_dir = merge_paths.results_dir

        if legacy_dir is None and legacy_dir_value:
            try:
                legacy_dir = Path(legacy_dir_value).resolve()
            except (TypeError, ValueError):
                legacy_dir = None
        if legacy_packs_dir is None and legacy_packs_dir_value:
            try:
                legacy_packs_dir = Path(legacy_packs_dir_value).resolve()
            except (TypeError, ValueError):
                legacy_packs_dir = None

        index_candidates: list[Path] = []
        if index_value:
            try:
                index_candidates.append(Path(index_value).resolve())
            except (TypeError, ValueError):
                pass
        index_candidates.append(merge_paths.index_file)
        if canonical_paths.index_file not in index_candidates:
            index_candidates.append(canonical_paths.index_file)
        if legacy_dir is not None:
            index_candidates.append((legacy_dir / "index.json").resolve())

        index_file: Path | None = None
        for candidate in index_candidates:
            if candidate.exists():
                index_file = candidate
                break
        if index_file is None and index_candidates:
            index_file = index_candidates[0]

        log_candidates: list[Path] = []
        if logs_value:
            try:
                log_candidates.append(Path(logs_value).resolve())
            except (TypeError, ValueError):
                pass
        log_candidates.append(merge_paths.log_file)
        if canonical_paths.log_file not in log_candidates:
            log_candidates.append(canonical_paths.log_file)
        if legacy_dir is not None:
            log_candidates.append((legacy_dir / "logs.txt").resolve())

        log_file: Path | None = None
        for candidate in log_candidates:
            if candidate.exists():
                log_file = candidate
                break
        if log_file is None and log_candidates:
            log_file = log_candidates[0]

        paths: dict[str, Path | None] = {
            "base": base_path,
            "packs_dir": packs_dir.resolve(),
            "packs": packs_dir.resolve(),
            "results_dir": results_dir.resolve(),
            "results": results_dir.resolve(),
            "index_file": index_file,
            "log_file": log_file,
        }

        if legacy_dir is not None:
            paths["legacy_dir"] = legacy_dir.resolve()
            if legacy_packs_dir is not None:
                paths["legacy_packs_dir"] = legacy_packs_dir.resolve()

        return paths

    def get_ai_packs_dir(self) -> Path | None:
        paths = self.get_ai_merge_paths()
        packs_dir = paths.get("packs_dir")
        return packs_dir if isinstance(packs_dir, Path) else None

    def get_ai_index_path(self) -> Path | None:
        paths = self.get_ai_merge_paths()
        index_path = paths.get("index_file")
        return index_path if isinstance(index_path, Path) else None

    def ensure_run_subdir(self, label: str, rel: str) -> Path:
        """
        Ensure runs/<SID>/<rel> exists, register it as ``base_dirs[label]``,
        and return the absolute :class:`~pathlib.Path`.
        """

        base = (self.path.parent / rel).resolve()
        base.mkdir(parents=True, exist_ok=True)
        self.set_base_dir(label, base)
        return base

    def get(self, group: str, key: str) -> str:
        cursor = self.data.get("artifacts", {})
        for part in group.split("."):
            cursor = cursor.get(part, {})
        if key not in cursor:
            raise KeyError(f"Missing {group}.{key} in manifest")
        return cursor[key]

    @property
    def sid(self) -> str:
        return str(self.data.get("sid"))

# -------- breadcrumbs --------
def write_breadcrumb(target_manifest: Path, breadcrumb_file: Path) -> None:
    breadcrumb_file.write_text(str(target_manifest.resolve()), encoding="utf-8")


def persist_manifest(
    manifest: RunManifest,
    *,
    artifacts: Mapping[str, Mapping[str, Path | str | int]] | None = None,
) -> RunManifest:
    """Persist ``manifest`` after applying artifact path updates.

    Parameters
    ----------
    manifest:
        The manifest instance to persist.
    artifacts:
        Optional mapping of artifact groups to update before saving.  Values are
        converted to absolute paths.
    """

    if artifacts:
        manifest_artifacts = manifest.data.setdefault("artifacts", {})
        for group, entries in artifacts.items():
            if group == "ai_packs":
                packs_updates = dict(entries)
                packs, _ = manifest._ensure_ai_section()

                base_value = packs_updates.pop("base", None)
                dir_value = packs_updates.pop("dir", None)
                packs_value = packs_updates.pop("packs", None)
                packs_dir_value = packs_updates.pop("packs_dir", None)
                results_value = packs_updates.pop("results", None)
                results_dir_value = packs_updates.pop("results_dir", None)

                index_value_raw = packs_updates.pop("index", None)
                logs_value_raw = packs_updates.pop("logs", None)
                pairs_value = packs_updates.pop("pairs", None)
                pairs_count_value = packs_updates.pop("pairs_count", None)
                last_built_value = packs_updates.pop("last_built_at", None)

                merge_paths: MergePaths | None = None
                candidate_pairs = [
                    ("base", base_value),
                    ("packs", packs_value),
                    ("results", results_value),
                    ("dir", dir_value),
                    ("packs_dir", packs_dir_value),
                    ("results_dir", results_dir_value),
                ]

                prefer_existing_index = True

                legacy_dir_resolved: Path | None = None
                for key, candidate in candidate_pairs:
                    if candidate is None or str(candidate) == "":
                        continue
                    try:
                        merge_paths = merge_paths_from_any(Path(candidate), create=False)
                    except ValueError:
                        if key in {"base", "dir", "packs", "packs_dir"} and legacy_dir_resolved is None:
                            try:
                                legacy_dir_resolved = Path(candidate).resolve()
                            except (TypeError, ValueError):
                                legacy_dir_resolved = None
                        continue

                    prefer_existing_index = index_value_raw is None or str(index_value_raw) == ""
                    manifest._apply_merge_paths_to_packs(
                        packs,
                        merge_paths,
                        prefer_existing_index=prefer_existing_index,
                    )
                    packs.pop("legacy_dir", None)
                    packs.pop("legacy_packs_dir", None)
                    break

                if legacy_dir_resolved is not None:
                    packs["legacy_dir"] = str(legacy_dir_resolved)
                    packs["legacy_packs_dir"] = str(legacy_dir_resolved)

                index_value: Path | None = None
                if index_value_raw is not None and str(index_value_raw) != "":
                    index_value = Path(index_value_raw).resolve()

                if index_value is not None:
                    packs["index"] = str(index_value)
                elif packs.get("dir") and not packs.get("index"):
                    packs["index"] = str((Path(packs["dir"]) / "index.json").resolve())

                if logs_value_raw is not None and str(logs_value_raw) != "":
                    packs["logs"] = str(Path(logs_value_raw).resolve())
                elif merge_paths is not None:
                    packs["logs"] = str(merge_paths.log_file)

                if pairs_value is not None:
                    packs["pairs"] = int(pairs_value)

                if pairs_count_value is not None:
                    packs["pairs"] = int(pairs_count_value)

                if last_built_value is not None:
                    packs["last_built_at"] = str(last_built_value)

                for extra_key, extra_value in packs_updates.items():
                    packs[str(extra_key)] = extra_value

                continue

            cursor = manifest_artifacts
            for part in str(group).split("."):
                cursor = cursor.setdefault(part, {})
            for key, value in entries.items():
                cursor[str(key)] = str(Path(value).resolve())
    return manifest.save()

