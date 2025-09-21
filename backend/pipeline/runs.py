from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone
from typing import Mapping
import json, os, glob, shutil, time

RUNS_ROOT_ENV = "RUNS_ROOT"                 # optional override
MANIFEST_ENV  = "REPORT_MANIFEST_PATH"      # explicit manifest path

def _runs_root() -> Path:
    rr = os.getenv(RUNS_ROOT_ENV)
    return Path(rr) if rr else Path("runs")

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
            "artifacts": {
                "uploads": {},
                "traces": {"accounts_table": {}},
                "cases": {},
                "exports": {},
                "logs": {},
                "ai": {},
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
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self.data, fh, ensure_ascii=False, indent=2)
        safe_replace(tmp, self.path)
        return self

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

    def set_artifact(self, group: str, key: str, value: Path | str) -> "RunManifest":
        artifacts = self.data.setdefault("artifacts", {})
        # nested group keys like "traces.accounts_table"
        cursor = artifacts
        for part in group.split("."):
            cursor = cursor.setdefault(part, {})
        cursor[key] = str(Path(value).resolve())
        return self.save()

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
    artifacts: Mapping[str, Mapping[str, Path | str]] | None = None,
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
            cursor = manifest_artifacts
            for part in str(group).split("."):
                cursor = cursor.setdefault(part, {})
            for key, value in entries.items():
                cursor[str(key)] = str(Path(value).resolve())
    return manifest.save()

