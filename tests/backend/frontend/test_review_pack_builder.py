from __future__ import annotations

from pathlib import Path

import pytest

from backend.frontend.review_pack_builder import build_review_packs
from backend.pipeline.runs import RUNS_ROOT_ENV, RunManifest


def test_build_review_packs_delegates_to_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "runs"
    sid = "SID-700"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))

    manifest = RunManifest.for_sid(sid)

    calls: list[dict[str, object]] = []

    def _fake_generate(
        sid_value: str, *, runs_root: Path | None = None, force: bool = False
    ) -> dict[str, object]:
        calls.append({"sid": sid_value, "runs_root": runs_root, "force": force})
        return {"status": "success", "packs_count": 3}

    monkeypatch.setattr(
        "backend.frontend.review_pack_builder.generate_frontend_packs_for_run",
        _fake_generate,
    )

    result = build_review_packs(sid, manifest)

    assert result == {"status": "success", "packs_count": 3}
    assert calls == [
        {"sid": sid, "runs_root": runs_root.resolve(), "force": True},
    ]
