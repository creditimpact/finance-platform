from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.frontend.review_pack_builder import build_review_packs


class _ManifestStub:
    def __init__(self, path: Path) -> None:
        self.path = path


def test_build_review_packs_delegates_to_generator(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    sid = "SID-700"

    manifest_path = runs_root / sid / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.touch()
    manifest = _ManifestStub(manifest_path)

    calls: list[dict[str, object]] = []

    def _fake_generate(
        sid_value: str, *, runs_root: Path | None = None, force: bool = False
    ) -> dict[str, object]:
        calls.append({"sid": sid_value, "runs_root": runs_root, "force": force})
        return {"status": "success", "packs_count": 3}

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "backend.frontend.review_pack_builder.generate_frontend_packs_for_run",
            _fake_generate,
        )
        result = build_review_packs(sid, manifest)

    assert result == {"status": "success", "packs_count": 3}
    assert calls == [{"sid": sid, "runs_root": runs_root.resolve(), "force": True}]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_review_packs_generates_stage_packs(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    sid = "SID-701"

    account_dir = runs_root / sid / "cases" / "accounts" / "idx-001"
    summary_payload = {
        "account_id": "idx-001",
        "holder_name": "Case Holder",
        "labels": {
            "creditor": "Case Bank",
            "account_type": {"normalized": "Credit Card"},
            "status": {"normalized": "Open"},
        },
    }
    fields_flat_payload = {
        "per_bureau": {
            "transunion": {
                "account_number_display": "****4242",
                "account_status": "Open",
                "account_type": "Credit Card",
            },
            "experian": {
                "account_number_display": "XXXX4242",
                "account_status": "Open",
                "account_type": "Credit Card",
            },
        },
        "account_number_display": {
            "per_bureau": {
                "transunion": "****4242",
                "experian": "XXXX4242",
            }
        },
        "account_status": {
            "per_bureau": {"transunion": "Open", "experian": "Open"}
        },
        "account_type": {
            "per_bureau": {"transunion": "Credit Card", "experian": "Credit Card"}
        },
        "balance_owed": {"per_bureau": {"transunion": "$250"}},
        "date_opened": {"per_bureau": {"transunion": "2023-01-15"}},
        "holder_name": "Case Holder",
    }
    tags_payload = [{"kind": "issue", "type": "wrong_account"}]

    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "fields_flat.json", fields_flat_payload)
    _write_json(account_dir / "tags.json", tags_payload)

    manifest_path = runs_root / sid / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")
    manifest = _ManifestStub(manifest_path)

    result = build_review_packs(sid, manifest)

    assert result["packs_count"] == 1

    pack_path = runs_root / sid / "frontend" / "review" / "packs" / "idx-001.json"
    payload = json.loads(pack_path.read_text(encoding="utf-8"))

    assert payload["holder_name"] == "Case Holder"
    assert payload["primary_issue"] == "wrong_account"
    display_block = payload["display"]
    assert display_block["account_number"]["per_bureau"]["transunion"] == "****4242"
    assert display_block["account_type"]["per_bureau"]["experian"] == "Credit Card"
