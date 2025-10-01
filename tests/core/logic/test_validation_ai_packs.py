from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.core.ai.paths import (
    ensure_validation_account_paths,
    ensure_validation_paths,
)
from backend.core.logic import validation_ai_packs
from backend.core.logic.validation_ai_packs import build_validation_ai_packs_for_accounts
from backend.pipeline.runs import RUNS_ROOT_ENV


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_builder_creates_validation_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "sid-validation"
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))

    build_validation_ai_packs_for_accounts(
        sid,
        account_indices=[14, "15", "14"],
        runs_root=runs_root,
    )

    validation_paths = ensure_validation_paths(runs_root, sid, create=False)
    base_dir = validation_paths.base

    created_indices = {"14", "15"}
    for idx in created_indices:
        account_paths = ensure_validation_account_paths(
            validation_paths, idx, create=False
        )
        assert account_paths.pack_file.exists()
        assert account_paths.prompt_file.exists()
        assert account_paths.model_results_file.exists()

        pack_payload = json.loads(_read(account_paths.pack_file))
        assert pack_payload == {"weak_items": []}
        assert json.loads(_read(account_paths.model_results_file)) == {}
        assert _read(account_paths.prompt_file) == ""

    manifest_path = runs_root / sid / "manifest.json"
    assert manifest_path.exists()

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_section = manifest_data["ai"]["validation"]
    assert validation_section["base"] == str(base_dir.resolve())
    assert validation_section["dir"] == str(base_dir.resolve())
    assert validation_section["accounts"] == str(base_dir.resolve())
    assert validation_section["accounts_dir"] == str(base_dir.resolve())
    assert isinstance(validation_section["last_prepared_at"], str)

    packs_validation = manifest_data["ai"]["packs"]["validation"]
    assert packs_validation["base"] == str(base_dir.resolve())
    packs_path = Path(packs_validation["packs"])
    assert packs_path.parent == base_dir.resolve()
    assert packs_path.name in created_indices
    expected_results = (packs_path / "results").resolve()
    assert packs_validation["results"] == str(expected_results)
    expected_index = (base_dir / "index.json").resolve()
    assert packs_validation["index"] == str(expected_index)
    assert isinstance(packs_validation["last_built_at"], str)


def test_builder_populates_pack_and_preserves_prompt_and_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "sid-existing"
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))

    validation_paths = ensure_validation_paths(runs_root, sid, create=True)
    account_paths = ensure_validation_account_paths(
        validation_paths, 42, create=True
    )

    account_paths.pack_file.write_text("{\"preseed\": true}\n", encoding="utf-8")
    account_paths.prompt_file.write_text("Existing prompt", encoding="utf-8")
    account_paths.model_results_file.write_text(
        "{\"status\": \"done\"}\n", encoding="utf-8"
    )

    accounts_root = runs_root / sid / "cases" / "accounts"
    account_dir = accounts_root / "42"
    account_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "validation_requirements": {
            "requirements": [
                {
                    "field": "balance_owed",
                    "category": "activity",
                    "min_days": 30,
                    "documents": ["statement"],
                    "ai_needed": True,
                },
                {
                    "field": "payment_status",
                    "category": "status",
                    "min_days": 10,
                    "documents": [],
                    "ai_needed": False,
                },
            ],
            "field_consistency": {
                "balance_owed": {
                    "consensus": "split",
                    "disagreeing_bureaus": ["experian"],
                    "missing_bureaus": ["equifax"],
                    "raw": {
                        "transunion": "100",
                        "experian": "150",
                        "equifax": None,
                    },
                    "normalized": {
                        "transunion": 100,
                        "experian": 150,
                        "equifax": None,
                    },
                }
            },
        }
    }
    (account_dir / "summary.json").write_text(
        json.dumps(summary_payload), encoding="utf-8"
    )

    build_validation_ai_packs_for_accounts(
        sid,
        account_indices=[42],
        runs_root=runs_root,
    )

    pack_payload = json.loads(_read(account_paths.pack_file))
    assert pack_payload == {
        "weak_items": [
            {
                "field": "balance_owed",
                "category": "activity",
                "min_days": 30,
                "documents": ["statement"],
                "consensus": "split",
                "disagreeing_bureaus": ["experian"],
                "missing_bureaus": ["equifax"],
                "values": {
                    "transunion": {"raw": "100", "normalized": 100},
                    "experian": {"raw": "150", "normalized": 150},
                    "equifax": {"raw": None, "normalized": None},
                },
            }
        ]
    }
    expected_prompt = validation_ai_packs._render_prompt(
        sid, 42, pack_payload["weak_items"]
    )
    assert _read(account_paths.prompt_file) == expected_prompt
    assert _read(account_paths.model_results_file) == "{\"status\": \"done\"}\n"
