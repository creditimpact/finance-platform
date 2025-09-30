from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.compact_merge_summaries import main


@pytest.mark.usefixtures("tmp_path")
def test_compact_merge_summaries_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    account_dir = tmp_path / "runs" / "SID123" / "cases" / "accounts" / "0001"
    account_dir.mkdir(parents=True)
    summary_path = account_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "merge_scoring": {
                    "best_with": "2",
                    "matched_fields": {"foo": 1, "bar": 0},
                    "aux": {"noise": True},
                },
                "merge_explanations": [
                    {
                        "kind": "merge_pair",
                        "with": "3",
                        "parts": {"match": "4"},
                        "matched_fields": {"baz": []},
                        "reasons": ("keep",),
                        "aux": {"noise": 1},
                    }
                ],
                "aux": {"nested": 1},
            }
        )
    )

    monkeypatch.chdir(tmp_path)

    exit_code = main(["SID123"])

    assert exit_code == 0

    data = json.loads(summary_path.read_text())
    assert "aux" not in data
    assert data["merge_scoring"] == {"best_with": 2, "matched_fields": {"foo": True, "bar": False}}
    assert data["merge_explanations"] == [
        {
            "kind": "merge_pair",
            "with": 3,
            "parts": {"match": 4},
            "matched_fields": {"baz": False},
            "reasons": ["keep"],
        }
    ]
