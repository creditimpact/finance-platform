from pathlib import Path

import pytest

from backend.pipeline.runs import RUNS_ROOT_ENV
from tests.test_split_accounts_from_tsv import _run_split


@pytest.fixture(autouse=True)
def _runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))


def _write_triad_tail_row(tsv_path: Path, tail_text: str) -> None:
    header = "page\tline\ty0\ty1\tx0\tx1\ttext\n"
    rows = [
        "1\t1\t10\t11\t60\t100\tTransUnion\n",
        "1\t1\t10\t11\t160\t200\tExperian\n",
        "1\t1\t10\t11\t260\t300\tEquifax\n",
        "1\t2\t20\t21\t0\t40\tAccount #\n",
        "1\t2\t20\t21\t60\t100\t123456789\n",
        "1\t2\t20\t21\t160\t200\t123456789\n",
        "1\t2\t20\t21\t260\t300\t123456789\n",
        "1\t3\t30\t31\t0\t40\tDate Opened:\n",
        f"1\t3\t30\t31\t60\t320\t{tail_text}\n",
    ]
    tsv_path.write_text(header + "".join(rows), encoding="utf-8")


def _write_triad_continuation_row(tsv_path: Path, continuation_text: str) -> None:
    header = "page\tline\ty0\ty1\tx0\tx1\ttext\n"
    rows = [
        "1\t1\t10\t11\t60\t100\tTransUnion\n",
        "1\t1\t10\t11\t160\t200\tExperian\n",
        "1\t1\t10\t11\t260\t300\tEquifax\n",
        "1\t2\t20\t21\t0\t40\tAccount #\n",
        "1\t2\t20\t21\t60\t100\t123456789\n",
        "1\t2\t20\t21\t160\t200\t123456789\n",
        "1\t2\t20\t21\t260\t300\t123456789\n",
        "1\t3\t30\t31\t0\t40\tPast Due Amount:\n",
        "1\t3\t30\t31\t60\t100\t100\n",
        "1\t3\t30\t31\t160\t200\t200\n",
        "1\t3\t30\t31\t260\t300\t300\n",
        f"1\t4\t40\t41\t60\t320\t{continuation_text}\n",
    ]
    tsv_path.write_text(header + "".join(rows), encoding="utf-8")


@pytest.mark.parametrize(
    "tail_text, expected",
    [
        ("30.3.2024 1.6.2025 1.2.2025", ("30.3.2024", "1.6.2025", "1.2.2025")),
        ("30.3.2024  1.6.2025  1.2.2025", ("30.3.2024", "1.6.2025", "1.2.2025")),
        ("JAN 2024 FEB 2025 MAR 2026", ("JAN 2024", "FEB 2025", "MAR 2026")),
        ("30.3.2024 -- 1.6.2025 -- 1.2.2025", ("30.3.2024", "1.6.2025", "1.2.2025")),
        ("-- -- --", ("--", "--", "--")),
    ],
)
def test_triad_space_delimited_tail_split(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, tail_text: str, expected: tuple[str, str, str]
) -> None:
    tsv_path = tmp_path / "_space_tail.tsv"
    _write_triad_tail_row(tsv_path, tail_text)

    data, _accounts_dir, _sid = _run_split(tsv_path, caplog)
    fields = data["accounts"][0]["triad_fields"]

    assert (
        fields["transunion"]["date_opened"],
        fields["experian"]["date_opened"],
        fields["equifax"]["date_opened"],
    ) == expected
    assert "TRIAD_TAIL_SPACE_SPLIT" in caplog.text

    for bureau in ("transunion", "experian", "equifax"):
        assert fields[bureau]["high_balance"] == "--"

    assert fields["transunion"]["account_number_display"] == "123456789"


def test_triad_space_delimited_continuation_split(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tsv_path = tmp_path / "_space_continuation.tsv"
    _write_triad_continuation_row(tsv_path, "400 500 600")

    monkeypatch.setenv("TRIAD_BAND_BY_X0", "1")

    data, _accounts_dir, _sid = _run_split(tsv_path, caplog)
    fields = data["accounts"][0]["triad_fields"]

    assert fields["transunion"]["past_due_amount"] == "100 400"
    assert fields["experian"]["past_due_amount"] == "200 500"
    assert fields["equifax"]["past_due_amount"] == "300 600"
