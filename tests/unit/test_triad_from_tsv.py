import json
import os
import subprocess
from pathlib import Path


def create_triad_tsv(path: Path) -> None:
    header = "page\tline\ty0\ty1\tx0\tx1\ttext\n"
    rows = [
        # page 1 header row
        "1\t1\t10\t11\t50\t100\tTransUnion\n",
        "1\t1\t10\t11\t150\t200\tExperian\n",
        "1\t1\t10\t11\t250\t300\tEquifax\n",
        # Account number row
        "1\t2\t20\t21\t0\t20\tAccount #\n",
        "1\t2\t20\t21\t60\t80\tTU12345\n",
        "1\t2\t20\t21\t160\t180\tXP12345\n",
        "1\t2\t20\t21\t260\t280\tEQ12345\n",
        # High Balance row
        "1\t3\t30\t31\t0\t20\tHigh Balance:\n",
        "1\t3\t30\t31\t60\t80\t1000\n",
        "1\t3\t30\t31\t160\t180\t2000\n",
        "1\t3\t30\t31\t260\t280\t3000\n",
        # Payment Status row
        "1\t4\t40\t41\t0\t20\tPayment Status:\n",
        "1\t4\t40\t41\t60\t80\tCurrent\n",
        "1\t4\t40\t41\t160\t180\tCurrent\n",
        "1\t4\t40\t41\t260\t280\tCurrent\n",
        # Creditor Remarks row
        "1\t5\t50\t51\t0\t20\tCreditor Remarks:\n",
        "1\t5\t50\t51\t260\t270\tFannie\n",
        "1\t5\t50\t51\t270\t280\tMae\n",
        # Continuation in EQ only
        "1\t6\t60\t61\t260\t280\taccount\n",
        # Dummy label to close open row before page break
        "1\t7\t70\t71\t0\t20\tEnd:\n",
        # page 2 header row
        "2\t1\t10\t11\t50\t100\tTransUnion\n",
        "2\t1\t10\t11\t150\t200\tExperian\n",
        "2\t1\t10\t11\t250\t300\tEquifax\n",
        # Account Type row
        "2\t2\t20\t21\t0\t20\tAccount Type:\n",
        "2\t2\t20\t21\t60\t80\tMortgage\n",
        "2\t2\t20\t21\t160\t180\tMortgage\n",
        "2\t2\t20\t21\t260\t280\tMortgage\n",
        # Payment Frequency row
        "2\t3\t30\t31\t0\t20\tPayment Frequency:\n",
        "2\t3\t30\t31\t60\t80\tMonthly\n",
        "2\t3\t30\t31\t160\t180\tMonthly\n",
        "2\t3\t30\t31\t260\t280\tMonthly\n",
        # Credit Limit row
        "2\t4\t40\t41\t0\t20\tCredit Limit:\n",
        "2\t4\t40\t41\t60\t80\t5000\n",
        "2\t4\t40\t41\t160\t180\t4000\n",
        "2\t4\t40\t41\t260\t280\t3000\n",
        # Two-Year Payment History row (should stop parsing before this)
        "2\t5\t50\t51\t0\t20\tTwo-Year Payment History:\n",
        "2\t5\t50\t51\t60\t80\tX\n",
        "2\t5\t50\t51\t160\t180\tY\n",
        "2\t5\t50\t51\t260\t280\tZ\n",
    ]
    path.write_text(header + "".join(rows), encoding="utf-8")


def test_triad_from_tsv(tmp_path: Path) -> None:
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    create_triad_tsv(tsv_path)

    env = os.environ.copy()
    env["RAW_TRIAD_FROM_X"] = "1"
    env["RAW_JOIN_TOKENS_WITH_SPACE"] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    subprocess.run(
        [
            "python",
            "scripts/split_accounts_from_tsv.py",
            "--full",
            str(tsv_path),
            "--json_out",
            str(json_path),
        ],
        check=True,
        env=env,
        cwd=Path(__file__).resolve().parents[2],
    )

    data = json.loads(json_path.read_text())
    acc = data["accounts"][0]
    assert acc["triad_fields"]["transunion"]["account_number"]
    assert acc["triad_fields"]["equifax"]["creditor_remarks"].endswith("Fannie Mae account")
    labels = [r["label"].lower() for r in acc["triad_rows"]]
    assert "two-year payment history" not in labels
