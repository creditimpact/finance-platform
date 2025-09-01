import json
import shutil
from pathlib import Path

from backend.core.logic.report_analysis.report_parsing import parse_account_block


FULL_BLOCK = """Field: TransUnion Experian Equifax
Account Number: 12345 23456 34567
High Balance: 100 200 300
Last Verified: 2023-01 2023-01 2023-01
Date of Last Activity: 2022-12 2022-11 2022-10
Date Reported: 2023-01 2023-02 2023-03
Date Opened: 2020-01 2020-02 2020-03
Balance Owed: 0 100 200
Closed Date: 2024-01 2024-02 2024-03
Account Rating: A B C
Account Description: Revolving Revolving Installment
Dispute Status: None None Disputed
Creditor Type: Bank Bank Bank
Account Status: Open Open Closed
Payment Status: Current Current Late
Creditor Remarks: N/A N/A N/A
Payment Amount: 10 20 30
Last Payment: 2023-01 2023-02 2023-03
Term Length: 12 24 36
Past Due Amount: 0 0 0
Account Type: Credit Card Loan Mortgage
Payment Frequency: Monthly Monthly Monthly
Credit Limit: 1000 2000 3000
Two-Year Payment History: 111111 222222 333333
Days Late - 7 Year History: 0/0/0 0/0/0 0/0/0
"""


def test_parsed_triples_debug(tmp_path):
    sid = "debugsid"
    acc_id = "acc1"
    trace_dir = Path("traces") / sid
    if trace_dir.exists():
        shutil.rmtree(trace_dir)

    lines = [ln for ln in FULL_BLOCK.strip().split("\n")]
    parse_account_block(lines, heading="Sample", sid=sid, account_id=acc_id)

    dbg_path = Path("traces") / sid / "parsed_triples" / f"{acc_id}.json"
    assert dbg_path.exists()
    data = json.loads(dbg_path.read_text())
    assert data["bureau_order"] == ["transunion", "experian", "equifax"]
    assert len(data["rows"]) >= 10

    shutil.rmtree(trace_dir)
