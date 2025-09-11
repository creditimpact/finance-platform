import pytest
from backend.core.logic.report_analysis.block_exporter import join_tokens_with_space


@pytest.mark.parametrize("tokens,expected", [
    (["PaymentStatus:", "Collection/Chargeoff", "Collection/Chargeoff"],
     "PaymentStatus: Collection/Chargeoff Collection/Chargeoff"),
    (["HighBalance:", "$11,374", "$0", "$10,792"], "HighBalance: $11,374 $0 $10,792"),
    (["Transunion", "®", "Experian", "®", "Equifax", "®"], "Transunion ® Experian ® Equifax ®"),
    (["OK", "OK", "OK"], "OK OK OK"),
    (["עברית", "English", "123"], "עברית English 123"),
])
def test_join_tokens_with_space(tokens, expected):
    assert join_tokens_with_space(tokens) == expected
