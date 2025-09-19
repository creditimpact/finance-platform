from backend.core.logic.report_analysis.account_merge import gen_unordered_pairs


def test_gen_unordered_pairs_sorted_unique_pairs() -> None:
    indices = [16, 11, 8, 11, 12]

    result = gen_unordered_pairs(indices)

    assert result == [
        (8, 11),
        (8, 12),
        (8, 16),
        (11, 12),
        (11, 16),
        (12, 16),
    ]
