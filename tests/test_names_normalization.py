from backend.core.logic.utils.names_normalization import normalize_creditor_name


def test_new_creditor_aliases():
    cases = {
        "Amex": "american express",
        "American Express National Bank": "american express",
        "CBNA": "citibank",
        "SYNCB/Amazon": "synchrony bank",
        "Fifth Third": "fifth third bank",
        "Santander Consumer": "santander bank",
        "BBVA Compass": "bbva usa",
        "One Main": "onemain financial",
        "Lending Club": "lendingclub",
        "Freedom Financial": "freedomplus",
    }
    for raw, expected in cases.items():
        assert normalize_creditor_name(raw) == expected
