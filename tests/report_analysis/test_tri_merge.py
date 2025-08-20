from backend.core.logic.report_analysis.tri_merge import normalize_and_match, compute_mismatches
from backend.core.logic.report_analysis.tri_merge_models import Tradeline
from backend.analytics.analytics_tracker import reset_counters, get_counters


def test_fuzzy_creditor_matching():
    tls = [
        Tradeline(
            creditor="Citi",
            bureau="Experian",
            account_number="11114444",
            data={"date_opened": "2020-01-01", "date_reported": "2020-02-01"},
        ),
        Tradeline(
            creditor="CBNA",
            bureau="Equifax",
            account_number="22224444",
            data={"date_opened": "2020-01-01", "date_reported": "2020-02-01"},
        ),
        Tradeline(
            creditor="Citicard",
            bureau="TransUnion",
            account_number="33334444",
            data={"date_opened": "2020-01-01", "date_reported": "2020-02-01"},
        ),
    ]

    families = normalize_and_match(tls)
    assert len(families) == 1
    fam = families[0]
    assert set(fam.tradelines.keys()) == {"Experian", "Equifax", "TransUnion"}
    assert fam.match_confidence == 1.0


def test_presence_and_field_mismatches():
    tls = [
        Tradeline(
            creditor="Chase",
            bureau="Experian",
            account_number="12341234",
            data={
                "balance": 100,
                "status": "open",
                "date_opened": "2020-01-01",
                "remarks": "OK",
                "utilization": 0.1,
                "personal_info": "PI1",
                "date_reported": "2020-02-01",
            },
        ),
        Tradeline(
            creditor="Chase Bank",
            bureau="Equifax",
            account_number="00001234",
            data={
                "balance": 100,
                "status": "open",
                "date_opened": "2020-01-01",
                "remarks": "OK",
                "utilization": 0.1,
                "personal_info": "PI1",
                "date_reported": "2020-02-01",
            },
        ),
    ]

    families = normalize_and_match(tls)
    fam = families[0]
    fam.tradelines["Equifax"].data.update(
        {
            "balance": 200,
            "status": "closed",
            "date_reported": "2020-03-01",
            "remarks": "Late",
            "utilization": 0.5,
            "personal_info": "PI2",
        }
    )

    families = compute_mismatches(families)
    fam = families[0]
    mism = {m.field: m for m in fam.mismatches}

    assert mism["presence"].values == {
        "Experian": True,
        "Equifax": True,
        "TransUnion": False,
    }
    assert mism["balance"].values == {"Experian": 100, "Equifax": 200}
    assert mism["status"].values == {"Experian": "open", "Equifax": "closed"}
    assert mism["dates"].values == {
        "Experian": ("2020-01-01", "2020-02-01"),
        "Equifax": ("2020-01-01", "2020-03-01"),
    }
    assert mism["remarks"].values == {"Experian": "OK", "Equifax": "Late"}
    assert mism["utilization"].values == {"Experian": 0.1, "Equifax": 0.5}
    assert mism["personal_info"].values == {"Experian": "PI1", "Equifax": "PI2"}


def test_duplicate_mismatch_counts():
    tls = [
        Tradeline(
            creditor="Chase",
            bureau="Experian",
            account_number="12341234",
            data={"date_opened": "2020-01-01", "date_reported": "2020-02-01"},
        ),
        Tradeline(
            creditor="Chase",
            bureau="Experian",
            account_number="12341234",
            data={"date_opened": "2020-01-01", "date_reported": "2020-02-01"},
        ),
    ]

    families = normalize_and_match(tls)
    families = compute_mismatches(families)
    fam = families[0]
    mism = {m.field: m for m in fam.mismatches}

    assert mism["duplicate"].values == {"Experian": 1}


def test_match_confidence_p95_metric():
    tls = [
        Tradeline(
            creditor="Cred",
            bureau="Experian",
            account_number=f"{i:04d}",
            data={},
        )
        for i in range(19)
    ]
    tls.append(
        Tradeline(
            creditor="Cred",
            bureau="Experian",
            account_number="9999",
            data={"date_opened": "2020-01-01", "date_reported": "2020-02-01"},
        )
    )

    reset_counters()
    normalize_and_match(tls)
    counters = get_counters()
    assert counters["tri_merge.match_confidence_p95"] == 0.5
