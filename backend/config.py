import os

UTILIZATION_PROBLEM_THRESHOLD = float(os.getenv("UTILIZATION_PROBLEM_THRESHOLD", "0.90"))
SERIOUS_DELINQUENCY_MIN_DPD = int(os.getenv("SERIOUS_DELINQUENCY_MIN_DPD", "60"))

TIER1_KEYWORDS = {
    "bankruptcy": ["bankruptcy"],
    "foreclosure": ["foreclosure"],
    "judgment": ["judgment"],
    "tax_lien": ["tax lien", "tax-lien"],
    "charge_off": ["charge off", "charge-off", "charged off", "chargeoff"],
    "collection": ["collection", "placed for collection", "collections"],
}

TIER2_KEYWORDS = {
    "serious_delinquency": [
        "60 days past due",
        "90 days past due",
        "120 days past due",
        "120+ days",
        "derogatory",
    ]
}

TIER3_KEYWORDS = {
    "potential_derogatory": ["derogatory", "collection", "charge-off"]
}
