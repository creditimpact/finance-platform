from backend.core.logic.report_analysis.problem_detection import evaluate_account_problem




def test_clean_account_not_flagged():
    clean = {"account_status": "Open", "payment_status": "Pays as agreed"}
