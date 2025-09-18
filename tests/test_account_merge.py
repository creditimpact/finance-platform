import copy
import json
import logging

import pytest

from backend.core.logic.report_analysis.account_merge import (
    _apply_account_number_ai_override,
    build_ai_decision_pack,
    cluster_problematic_accounts,
    decide_merge,
    score_accounts,
)


@pytest.fixture
def identical_account():
    return {
        "account_number": "1234567890",
        "date_opened": "01-01-2020",
        "date_of_last_activity": "05-01-2021",
        "closed_date": "05-02-2021",
        "balance_owed": "1500",
        "past_due_amount": "300",
        "payment_status": "Charge Off",
        "account_status": "Charge Off",
        "creditor": "Example Bank",
        "remarks": "Original creditor account",
    }


def test_identical_accounts_auto_cluster(identical_account):
    account_a = copy.deepcopy(identical_account)
    account_b = copy.deepcopy(identical_account)

    score, parts, aux = score_accounts(account_a, account_b)

    assert score >= 0.9
    for value in parts.values():
        assert value == pytest.approx(1.0)
    assert aux["acctnum_level"] == "exact"
    assert aux["acctnum_masked_any"] is False

    assert decide_merge(score) == "auto"

    merged = cluster_problematic_accounts([account_a, account_b], sid="identical-run")

    first_tag = merged[0]["merge_tag"]
    second_tag = merged[1]["merge_tag"]

    assert first_tag["decision"] == "auto"
    assert second_tag["decision"] == "auto"
    assert first_tag["group_id"] == second_tag["group_id"]
    assert first_tag["best_match"]["account_index"] == 1
    assert second_tag["best_match"]["account_index"] == 0
    assert first_tag["best_match"]["decision"] == "auto"
    assert second_tag["best_match"]["decision"] == "auto"
    assert first_tag["aux"]["acctnum_level"] == "exact"


def test_unrelated_accounts_different_decision():
    account_a = {
        "account_number": "11112222",
        "date_opened": "01-01-2020",
        "balance_owed": "5000",
        "payment_status": "Charge Off",
        "creditor": "Acme Bank",
    }
    account_b = {
        "account_number": "99998888",
        "date_opened": "01-01-2010",
        "balance_owed": "0",
        "payment_status": "Current",
        "creditor": "Different Creditor",
    }

    score, _, aux = score_accounts(account_a, account_b)
    assert score < 0.2
    assert decide_merge(score) == "different"
    assert aux["acctnum_level"] == "none"
    assert aux["acctnum_masked_any"] is False

    merged = cluster_problematic_accounts(
        [copy.deepcopy(account_a), copy.deepcopy(account_b)], sid="unrelated"
    )

    first_tag = merged[0]["merge_tag"]
    second_tag = merged[1]["merge_tag"]

    assert first_tag["decision"] == "different"
    assert second_tag["decision"] == "different"
    assert first_tag["group_id"] != second_tag["group_id"]


def test_partial_overlap_ai_decision():
    account_a = {
        "account_number": "1234567812345678",
        "date_opened": "01-01-2020",
        "balance_owed": "5000",
        "payment_status": "Charge Off",
        "creditor": "Acme Bank",
    }
    account_b = {
        "account_number": "000012345678",
        "date_opened": "01-01-2020",
        "balance_owed": 5000,
        "payment_status": "Current",
        "creditor": "Acme Collections",
    }

    score, parts, aux = score_accounts(account_a, account_b)

    assert 0.35 <= score < 0.78
    assert parts["acct"] == pytest.approx(0.7)
    assert parts["dates"] == pytest.approx(1.0)
    assert parts["balowed"] == pytest.approx(1.0)
    assert parts["status"] == pytest.approx(0.0)
    assert parts["strings"] == pytest.approx(0.48)
    assert aux["acctnum_level"] == "last4"
    assert aux["acctnum_masked_any"] is False

    assert decide_merge(score) == "ai"

    merged = cluster_problematic_accounts(
        [copy.deepcopy(account_a), copy.deepcopy(account_b)], sid="partial"
    )

    first_tag = merged[0]["merge_tag"]
    second_tag = merged[1]["merge_tag"]

    assert first_tag["decision"] == "ai"
    assert second_tag["decision"] == "ai"
    assert first_tag["best_match"]["decision"] == "ai"
    assert second_tag["best_match"]["decision"] == "ai"


def test_graph_clustering_transitive_auto():
    account_a = {
        "account_number": "1111 2222 3333 4444",
        "date_opened": "01-01-2020",
        "balance_owed": 1000,
        "payment_status": "Charge Off account",
        "creditor": "Acme Bank",
    }
    account_b = {
        "account_number": "9999000011114444",
        "date_opened": "01-01-2020",
        "date_of_last_activity": "05-01-2021",
        "balance_owed": 1000,
        "past_due_amount": 500,
        "payment_status": "Charge Off account now paid as agreed",
        "creditor": "Acme Bank Collections Dept",
    }
    account_c = {
        "account_number": "9999000011114444",
        "date_of_last_activity": "05-01-2021",
        "past_due_amount": 500,
        "payment_status": "Paid as agreed",
        "creditor": "Collections Department",
    }

    score_ab, _, aux_ab = score_accounts(account_a, account_b)
    score_bc, _, aux_bc = score_accounts(account_b, account_c)
    score_ac, _, aux_ac = score_accounts(account_a, account_c)

    assert score_ab >= 0.78
    assert score_bc >= 0.78
    assert score_ac < 0.35
    assert aux_ab["acctnum_level"] == "last4"
    assert aux_bc["acctnum_level"] == "exact"
    assert aux_ac["acctnum_level"] == "last4"

    merged = cluster_problematic_accounts(
        [copy.deepcopy(account_a), copy.deepcopy(account_b), copy.deepcopy(account_c)],
        sid="graph",
    )

    tags = [item["merge_tag"] for item in merged]
    group_ids = {tag["group_id"] for tag in tags}
    assert group_ids == {"g1"}
    assert all(tag["decision"] == "auto" for tag in tags)
    assert tags[0]["best_match"]["account_index"] == 1
    assert tags[1]["best_match"]["account_index"] == 2
    assert tags[2]["best_match"]["account_index"] == 1

    assert [entry["account_index"] for entry in tags[0]["score_to"]] == [1]
    assert {entry["account_index"] for entry in tags[1]["score_to"]} == {0, 2}
    assert [entry["account_index"] for entry in tags[2]["score_to"]] == [1]


def test_cluster_problematic_accounts_logs_merge_events(caplog):
    accounts = [
        {
            "account_number": "1111 2222 3333 4444",
            "date_opened": "01-01-2020",
            "balance_owed": 1000,
            "payment_status": "Charge Off account",
            "creditor": "Acme Bank",
        },
        {
            "account_number": "9999000011114444",
            "date_opened": "01-01-2020",
            "date_of_last_activity": "05-01-2021",
            "balance_owed": 1000,
            "past_due_amount": 500,
            "payment_status": "Charge Off account now paid as agreed",
            "creditor": "Acme Bank Collections Dept",
        },
        {
            "account_number": "9999000011114444",
            "date_of_last_activity": "05-01-2021",
            "past_due_amount": 500,
            "payment_status": "Paid as agreed",
            "creditor": "Collections Department",
        },
    ]

    with caplog.at_level(logging.INFO, logger="backend.core.logic.report_analysis.account_merge"):
        cluster_problematic_accounts(accounts, sid="log-test")

    merge_messages = [record.getMessage() for record in caplog.records]
    assert any("MERGE_DECISION" in msg for msg in merge_messages)
    summary_messages = [msg for msg in merge_messages if "MERGE_SUMMARY" in msg]
    assert summary_messages
    assert "skipped_pairs=<1>" in summary_messages[-1]


def test_scoring_handles_missing_fields():
    account_a = {
        "acct_num": "--",
        "date_opened": "--",
        "date_of_last_activity": "05-01-2020",
        "balance_owed": "--",
        "past_due_amount": "50",
        "account_status": "--",
        "creditor": "Sample Creditor",
        "remarks": None,
    }
    account_b = {
        "account_number": None,
        "date_opened": "05-01-2020",
        "date_of_last_activity": "--",
        "balance_owed": 0,
        "past_due_amount": "50.00 USD",
        "account_status": "Charge-Off",
        "creditor": "Sample Creditor LLC",
        "remarks": "--",
    }

    score_first, parts_first, aux_first = score_accounts(account_a, account_b)
    score_second, parts_second, aux_second = score_accounts(account_a, account_b)

    assert score_first == pytest.approx(score_second)
    assert parts_first == parts_second
    assert aux_first == aux_second
    assert 0 < score_first < 0.35
    assert parts_first["acct"] == 0.0
    assert parts_first["dates"] == 0.0
    assert parts_first["balowed"] == pytest.approx(1.0)
    assert parts_first["strings"] > 0.5
    assert aux_first["acctnum_level"] == "none"
    assert aux_first["acctnum_masked_any"] is False

    merged = cluster_problematic_accounts(
        [copy.deepcopy(account_a), copy.deepcopy(account_b)], sid="missing"
    )

    first_tag = merged[0]["merge_tag"]
    second_tag = merged[1]["merge_tag"]

    assert first_tag["best_match"]["score"] == pytest.approx(score_first)
    assert second_tag["best_match"]["score"] == pytest.approx(score_first)
    assert decide_merge(score_first) == "different"


def _extract_override_log_messages(records):
    return [
        record.getMessage()
        for record in records
        if "MERGE_OVERRIDE" in record.getMessage()
    ]


def test_acctnum_exact_override_forces_ai(monkeypatch, caplog):
    monkeypatch.setenv("MERGE_ACCTNUM_TRIGGER_AI", "exact")
    monkeypatch.setenv("MERGE_ACCTNUM_MIN_SCORE", "0.45")
    monkeypatch.setenv("MERGE_ACCTNUM_REQUIRE_MASKED", "0")

    account_a = {"account_number": "1234567890"}
    account_b = {"account_number": "1234567890"}

    base_score, parts, aux = score_accounts(account_a, account_b)
    assert base_score == pytest.approx(0.25)
    assert parts["acct"] == pytest.approx(1.0)
    assert aux["acctnum_level"] == "exact"
    assert aux["acctnum_masked_any"] is False

    with caplog.at_level(
        logging.INFO, logger="backend.core.logic.report_analysis.account_merge"
    ):
        merged = cluster_problematic_accounts(
            [dict(account_a), dict(account_b)], sid="acctnum-exact"
        )

    first_tag = merged[0]["merge_tag"]
    best = first_tag["best_match"]

    assert first_tag["decision"] == "ai"
    assert best["decision"] == "ai"
    assert best["score"] == pytest.approx(0.45)
    assert first_tag["parts"]["acct"] == pytest.approx(1.0)
    assert first_tag["reasons"]["acctnum_only_triggers_ai"] is True
    assert first_tag["reasons"]["acctnum_match_level"] == "exact"
    assert best["reasons"]["acctnum_only_triggers_ai"] is True
    assert best["reasons"]["acctnum_match_level"] == "exact"
    assert first_tag["aux"]["override_reasons"]["acctnum_only_triggers_ai"] is True

    override_messages = _extract_override_log_messages(caplog.records)
    assert any(
        "MERGE_OVERRIDE sid=<acctnum-exact> i=<0> j=<1> kind=acctnum level=<exact>"
        " masked_any=<0> lifted_to=<0.4500>" in msg
        for msg in override_messages
    )


@pytest.mark.parametrize("trigger", ["any", "last4"])
def test_acctnum_last4_override_honors_trigger(monkeypatch, trigger):
    monkeypatch.setenv("MERGE_ACCTNUM_TRIGGER_AI", trigger)
    monkeypatch.setenv("MERGE_ACCTNUM_MIN_SCORE", "0.43")
    monkeypatch.setenv("MERGE_ACCTNUM_REQUIRE_MASKED", "0")

    account_a = {"account_number": "XXXX-1111-2222-5678"}
    account_b = {"account_number": "0000000000005678"}

    base_score, _, aux = score_accounts(account_a, account_b)
    assert base_score == pytest.approx(0.175)
    assert aux["acctnum_level"] == "last4"
    assert aux["acctnum_masked_any"] is True

    merged = cluster_problematic_accounts(
        [dict(account_a), dict(account_b)], sid=f"acctnum-last4-{trigger}"
    )

    first_tag = merged[0]["merge_tag"]
    best = first_tag["best_match"]

    assert first_tag["decision"] == "ai"
    assert best["decision"] == "ai"
    assert best["score"] == pytest.approx(0.43)
    assert best["reasons"]["acctnum_only_triggers_ai"] is True
    assert best["reasons"]["acctnum_match_level"] == "last4"
    assert first_tag["aux"]["override_reasons"]["acctnum_match_level"] == "last4"


def test_acctnum_override_respects_mask_requirement(monkeypatch):
    monkeypatch.setenv("MERGE_ACCTNUM_TRIGGER_AI", "exact")
    monkeypatch.setenv("MERGE_ACCTNUM_MIN_SCORE", "0.4")
    monkeypatch.setenv("MERGE_ACCTNUM_REQUIRE_MASKED", "1")

    account_a = {"account_number": "1234567890"}
    account_b = {"account_number": "1234567890"}

    base_score, _, aux = score_accounts(account_a, account_b)
    assert base_score == pytest.approx(0.25)
    assert aux["acctnum_masked_any"] is False

    merged = cluster_problematic_accounts(
        [dict(account_a), dict(account_b)], sid="acctnum-mask-required"
    )

    first_tag = merged[0]["merge_tag"]
    best = first_tag["best_match"]

    assert first_tag["decision"] == "different"
    assert best["decision"] == "different"
    assert best["score"] == pytest.approx(0.25)
    assert "reasons" not in best
    assert "override_reasons" not in first_tag.get("aux", {})


def test_acctnum_override_combines_with_existing_reasons(monkeypatch):
    monkeypatch.setenv("MERGE_ACCTNUM_TRIGGER_AI", "any")
    monkeypatch.setenv("MERGE_ACCTNUM_MIN_SCORE", "0.44")
    monkeypatch.setenv("MERGE_ACCTNUM_REQUIRE_MASKED", "0")

    thresholds = {"ai_band_min": 0.35, "ai_hard_min": 0.3}
    base_score = 0.2
    balance_lifted_score = 0.5
    existing_reasons = {"balance_only_triggers_ai": True}
    aux = {"acctnum_level": "exact", "acctnum_masked_any": False}

    lifted_score, decision, reasons, aux_out = _apply_account_number_ai_override(
        base_score,
        balance_lifted_score,
        "ai",
        aux,
        existing_reasons,
        thresholds,
        sid_value="acctnum-balance",
        idx_a=0,
        idx_b=1,
    )

    assert lifted_score == pytest.approx(0.5)
    assert decision == "ai"
    assert reasons["balance_only_triggers_ai"] is True
    assert reasons["acctnum_only_triggers_ai"] is True
    assert reasons["acctnum_match_level"] == "exact"
    override_reasons = aux_out["override_reasons"]
    assert override_reasons["balance_only_triggers_ai"] is True
    assert override_reasons["acctnum_only_triggers_ai"] is True


def test_acctnum_override_emits_logs_and_builds_ai_pack(monkeypatch, caplog, tmp_path):
    monkeypatch.setenv("MERGE_ACCTNUM_TRIGGER_AI", "exact")
    monkeypatch.setenv("MERGE_ACCTNUM_MIN_SCORE", "0.4")
    monkeypatch.setenv("MERGE_ACCTNUM_REQUIRE_MASKED", "0")

    account_a = {"account_index": 0, "account_number": "000012345678"}
    account_b = {"account_index": 1, "account_number": "000012345678"}

    with caplog.at_level(
        logging.INFO, logger="backend.core.logic.report_analysis.account_merge"
    ):
        merged = cluster_problematic_accounts(
            [dict(account_a), dict(account_b)], sid="acctnum-pack"
        )

    first_tag = merged[0]["merge_tag"]
    best = first_tag["best_match"]

    override_messages = _extract_override_log_messages(caplog.records)
    assert any(
        "MERGE_OVERRIDE sid=<acctnum-pack> i=<0> j=<1> kind=acctnum" in msg
        for msg in override_messages
    )

    pack = build_ai_decision_pack(
        account_a,
        account_b,
        score=best["score"],
        parts=first_tag["parts"],
        aux=first_tag["aux"],
        reasons=first_tag.get("reasons"),
    )

    pack_path = tmp_path / "ai_pack.json"
    pack_path.write_text(json.dumps(pack, indent=2))
    saved = json.loads(pack_path.read_text())

    assert saved["decision"] == "ai"
    assert saved["acctnum"] == {"level": "exact", "masked_any": False}
    assert saved["reasons"]["acctnum_only_triggers_ai"] is True
    assert pack_path.exists()


def test_cluster_uses_case_files_for_scoring(tmp_path):
    sid = "case-loader"
    runs_root = tmp_path / "runs"
    base_dir = runs_root / sid / "cases" / "accounts"
    for idx in (7, 10):
        account_dir = base_dir / str(idx)
        account_dir.mkdir(parents=True, exist_ok=True)
        fields_flat = {
            "balance_owed": 5912.0,
            "date_opened": "2020-01-01",
            "payment_status": "Charge Off",
            "creditor": "Case Bank",
        }
        bureaus = {
            "transunion": {"account_number": "0000 1234"},
            "experian": {},
            "equifax": {},
        }
        raw_lines = [{"text": "Account #0000 1234"}]
        (account_dir / "fields_flat.json").write_text(
            json.dumps(fields_flat), encoding="utf-8"
        )
        (account_dir / "bureaus.json").write_text(json.dumps(bureaus), encoding="utf-8")
        (account_dir / "raw_lines.json").write_text(json.dumps(raw_lines), encoding="utf-8")

    candidates = [
        {"account_index": 7},
        {"account_index": 10},
    ]

    merged = cluster_problematic_accounts(
        candidates,
        sid=sid,
        runs_root=runs_root,
    )

    parts = merged[0]["merge_tag"]["parts"]
    assert parts["balowed"] == pytest.approx(1.0)
    assert parts["acct"] == pytest.approx(1.0)
