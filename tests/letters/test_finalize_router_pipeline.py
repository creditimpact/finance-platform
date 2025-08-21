import json
import os
import re
from pathlib import Path

import pytest

from backend.analytics.analytics_tracker import get_counters, reset_counters
from backend.core.letters.client_context import format_safe_client_context
from backend.core.letters.router import select_template
from backend.core.logic.rendering.letter_rendering import render_dispute_letter_html
import planner
import tactical

BASE_CTX = {
    "client": {"full_name": "Jane Doe", "address_line": "123 Main St"},
    "today": "January 1, 2024",
    "account_number_masked": "1234",
    "bureau": "Experian",
}

POSITIVES = [
    Path("tests/letters/goldens/router_finalize_pay_for_delete_positive.json"),
    Path("tests/letters/goldens/router_finalize_mov_positive.json"),
    Path("tests/letters/goldens/router_finalize_debt_validation_positive.json"),
]

NEGATIVES = [
    Path("tests/letters/goldens/router_finalize_pay_for_delete_negative.json"),
    Path("tests/letters/goldens/router_finalize_mov_negative.json"),
    Path("tests/letters/goldens/router_finalize_debt_validation_negative.json"),
]


def _load_case(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.mark.parametrize("path", POSITIVES, ids=[p.stem for p in POSITIVES])
def test_finalize_router_positive(path: Path):
    case = _load_case(path)
    os.environ["LETTERS_ROUTER_PHASED"] = "1"
    reset_counters()

    ctx = BASE_CTX.copy()
    ctx.update(case["initial_ctx"])
    ctx["client_context_sentence"] = format_safe_client_context(case["action_tag"], "", {}, [])

    # Candidate phase
    select_template(case["action_tag"], ctx, phase="candidate")
    pre_counters = get_counters()
    assert pre_counters.get("router.candidate_selected") == 1
    assert pre_counters.get(f"router.candidate_selected.{case['action_tag']}") == 1

    def fake_plan(session, tags):
        assert get_counters().get("router.candidate_selected") == 1
        return tags

    def fake_generate(session, tags):
        assert get_counters().get("router.candidate_selected") == 1

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(planner, "plan_next_step", fake_plan)
    monkeypatch.setattr(tactical, "generate_letters", fake_generate)
    planner.plan_next_step({}, [case["action_tag"]])
    tactical.generate_letters({}, [case["action_tag"]])
    monkeypatch.undo()

    decision = select_template(case["action_tag"], ctx, phase="finalize")
    assert decision.template_path == case["template"]
    assert decision.missing_fields == []

    artifact = render_dispute_letter_html(ctx, decision.template_path)
    html = re.sub(r"\s+", " ", artifact.html).strip()
    expected = re.sub(r"\s+", " ", Path(case["golden"]).read_text()).strip()
    assert html == expected

    counters = get_counters()
    tag = case["action_tag"]
    template = case["template"]
    assert counters.get("router.finalized") == 1
    assert counters.get(f"router.finalized.{tag}") == 1
    assert counters.get(f"router.finalized.{tag}.{template}") == 1
    assert counters.get(f"sanitizer.applied.{template}") == 1


@pytest.mark.parametrize("path", NEGATIVES, ids=[p.stem for p in NEGATIVES])
def test_finalize_router_negative(path: Path):
    case = _load_case(path)
    os.environ["LETTERS_ROUTER_PHASED"] = "1"
    reset_counters()

    ctx = BASE_CTX.copy()
    ctx.update(case["initial_ctx"])
    ctx["client_context_sentence"] = format_safe_client_context(case["action_tag"], "", {}, [])

    # Candidate phase
    select_template(case["action_tag"], ctx, phase="candidate")
    pre = get_counters()
    assert pre.get("router.candidate_selected") == 1
    assert pre.get(f"router.candidate_selected.{case['action_tag']}") == 1
    assert any(
        k.startswith(f"router.missing_fields.{case['action_tag']}") for k in pre
    )

    def fake_plan(session, tags):
        assert get_counters().get("router.candidate_selected") == 1
        return tags

    def fake_generate(session, tags):
        assert get_counters().get("router.candidate_selected") == 1

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(planner, "plan_next_step", fake_plan)
    monkeypatch.setattr(tactical, "generate_letters", fake_generate)
    planner.plan_next_step({}, [case["action_tag"]])
    tactical.generate_letters({}, [case["action_tag"]])
    monkeypatch.undo()

    decision = select_template(case["action_tag"], ctx, phase="finalize")
    assert decision.template_path == case["template"]
    assert decision.missing_fields == case["missing_fields"]

    counters = get_counters()
    tag = case["action_tag"]
    assert counters.get("router.finalized") == 1
    assert counters.get(f"router.finalized.{tag}") == 1
    assert counters.get(f"router.finalized.{tag}.{case['template']}") == 1
    assert not any(k.startswith("sanitizer.applied") for k in counters)
