from typing import Tuple, List, Any
from backend.core.services.ai_client import AIClient
from backend.core.models.letter import LetterContext

from backend.core.logic.rule_checker import check_letter, RuleViolation
from backend.core.logic.rules_loader import load_rules
from backend.api.session_manager import get_session, update_session


def _build_system_prompt() -> str:
    rules = load_rules()
    lines = [
        "You are a credit dispute letter generator. Follow the systemic rules provided:",
    ]
    for rule in rules:
        desc = rule.get("description")
        if desc:
            lines.append(f"- {desc}")
    lines.append("Use only neutral, factual language.")
    return "\n".join(lines)


SYSTEM_PROMPT = _build_system_prompt()


def _record_letter(
    session_id: str,
    letter_type: str,
    text: str,
    violations: List[RuleViolation],
    iterations: int,
) -> None:
    session = get_session(session_id) or {}
    letters = session.get("letters_generated", [])
    letters.append(
        {
            "type": letter_type,
            "text": text,
            "violations": violations,
            "iterations": iterations,
        }
    )
    update_session(session_id, letters_generated=letters)


def generate_letter_with_guardrails(
    user_prompt: str,
    state: str | None,
    context: LetterContext | dict[str, Any],
    session_id: str,
    letter_type: str,
    ai_client: AIClient,
) -> Tuple[str, List[RuleViolation], int]:
    """Generate a letter via LLM and ensure compliance with rule checker."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    iterations = 0
    text = ""
    violations: List[RuleViolation] = []
    while iterations < 2:
        iterations += 1
        response = ai_client.chat_completion(
            messages=messages,
            temperature=0.3,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.replace("```", "").strip()
        text, violations = check_letter(text, state, context)
        critical = [v for v in violations if v["severity"] == "critical"]
        if not critical or iterations >= 2:
            break
        rule_list = ", ".join(v["rule_id"] for v in critical)
        messages.append({"role": "assistant", "content": text})
        messages.append(
            {
                "role": "user",
                "content": f"The draft contains violations of {rule_list}. Please fix them and return a compliant version.",
            }
        )
    _record_letter(session_id, letter_type, text, violations, iterations)
    return text, violations, iterations


def fix_draft_with_guardrails(
    draft_text: str,
    state: str | None,
    context: LetterContext | dict[str, Any],
    session_id: str,
    letter_type: str,
    ai_client: AIClient,
) -> Tuple[str, List[RuleViolation], int]:
    """Check and optionally repair an existing draft letter."""

    text, violations = check_letter(draft_text, state, context)
    iterations = 1
    critical = [v for v in violations if v["severity"] == "critical"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": text},
    ]
    while critical and iterations < 2:
        rule_list = ", ".join(v["rule_id"] for v in critical)
        messages.append(
            {
                "role": "user",
                "content": f"The draft contains violations of {rule_list}. Please fix them and return a compliant version.",
            }
        )
        response = ai_client.chat_completion(
            messages=messages,
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.replace("```", "").strip()
        text, violations = check_letter(text, state, context)
        iterations += 1
        critical = [v for v in violations if v["severity"] == "critical"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": text},
        ]
    _record_letter(session_id, letter_type, text, violations, iterations)
    return text, violations, iterations
