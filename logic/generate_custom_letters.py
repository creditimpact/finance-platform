from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
import pdfkit
from logic.utils.pdf_ops import gather_supporting_docs
from .summary_classifier import classify_client_summary
from session_manager import get_session
from logic.guardrails import generate_letter_with_guardrails
from .rules_loader import get_neutral_phrase
from audit import AuditLogger, AuditLevel
from services.ai_client import AIClient
from config import get_app_config

env = Environment(loader=FileSystemLoader("templates"))
template = env.get_template("general_letter_template.html")


def _pdf_config(wkhtmltopdf_path: str | None):
    path = wkhtmltopdf_path or get_app_config().wkhtmltopdf_path
    return pdfkit.configuration(wkhtmltopdf=path)


def call_gpt_for_custom_letter(
    client_name: str,
    recipient_name: str,
    account_name: str,
    account_number: str,
    docs_text: str,
    structured_summary: dict,
    state: str,
    session_id: str,
    audit: AuditLogger | None,
    ai_client: AIClient,
) -> str:
    docs_line = f"Supporting documents summary:\n{docs_text}" if docs_text else ""
    classification = classify_client_summary(structured_summary, ai_client, state)
    neutral_phrase, neutral_reason = get_neutral_phrase(
        classification.get("category"), structured_summary
    )
    prompt = f"""
Neutral legal phrase for this dispute type:
"{neutral_phrase or ''}"

Here is what the client explained about this account (structured summary):
{json.dumps(structured_summary, indent=2)}
Classification: {json.dumps(classification)}
Client name: {client_name}
Recipient: {recipient_name}
State: {state}
Account: {account_name} {account_number}
{docs_line}
Please draft a compliant letter body that blends the neutral legal phrase with the client's explanation. Do not copy either source verbatim.
"""
    if audit:
        audit.log_account(
            structured_summary.get("account_id"),
            {
                "stage": "custom_letter",
                "classification": classification,
                "neutral_phrase": neutral_phrase,
                "neutral_phrase_reason": neutral_reason,
                "structured_summary": structured_summary,
            },
        )
    body, _, _ = generate_letter_with_guardrails(
        prompt,
        state,
        {
            "debt_type": structured_summary.get("debt_type"),
            "dispute_reason": classification.get("category"),
        },
        session_id,
        "custom",
        ai_client=ai_client,
    )
    if audit and audit.level is AuditLevel.VERBOSE:
        audit.log_step(
            "custom_letter_prompt",
            {
                "account_id": structured_summary.get("account_id"),
                "prompt": prompt,
            },
        )
        audit.log_step(
            "custom_letter_response",
            {
                "account_id": structured_summary.get("account_id"),
                "response": body,
            },
        )
    return body


def generate_custom_letter(
    account: dict,
    client_info: dict,
    output_path: Path,
    audit: AuditLogger | None,
    *,
    ai_client: AIClient,
    run_date: str | None = None,
    wkhtmltopdf_path: str | None = None,
) -> None:
    client_name = client_info.get("legal_name") or client_info.get("name", "Client")
    date_str = run_date or datetime.now().strftime("%B %d, %Y")
    recipient = account.get("name", "")
    acc_name = account.get("name", "")
    acc_number = account.get("account_number", "")
    session_id = client_info.get("session_id", "")
    state = client_info.get("state", "")

    session = get_session(session_id) or {}
    structured_summary = session.get("structured_summaries", {}).get(
        account.get("account_id"), {}
    )

    docs_text, doc_names, _ = gather_supporting_docs(session_id)
    if docs_text and audit and audit.level is AuditLevel.VERBOSE:
        print(f"[📎] Including supplemental docs for custom letter to {recipient}.")

    body_paragraph = call_gpt_for_custom_letter(
        client_name,
        recipient,
        acc_name,
        acc_number,
        docs_text,
        structured_summary,
        state,
        session_id,
        audit,
        ai_client,
    )

    greeting = f"Dear {recipient}" if recipient else "To whom it may concern"

    context = {
        "date": date_str,
        "client_name": client_name,
        "client_street": client_info.get("street", ""),
        "client_city": client_info.get("city", ""),
        "client_state": client_info.get("state", ""),
        "client_zip": client_info.get("zip", ""),
        "recipient_name": recipient,
        "greeting_line": greeting,
        "body_paragraph": body_paragraph,
        "supporting_docs": doc_names,
    }

    html = template.render(**context)
    safe_recipient = (recipient or "Recipient").replace("/", "_").replace("\\", "_")
    filename = f"Custom Letter - {safe_recipient}.pdf"
    full_path = output_path / filename
    options = {"quiet": ""}
    pdfkit.from_string(
        html,
        str(full_path),
        configuration=_pdf_config(wkhtmltopdf_path),
        options=options,
    )
    print(f"[📝] Custom letter generated: {full_path}")

    response_path = output_path / f"{safe_recipient}_custom_gpt_response.txt"
    with open(response_path, "w", encoding="utf-8") as f:
        f.write(body_paragraph)

    if audit and audit.level is AuditLevel.VERBOSE:
        audit.log_step(
            "custom_letter_generated",
            {
                "account_id": account.get("account_id"),
                "output_pdf": str(full_path),
                "response": body_paragraph,
            },
        )


def generate_custom_letters(
    client_info: dict,
    bureau_data: dict,
    output_path: Path,
    audit: AuditLogger | None,
    *,
    ai_client: AIClient,
    run_date: str | None = None,
    log_messages: list[str] | None = None,
    wkhtmltopdf_path: str | None = None,
) -> None:
    if log_messages is None:
        log_messages = []
    for bureau, content in bureau_data.items():
        for acc in content.get("all_accounts", []):
            action = str(
                acc.get("action_tag") or acc.get("recommended_action") or ""
            ).lower()
            if acc.get("letter_type") == "custom" or action == "custom_letter":
                generate_custom_letter(
                    acc,
                    client_info,
                    output_path,
                    audit,
                    ai_client=ai_client,
                    run_date=run_date,
                    wkhtmltopdf_path=wkhtmltopdf_path,
                )
            else:
                log_messages.append(
                    f"[{bureau}] No custom letter for '{acc.get('name')}' — not marked for custom correspondence"
                )
