"""GPT prompt construction and OpenAI interaction for dispute letters."""

from __future__ import annotations

import json
from typing import Any, List, Mapping

from backend.audit.audit import AuditLevel, AuditLogger
from backend.core.logic.compliance.rules_loader import get_neutral_phrase
from backend.core.logic.strategy.summary_classifier import ClassificationRecord
from backend.core.logic.utils.json_utils import parse_json
from backend.core.logic.utils.pdf_ops import gather_supporting_docs
from backend.core.models.account import Account, Inquiry
from backend.core.models.client import ClientInfo
from backend.core.models.letter import LetterContext
from backend.core.services.ai_client import AIClient


def call_gpt_dispute_letter(
    client_info: ClientInfo | Mapping[str, Any],
    bureau_name: str,
    disputes: List[Account],
    inquiries: List[Inquiry],
    is_identity_theft: bool,
    structured_summaries: Mapping[str, Mapping[str, Any]],
    state: str,
    classification_map: Mapping[str, ClassificationRecord] | None,
    ai_client: AIClient,
    audit: AuditLogger | None = None,
) -> LetterContext:
    """Generate GPT-powered dispute letter content."""

    client_dict = (
        client_info.to_dict()
        if isinstance(client_info, ClientInfo)
        else dict(client_info)
    )
    client_name = client_dict.get("legal_name") or client_dict.get("name", "Client")
    session_id = client_dict.get("session_id", "")

    dispute_blocks = []
    for acc in disputes:
        struct = structured_summaries.get(acc.account_id or "", {})
        record = None
        if classification_map:
            record = classification_map.get(acc.account_id or "")
        classification = record.classification if record else {}
        neutral_phrase, neutral_reason = get_neutral_phrase(
            classification.get("category"), struct
        )
        block = {
            "name": acc.name or "Unknown",
            "account_number": (acc.account_number or "").replace("*", "") or "N/A",
            "status": acc.reported_status or acc.status or "N/A",
            "dispute_type": classification.get(
                "category", acc.dispute_type or "unspecified"
            ),
            "legal_hook": classification.get("legal_tag"),
            "tone": classification.get("tone"),
            "dispute_approach": classification.get("dispute_approach"),
            "structured_summary": struct,
        }
        if neutral_phrase:
            block["neutral_phrase"] = neutral_phrase
        if classification.get("state_hook"):
            block["state_hook"] = classification["state_hook"]
        if acc.advisor_comment:
            block["advisor_comment"] = acc.advisor_comment
        if acc.action_tag:
            block["action_tag"] = acc.action_tag
        if acc.recommended_action:
            block["recommended_action"] = acc.recommended_action
        if acc.flags:
            block["flags"] = acc.flags
        dispute_blocks.append(block)
        if audit:
            audit.log_account(
                acc.account_id or acc.name,
                {
                    "stage": "dispute_letter",
                    "bureau": bureau_name,
                    "structured_summary": struct,
                    "classification": classification,
                    "neutral_phrase": neutral_phrase,
                    "neutral_phrase_reason": neutral_reason,
                    "recommended_action": acc.recommended_action,
                },
            )

    inquiry_blocks = [
        {
            "creditor_name": inq.creditor_name or "Unknown",
            "date": inq.date or "Unknown",
            "bureau": inq.bureau or bureau_name,
        }
        for inq in inquiries
    ]

    instruction_text = """
Return a JSON object with:
- opening_paragraph (should start with 'I am formally requesting an investigation')
- accounts: list of objects containing
    - name
    - account_number
    - status
    - paragraph (2-3 sentence description referencing FCRA rights and any notes)
    - requested_action
- inquiries: list of {creditor_name, date}
- closing_paragraph
  (should mention the bureau must respond in writing within 30 days under section 611 of the FCRA)

Respond only with JSON. The output must be strictly valid JSON: all property names and strings in double quotes, no trailing commas or comments, and no text outside the JSON.
"""

    prompt = f"""
You are a professional legal assistant helping a consumer draft a formal credit dispute letter. Write the content **in the first person** as if the client is speaking directly. The letter must comply with the Fair Credit Reporting Act (FCRA).

Client: {client_name}
Credit Bureau: {bureau_name}
State: {state}
Identity Theft (confirmed by client): {'Yes' if is_identity_theft else 'No'}

Each disputed account below includes a dispute_type classification, a neutral_phrase template, and the client's structured_summary. For each account, write a short custom paragraph that blends the neutral_phrase with the client's explanation, guided by the legal_hook and tone. Do not copy the phrase or explanation verbatim; instead, craft a professional summary and include a clear requested action such as deletion or correction.

Disputed Accounts:
{json.dumps(dispute_blocks, indent=2)}

Unauthorized Inquiries:
{json.dumps(inquiry_blocks, indent=2)}

{instruction_text}
"""

    docs_text, doc_names, _ = gather_supporting_docs(session_id)
    if docs_text:
        if audit and audit.level is AuditLevel.VERBOSE:
            print(f"[📎] Including supplemental docs for {bureau_name} prompt.")
        prompt += (
            "\nThe client also provided the following supporting documents:\n"
            f"{docs_text}\n"
            "You may reference them in the overall letter if helpful, but do not "
            "include separate document notes for each account."
        )
    if audit and audit.level is AuditLevel.VERBOSE:
        audit.log_step(
            "dispute_prompt",
            {
                "bureau": bureau_name,
                "prompt": prompt,
                "accounts": dispute_blocks,
                "inquiries": inquiry_blocks,
            },
        )
    response = ai_client.chat_completion(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    if audit and audit.level is AuditLevel.VERBOSE:
        print("\n----- GPT RAW RESPONSE -----")
        print(content)
        print("----- END RESPONSE -----\n")

    result, _ = parse_json(content)
    context = LetterContext.from_dict(result)
    if audit and audit.level is AuditLevel.VERBOSE:
        audit.log_step(
            "dispute_response",
            {"bureau": bureau_name, "response": result},
        )
    return context


__all__ = ["call_gpt_dispute_letter"]
