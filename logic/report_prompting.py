"""Prompt construction and AI calls for credit report analysis."""

from pathlib import Path

from logic.utils.names_normalization import (
    normalize_creditor_name,
)
from logic.utils.text_parsing import extract_late_history_blocks
from logic.utils.inquiries import extract_inquiries
from .json_utils import parse_json
from services.ai_client import AIClient


def call_ai_analysis(
    text: str,
    client_goal: str,
    is_identity_theft: bool,
    output_json_path: Path,
    ai_client: AIClient,
):
    """Analyze raw report text using an AI model and return parsed JSON.

    Parameters
    ----------
    text:
        Raw text extracted from the report.
    client_goal:
        High level goal provided by the client.
    is_identity_theft:
        Flag indicating if the report relates to an identity theft case.
    output_json_path:
        Location where the resulting JSON (and raw response debug file) should
        be written.
    ai_client:
        Instance of :class:`services.ai_client.AIClient` used to make the
        request.

    Returns
    -------
    dict
        Parsed JSON result from the AI model.
    """
    late_blocks, late_raw_map = extract_late_history_blocks(text, return_raw_map=True)
    if late_blocks:
        late_summary_text = "Late payment history extracted from report:\n"
        for account, bureaus in late_blocks.items():
            raw = late_raw_map.get(account, account)
            for bureau, counts in bureaus.items():
                parts = [f"{k} days late: {v}" for k, v in counts.items()]
                goodwill = False
                thirty = counts.get("30", 0)
                sixty = counts.get("60", 0)
                ninety = counts.get("90", 0)
                if (thirty == 1 and not sixty and not ninety) or (
                    thirty == 2 and not sixty and not ninety
                ):
                    goodwill = True
                line = f"- {account} (raw: {raw}) ({bureau}): {', '.join(parts)}"
                if goodwill:
                    line += " \u2192 Possible goodwill candidate"
                late_summary_text += line + "\n"
    else:
        late_summary_text = (
            "No late payment blocks were detected by parsing logic. "
            "Do not infer or guess late payment history unless clearly shown."
        )

    inquiry_list = extract_inquiries(text)
    if inquiry_list:
        inquiry_summary = "\nParsed inquiries from report:\n" + "\n".join(
            f"- {normalize_creditor_name(i['creditor_name'])} (raw: {i['creditor_name']}) {i['date']} ({i['bureau']})"
            for i in inquiry_list
        )
    else:
        inquiry_summary = "\nNo inquiries were detected by the parser."

    prompt = f"""
You are a senior credit repair expert with deep knowledge of credit reports, FCRA regulations, dispute strategies, and client psychology.

Your task: deeply analyze the following SmartCredit report text and extract a structured JSON summary for use in automated dispute and goodwill letters, and personalized client instructions.

Client goal: "{client_goal}"
Identity theft case: {"Yes" if is_identity_theft else "No"}

Return only valid JSON with all property names in double quotes. No comments or extra text outside the JSON object.

Return this exact JSON structure:

1. personal_info_issues: List of inconsistencies or mismatches in personal info (name, address, DOB) across bureaus.

2. negative_accounts: Accounts marked as Chargeoff or in Collections.
   For each account, include:
   - name
   - balance (e.g., "$2,356")
   - status ("Chargeoff" or "Collections")
   - account_status ("Open" or "Closed")
   - account_number (if visible)
   - bureaus (["Experian", "Equifax", "TransUnion"])
   - is_suspected_identity_theft: true/false
   - dispute_type: "identity_theft" / "unauthorized_or_unverified" / "inaccurate_reporting"
   - impact: short sentence describing how this negatively affects the client
   - advisor_comment: 1–2 sentences explaining what we’re doing about this account and why.

3. open_accounts_with_issues: Non-collection accounts with any late payment history or remarks like "past due", "derogatory", etc. Look carefully for text such as "30 days late", "60 days late", or "past due" even if formatting is odd. If an account is otherwise in good standing with only one or two late payments, mark ``goodwill_candidate`` true.
   Include:
   - name
   - status
   - account_status
   - account_number (if available)
   - bureaus
   - goodwill_candidate: true/false
   - is_suspected_identity_theft: true/false
   - dispute_type
   - hardship_reason: (e.g. "temporary job loss")
   - recovery_summary
   - advisor_comment

4. high_utilization_accounts: Revolving accounts with balance over 30% of limit.
   Include:
   - name
   - balance
   - limit
   - utilization (e.g., "76%")
   - bureaus
   - is_suspected_identity_theft: true/false
   - advisor_comment
   - recommended_payment_amount

5. positive_accounts: Accounts in good standing that support the credit profile.
   Include:
   - name
   - account_number (if available)
   - opened_date (if available)
   - balance
   - status
   - utilization (if revolving)
   - bureaus
   - advisor_comment

6. inquiries: Hard inquiries from the last 2 years that are NOT tied to any existing account.
   Exclude inquiries that clearly relate to open or closed accounts listed. Treat names with minor spelling differences as the same creditor when deciding to exclude.
   Include:
   - creditor_name
   - date (MM/YYYY)
   - bureau

7. account_inquiry_matches: Inquiries clearly matching existing accounts (to be excluded from disputes).
   Include:
   - creditor_name
   - matched_account_name

8. summary_metrics:
   - num_collections
   - num_late_payments
   - high_utilization (true/false)
   - recent_inquiries
   - total_inquiries
   - num_negative_accounts
   - num_accounts_over_90_util
   - account_types_in_problem

9. strategic_recommendations: Clear action items based on the client goal, such as:
   - "Open secured card"
   - "Lower utilization on Capital One"
   - "Dispute Midland account with Experian"

10. all_accounts: ✅ For every account (positive or negative), return:
   - name
   - bureaus
   - status
   - utilization (if available)
   - advisor_comment: 1–2 sentence explanation of the account’s effect and what the client should do (dispute, pay down, keep healthy, goodwill, etc.)

⚠️ Rules:
- Return strictly valid JSON
- All property names and strings must use double quotes
- No trailing commas, comments, or text outside the JSON
- No markdown or explanations
- Use proper casing, punctuation, and clean formatting
- Never guess — only include facts that are visible
Use the following late payment data to help you accurately tag late accounts, even if the report formatting is inconsistent:

{late_summary_text}

{inquiry_summary}

Report text:
===
{text}
===
"""

    response = ai_client.chat_completion(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    content = response.choices[0].message.content.strip()

    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    raw_path = output_json_path.with_name(output_json_path.stem + "_raw.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(content)
        f.write("\n\n---\n[Debug] Late Payment Summary Used in Prompt:\n")
        f.write(late_summary_text)
        f.write("\n\n---\n[Debug] Inquiry Summary Used in Prompt:\n")
        f.write(inquiry_summary)

    data, _ = parse_json(content)
    return data
