import os
import json
from openai import OpenAI
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
import pdfkit
from logic.utils import (
    gather_supporting_docs,
    get_client_address_lines,
    analyze_custom_notes,
    CHARGEOFF_RE,
    COLLECTION_RE,
)


def dedupe_disputes(disputes: list[dict], bureau_name: str, log: list[str]) -> list[dict]:
    """Remove duplicate dispute entries based on creditor name and account number."""
    from .generate_goodwill_letters import normalize_creditor_name

    def _sanitize(num: str | None) -> str | None:
        if not num:
            return None
        digits = "".join(c for c in str(num) if c.isdigit())
        if not digits:
            return None
        return digits[-4:] if len(digits) >= 4 else digits

    seen: set[tuple[str, str | None]] = set()
    deduped: list[dict] = []
    for d in disputes:
        name_key = normalize_creditor_name(d.get("name", "")).lower()
        num = _sanitize(d.get("account_number"))
        key = (name_key, num)
        if key in seen:
            log.append(f"[{bureau_name}] Skipping duplicate account '{d.get('name')}'")
            continue
        seen.add(key)
        deduped.append(d)
    return deduped

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
WKHTMLTOPDF_PATH = os.getenv("WKHTMLTOPDF_PATH", "wkhtmltopdf")

CREDIT_BUREAU_ADDRESSES = {
    "Experian": "P.O. Box 4500, Allen, TX 75013",
    "Equifax": "P.O. Box 740256, Atlanta, GA 30374-0256",
    "TransUnion": "P.O. Box 2000, Chester, PA 19016-2000"
}

# Default dispute reason inserted when no custom note is provided.
# This language cites the relevant FCRA sections and requests
# signed documentation. It is used for every bureau when the
# client has not supplied a custom note.
DEFAULT_DISPUTE_REASON = (
    "I formally dispute this account as inaccurate and unverifiable. "
    "Under my rights granted by the Fair Credit Reporting Act (FCRA) "
    "sections 609(a) and 611, I demand that you provide copies of any "
    "original signed contracts, applications or other documents bearing my "
    "signature that you relied upon to report this account. If these documents "
    "cannot be produced within 30 days, the account must be deleted from my "
    "credit file."
)

# Additional closing paragraph warning about escalation.
ESCALATION_NOTE = (
    "If you fail to fully verify these accounts with proper documentation within "
    "30 days, I expect them to be deleted immediately as required by law. Failure "
    "to comply may result in further legal actions or formal complaints filed with "
    "the FTC and CFPB."
)

def render_html_to_pdf(html_string: str, output_path: Path):
    try:
        config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)
        options = {"quiet": ""}
        pdfkit.from_string(html_string, str(output_path), configuration=config, options=options)
        print(f"[📄] PDF rendered: {output_path}")
    except Exception as e:
        print(f"[❌] Failed to render PDF: {e}")

def render_dispute_letter_html(context: dict) -> str:
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("dispute_letter_template.html")
    return template.render(**context)

def call_gpt_dispute_letter(client_info, bureau_name, disputes, inquiries, is_identity_theft):
    """Generate GPT-powered dispute letter content."""
    import json

    client_name = client_info.get("legal_name") or client_info.get("name", "Client")
    custom_notes = client_info.get("custom_dispute_notes", {}) or {}

    dispute_blocks = []
    for acc in disputes:
        block = {
            "name": acc.get("name", "Unknown"),
            "account_number": acc.get("account_number", "").replace("*", "") or "N/A",
            "status": acc.get("reported_status") or acc.get("status", "N/A"),
            "dispute_type": acc.get("dispute_type", "unspecified"),
        }
        if acc.get("advisor_comment"):
            block["advisor_comment"] = acc.get("advisor_comment")
        if acc.get("action_tag"):
            block["action_tag"] = acc.get("action_tag")
        if acc.get("recommended_action"):
            block["recommended_action"] = acc.get("recommended_action")
        if acc.get("flags"):
            block["flags"] = acc.get("flags")
        note = custom_notes.get(acc.get("name"))
        if note:
            block["personal_note"] = note
        dispute_blocks.append(block)

    inquiry_blocks = [
        {
            "creditor_name": inq.get("creditor_name", "Unknown"),
            "date": inq.get("date", "Unknown"),
            "bureau": inq.get("bureau", bureau_name)
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

Respond only with JSON.
"""

    prompt = f'''
You are a professional legal assistant helping a consumer draft a formal credit dispute letter. Write the content **in the first person** as if the client is speaking directly. The letter must comply with the Fair Credit Reporting Act (FCRA).

Client: {client_name}
Credit Bureau: {bureau_name}
Identity Theft (confirmed by client): {"Yes" if is_identity_theft else "No"}

Each disputed account below includes a dispute_type — identity_theft / unauthorized_or_unverified / inaccurate_reporting. Write a short custom paragraph per account referencing the appropriate FCRA section (e.g. 611 or 609(a)(1)) and any personal notes. Include a clear requested action such as deletion or correction.

Disputed Accounts:
{json.dumps(dispute_blocks, indent=2)}

Unauthorized Inquiries:
{json.dumps(inquiry_blocks, indent=2)}

{instruction_text}
'''

    session_id = client_info.get("session_id", "")
    docs_text, doc_names, _ = gather_supporting_docs(session_id)
    if docs_text:
        print(f"[📎] Including supplemental docs for {bureau_name} prompt.")
        prompt += (
            "\nThe client also provided the following supporting documents:\n"
            f"{docs_text}\n"
            "You may reference them in the overall letter if helpful, but do not "
            "include separate document notes for each account."
        )

    from openai import OpenAI
    import os
    from dotenv import load_dotenv

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    print("\n----- GPT RAW RESPONSE -----")
    print(content)
    print("----- END RESPONSE -----\n")

    result = json.loads(content)
    return result

def generate_all_dispute_letters_with_ai(
    client_info,
    bureau_data: dict,
    output_path: Path,
    is_identity_theft: bool,
    run_date: str = None,
    log_messages: list[str] | None = None,
):
    output_path.mkdir(parents=True, exist_ok=True)

    if log_messages is None:
        log_messages = []

    account_inquiry_matches = client_info.get("account_inquiry_matches", [])
    from .generate_goodwill_letters import normalize_creditor_name
    client_name = client_info.get("legal_name") or client_info.get("name", "Client")

    if not client_info.get("legal_name"):
        print("[⚠️] Warning: legal_name not found in client_info. Using fallback name.")

    for bureau_name, payload in bureau_data.items():
        print(f"[🤖] Generating letter for {bureau_name}...")

        disputes = []
        for d in payload.get("disputes", []):
            action = str(d.get("action_tag") or d.get("recommended_action") or "").lower()
            if action == "dispute":
                disputes.append(d)
            else:
                log_messages.append(
                    f"[{bureau_name}] Skipping account '{d.get('name')}' – recommended_action='{action}'"
                )

        disputes = dedupe_disputes(disputes, bureau_name, log_messages)

        acc_type_map = {}
        for d in disputes:
            key = (
                normalize_creditor_name(d.get("name", "")),
                (d.get("account_number") or "").replace("*", "").strip(),
            )
            acc_type_map[key] = {
                "account_type": str(d.get("account_type") or ""),
                "status": str(d.get("status") or d.get("account_status") or ""),
            }

        inquiries = payload.get("inquiries", [])
        print(f"[🔍] {len(inquiries)} inquiries for {bureau_name} to evaluate:")
        for raw_inq in inquiries:
            print(
                f"    ➡️ {raw_inq.get('creditor_name')} — {raw_inq.get('date')} ({raw_inq.get('bureau', bureau_name)})"
            )

        # Build a set of normalized names from provided account matches

        matched_set = {normalize_creditor_name(m.get("creditor_name", "")) for m in account_inquiry_matches}
        open_account_names = set()
        open_account_map = {}
        for a in payload.get("all_accounts", []):
            status_text = str(a.get("account_status") or a.get("status") or "").lower()
            if "closed" not in status_text:
                norm_name = normalize_creditor_name(a.get("name", ""))
                open_account_names.add(norm_name)
                open_account_map[norm_name] = a.get("name")

        # Include open accounts from the full client data as well
        for section in [
            "all_accounts",
            "open_accounts_with_issues",
            "positive_accounts",
            "negative_accounts",
        ]:
            for a in client_info.get(section, []):
                status_text = str(a.get("account_status") or a.get("status") or "").lower()
                if "closed" not in status_text:
                    norm_name = normalize_creditor_name(a.get("name", ""))
                    open_account_names.add(norm_name)
                    open_account_map.setdefault(norm_name, a.get("name"))

        filtered_inquiries = []
        for inq in inquiries:
            name_norm = normalize_creditor_name(inq.get("creditor_name", ""))
            matched = name_norm in matched_set or name_norm in open_account_names
            matched_label = open_account_map.get(name_norm)
            if name_norm in matched_set and not matched_label:
                matched_label = "matched list"
            print(
                f"📄 Inquiry being evaluated: {inq.get('creditor_name')} on {inq.get('bureau', bureau_name)} {inq.get('date')} -> {'matched to ' + matched_label if matched_label else 'no match'}"
            )
            if not matched:
                filtered_inquiries.append(inq)
                print(
                    f"[Will be disputed] Inquiry detected: {inq.get('creditor_name')}, {inq.get('date')}, {bureau_name}"
                )
                print(
                    f"✅ Inquiry added to dispute letter: {inq.get('creditor_name')} — {inq.get('date')} ({bureau_name})"
                )
            else:
                print(
                    f"🚫 Inquiry skipped due to open account match: {matched_label or inq.get('creditor_name')}"
                )
                log_messages.append(
                    f"[{bureau_name}] Skipping inquiry '{inq.get('creditor_name')}' matched to existing account"
                )

        if not disputes and not filtered_inquiries:
            msg = f"[{bureau_name}] No disputes or inquiries after filtering - letter skipped"
            print(f"[⚠️] No data to dispute for {bureau_name}, skipping.")
            log_messages.append(msg)
            continue

        bureau_address = CREDIT_BUREAU_ADDRESSES.get(bureau_name, "Unknown")

        for d in disputes:
            if not (is_identity_theft and d.get("is_suspected_identity_theft", False)):
                if d.get("is_suspected_identity_theft"):
                    d["dispute_type"] = "unauthorized_or_unverified"
                else:
                    d["dispute_type"] = d.get("dispute_type", "inaccurate_reporting")
            else:
                d["dispute_type"] = "identity_theft"

        # Analyze custom notes to separate account-specific vs general hardship content
        raw_notes = client_info.get("custom_dispute_notes", {}) or {}
        acc_names = [d.get("name", "") for d in disputes]
        specific_notes, general_notes = analyze_custom_notes(raw_notes, acc_names)
        if general_notes:
            print(
                f"[~] {len(general_notes)} general note(s) detected – excluded from dispute letter"
            )

        client_info_for_gpt = dict(client_info)
        client_info_for_gpt["custom_dispute_notes"] = specific_notes

        gpt_data = call_gpt_dispute_letter(
            client_info_for_gpt,
            bureau_name,
            disputes,
            filtered_inquiries,
            is_identity_theft,
        )

        # 🔍 Post-process GPT output to enforce dispute-only language
        note_map = {normalize_creditor_name(k): v for k, v in specific_notes.items()}
        for acc in gpt_data.get("accounts", []):
            name_key = normalize_creditor_name(acc.get("name", ""))
            custom_note = note_map.get(name_key)
            if custom_note:
                # Combine client's note with the strong default language
                acc["paragraph"] = (custom_note.strip() + " " + DEFAULT_DISPUTE_REASON)
                acc.pop("requested_action", None)
                acc.pop("personal_note", None)
            else:
                # Apply the standard dispute language when no custom note exists
                acc["paragraph"] = DEFAULT_DISPUTE_REASON
                acc.pop("personal_note", None)
                # Override any GPT requested action to avoid goodwill wording
                action = acc.get("requested_action", "")
                if isinstance(action, str) and ("goodwill" in action.lower() or "hardship" in action.lower()):
                    acc["requested_action"] = "Please verify this item and correct or remove any inaccuracies."

            lookup_key = (
                name_key,
                (acc.get("account_number") or "").replace("*", "").strip(),
            )
            acc_info = acc_type_map.get(lookup_key) or acc_type_map.get((name_key, ""))
            if acc_info:
                status_text = (
                    (acc_info.get("account_type", "") + " " + acc_info.get("status", "")).lower()
                )
                if "collection" in status_text:
                    acc["paragraph"] = acc["paragraph"].rstrip() + " Please also provide evidence of assignment or purchase agreements from the original creditor to the collection agency proving legal authority to collect this debt."
                elif CHARGEOFF_RE.search(status_text):
                    acc["paragraph"] = acc["paragraph"].rstrip() + " Please provide all original signed contracts or documents directly from the original creditor supporting this charge-off."

            acct_num = acc.get("account_number")
            if (
                isinstance(acct_num, str)
                and acct_num.strip()
                and acct_num.upper() != "N/A"
                and not acct_num.endswith("***")
            ):
                acc["account_number"] = acct_num + "***"

        # Append escalation language to the closing paragraph
        closing = gpt_data.get("closing_paragraph", "").strip()
        gpt_data["closing_paragraph"] = (closing + (" " if closing else "") + ESCALATION_NOTE).strip()

        included_set = {
            (
                normalize_creditor_name(i.get("creditor_name", "")),
                i.get("date"),
            )
            for i in gpt_data.get("inquiries", [])
        }
        for inq in filtered_inquiries:
            key = (
                normalize_creditor_name(inq.get("creditor_name", "")),
                inq.get("date"),
            )
            if key not in included_set:
                print(
                    f"[⚠️] Inquiry '{inq.get('creditor_name')}' expected in dispute letter but was not included."
                )

        if run_date is None:
            run_date = datetime.now().strftime("%B %d, %Y")

        context = {
            "client_name": client_name,
            "client_street": client_info.get("street", ""),
            "client_city": client_info.get("city", ""),
            "client_state": client_info.get("state", ""),
            "client_zip": client_info.get("zip", ""),
            "client_address_lines": get_client_address_lines(client_info),
            "bureau_name": bureau_name,
            "bureau_address": bureau_address,
            "date": run_date,
            "opening_paragraph": gpt_data.get("opening_paragraph", ""),
            "accounts": gpt_data.get("accounts", []),
            "inquiries": gpt_data.get("inquiries", []),
            "closing_paragraph": gpt_data.get("closing_paragraph", ""),
            "is_identity_theft": is_identity_theft,
        }

        html = render_dispute_letter_html(context)
        filename = f"Dispute Letter - {bureau_name}.pdf"
        filepath = output_path / filename
        render_html_to_pdf(html, filepath)

        with open(output_path / f"{bureau_name}_gpt_response.json", 'w') as f:
            json.dump(gpt_data, f, indent=2)

generate_dispute_letters_for_all_bureaus = generate_all_dispute_letters_with_ai
