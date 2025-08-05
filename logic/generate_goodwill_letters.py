import os
import json
from openai import OpenAI
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
import pdfkit
import re
from logic.utils import (
    gather_supporting_docs_text,
    gather_supporting_docs,
    has_late_indicator,
    safe_filename,
    get_client_address_lines,
    analyze_custom_notes,
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

WKHTMLTOPDF_PATH = os.getenv("WKHTMLTOPDF_PATH", "wkhtmltopdf")
pdf_config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)

template_env = Environment(loader=FileSystemLoader("templates"))
template = template_env.get_template("goodwill_letter_template.html")

COMMON_CREDITOR_ALIASES = {
    "citi": "citibank",
    "citicard": "citibank",
    "citi bank": "citibank",
    "bofa": "bank of america",
    "boa": "bank of america",
    "bk of amer": "bank of america",
    "bank of america": "bank of america",
    "capital one": "capital one",
    "cap one": "capital one",
    "cap1": "capital one",
    "capital 1": "capital one",
    "cap 1": "capital one",
    "chase": "chase bank",
    "jp morgan chase": "chase bank",
    "jpm chase": "chase bank",
    "wells": "wells fargo",
    "wells fargo": "wells fargo",
    "us bank": "us bank",
    "usbank": "us bank",
    "usaa": "usaa",
    "ally": "ally bank",
    "ally financial": "ally bank",
    "synchrony": "synchrony bank",
    "synchrony financial": "synchrony bank",
    "paypal credit": "paypal credit (synchrony)",
    "barclay": "barclays",
    "barclays": "barclays",
    "discover": "discover",
    "comenity": "comenity bank",
    "comenity bank": "comenity bank",
    "td": "td bank",
    "td bank": "td bank",
    "pnc": "pnc bank",
    "pnc bank": "pnc bank",
    "regions": "regions bank",
    "truist": "truist",
    "bbt": "bb&t (now truist)",
    "suntrust": "suntrust (now truist)",
    "avant": "avant",
    "upgrade": "upgrade",
    "sofi": "sofi",
    "earnest": "earnest",
    "upstart": "upstart",
    "marcus": "marcus by goldman sachs",
    "goldman": "marcus by goldman sachs",
    "toyota": "toyota financial",
    "nissan": "nissan motor acceptance corp.",
    "ford": "ford credit",
    "honda": "honda financial services",
    "hyundai": "hyundai motor finance",
    "kia": "kia motors finance",
    "tesla": "tesla finance",
    "navient": "navient",
    "great lakes": "great lakes (nelnet)",
    "mohela": "mohela",
    "aes": "aes (american education services)",
    "fedloan": "fedloan servicing",
    "credit one": "credit one bank",
    "first premier": "first premier bank",
    "mission lane": "mission lane",
    "ollo": "ollo card",
    "reflex": "reflex card",
    "indigo": "indigo card",
    "merrick": "merrick bank",
    "hsbc": "hsbc",
    "bmw financial": "bmw financial",
    "bmw fin svc": "bmw financial",
    "bmw finance": "bmw financial"
}

def normalize_creditor_name(raw_name: str) -> str:
    name = raw_name.lower().strip()
    for alias, canonical in COMMON_CREDITOR_ALIASES.items():
        if alias in name:
            if canonical != name:
                print(f"[~] Alias match: '{raw_name}' -> '{canonical}'")
            return canonical
    name = re.sub(r"\b(bank|usa|na|n.a\\.|llc|inc|corp|co|company)\b", "", name)
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()

def render_html_to_pdf(html: str, output_path: Path):
    options = {"quiet": ""}
    try:
        pdfkit.from_string(html, str(output_path), configuration=pdf_config, options=options)
        print(f"[📬] PDF rendered: {output_path}")
    except Exception as e:
        print(f"[❌] Failed to render PDF: {e}")

def call_gpt_for_goodwill_letter(
    client_name,
    creditor,
    accounts,
    personal_story=None,
    tone="neutral",
    session_id=None,
    custom_note=None,
):
    """Compose a goodwill letter prompt and call GPT.

    ``accounts`` may contain duplicate entries for the same creditor where one
    record has an account number and another does not. We merge those here so
    that GPT receives a single clean summary per account/creditor.
    """

    merged_accounts: list[dict] = []
    seen_numbers: dict[str, dict] = {}

    for acc in accounts:
        acc_num = str(acc.get("account_number") or "").strip()
        name_norm = normalize_creditor_name(acc.get("name", ""))

        target = None
        if acc_num and acc_num in seen_numbers:
            target = seen_numbers[acc_num]
        else:
            # try to locate an existing account with the same creditor name when
            # one of the entries lacks an account number
            for existing in merged_accounts:
                if normalize_creditor_name(existing.get("name", "")) == name_norm:
                    if not acc_num or not existing.get("account_number"):
                        target = existing
                        break

        if target is None:
            target = acc.copy()
            merged_accounts.append(target)
            if acc_num:
                seen_numbers[acc_num] = target
        else:
            for k, v in acc.items():
                if v and not target.get(k):
                    target[k] = v
            if acc_num and not target.get("account_number"):
                target["account_number"] = acc_num
                seen_numbers[acc_num] = target

    account_summaries: list[dict] = []
    
    def summarize_late(late):
        if not isinstance(late, dict):
            return None
        parts = []
        for b, vals in late.items():
            for k, v in vals.items():
                if v:
                    parts.append(f"{v}x{k}-day late ({b})")
        return ", ".join(parts) if parts else None

    seen_numbers_set: set[str] = set()

    for acc in merged_accounts:
        account_number = acc.get("account_number") or acc.get("acct_number")
        status = (
            acc.get("reported_status")
            or acc.get("status")
            or acc.get("account_status")
            or acc.get("payment_status")
        )
        if not account_number:
            print(f"[⚠️] Missing account number for {acc.get('name')}")
        if not status:
            print(f"[⚠️] Missing status for {acc.get('name')}")

        account_number_str = str(account_number or "").strip()
        if account_number_str in seen_numbers_set:
            continue
        seen_numbers_set.add(account_number_str)

        summary = {
            "name": acc.get("name", "Unknown"),
            "account_number": account_number_str or "Unavailable",
            "status": status or "N/A",
            "hardship_reason": acc.get("hardship_reason"),
            "recovery_summary": acc.get("recovery_summary"),
            "personal_note": acc.get("personal_note"),
            "repayment_status": acc.get("account_status") or acc.get("payment_status")
        }
        late_summary = summarize_late(acc.get("late_payments"))
        if late_summary:
            summary["late_history"] = late_summary
        if acc.get("advisor_comment"):
            summary["advisor_comment"] = acc.get("advisor_comment")
        if acc.get("action_tag"):
            summary["action_tag"] = acc.get("action_tag")
        if acc.get("recommended_action"):
            summary["recommended_action"] = acc.get("recommended_action")
        if acc.get("flags"):
            summary["flags"] = acc.get("flags")
        account_summaries.append(summary)

    story_text = f"\nClient's personal story: {personal_story}" if personal_story else ""
    note_text = f"\nClient note for {creditor}: {custom_note}" if custom_note else ""

    docs_text, doc_names, _ = gather_supporting_docs(session_id or "")
    if docs_text:
        print(f"[📎] Including supplemental docs for goodwill letter to {creditor}.")
        docs_section = f"\nThe following additional documents were provided by the client:\n{docs_text}"
    else:
        docs_section = ""

    prompt = f"""
Write a goodwill adjustment letter for credit reporting purposes. Write it **in the first person**, in a {tone} tone as if the client wrote it.
For each account below, craft a short story-style paragraph summarizing the situation and referencing any hardship and recovery details.
If certain details are missing, create a brief empathetic explanation. Mention supporting documents by name when helpful.

Include these fields in the JSON response:
- intro_paragraph: opening lines
- hardship_paragraph: brief explanation of the hardship
- recovery_paragraph: how things improved
- accounts: list of {{name, account_number, status, paragraph}}
- closing_paragraph: polite request for goodwill adjustment

Creditor: {creditor}
Client name: {client_name}
Accounts: {json.dumps(account_summaries, indent=2)}
Supporting doc names: {', '.join(doc_names) if doc_names else 'None'}
{story_text}
{docs_section}
{note_text}

Return valid JSON only. No markdown.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    print("\n----- GPT RAW RESPONSE -----")
    print(content)
    print("----- END RESPONSE -----\n")

    return json.loads(content)

def load_creditor_address_map():
    try:
        with open("data/creditor_addresses.json", encoding="utf-8") as f:
            raw = json.load(f)
            if isinstance(raw, list):
                return {
                    normalize_creditor_name(entry["name"]): entry["address"]
                    for entry in raw if "name" in entry and "address" in entry
                }
            elif isinstance(raw, dict):
                return {normalize_creditor_name(k): v for k, v in raw.items()}
            else:
                print("[⚠️] Unknown address file format.")
                return {}
    except Exception as e:
        print(f"[❌] Failed to load creditor addresses: {e}")
        return {}

def generate_goodwill_letter_with_ai(creditor, accounts, client_info, output_path: Path, run_date: str = None):
    client_name = client_info.get("legal_name") or client_info.get("name", "Your Name")
    if not client_info.get("legal_name"):
        print("[⚠️] Warning: legal_name not found in client_info. Using fallback name.")

    personal_story = client_info.get("story") or ""
    tone = client_info.get("tone", "neutral")
    date_str = run_date or datetime.now().strftime("%B %d, %Y")

    address_map = load_creditor_address_map()
    creditor_key = normalize_creditor_name(creditor)
    creditor_address = address_map.get(creditor_key)

    if not creditor_address:
        print(f"[⚠️] No address found for: {creditor}")
        creditor_address = "Address not provided — please enter manually"

    session_id = client_info.get("session_id")
    raw_notes = client_info.get("custom_dispute_notes", {}) or {}
    acc_names = [a.get("name", "") for a in accounts]
    specific_notes, general_notes = analyze_custom_notes(raw_notes, acc_names)
    notes_map = {normalize_creditor_name(k): v for k, v in specific_notes.items()}
    custom_note = notes_map.get(normalize_creditor_name(creditor))
    if custom_note:
        print(f"[📌] Included client note for goodwill letter on: '{creditor}'")
    if general_notes:
        personal_story = (personal_story + " " + " ".join(general_notes)).strip()

    gpt_data = call_gpt_for_goodwill_letter(
        client_name,
        creditor,
        accounts,
        personal_story,
        tone,
        session_id,
        custom_note,
    )

    _, doc_names, _ = gather_supporting_docs(session_id or "")

    context = {
        "date": date_str,
        "client_name": client_name,
        "client_address_lines": get_client_address_lines(client_info),
        "creditor": creditor,
        "creditor_address": creditor_address,
        "accounts": gpt_data.get("accounts", []),
        "intro_paragraph": gpt_data.get("intro_paragraph", ""),
        "hardship_paragraph": gpt_data.get("hardship_paragraph", ""),
        "recovery_paragraph": gpt_data.get("recovery_paragraph", ""),
        "closing_paragraph": gpt_data.get("closing_paragraph", ""),
        "supporting_docs": doc_names
    }

    html = template.render(**context)
    safe_name = safe_filename(creditor)
    filename = f"Goodwill Request - {safe_name}.pdf"
    full_path = output_path / filename
    render_html_to_pdf(html, full_path)

    with open(output_path / f"{safe_name}_gpt_response.json", 'w') as f:
        json.dump(gpt_data, f, indent=2)

def generate_goodwill_letters(client_info, bureau_data, output_path: Path, run_date: str = None):
    seen_creditors = set()
    goodwill_accounts = {}
    detected_late_accounts: dict[str, str] = {}
    flagged_candidates: dict[str, str] = {}

    def clean_num(num: str | None) -> str:
        import re
        return re.sub(r"\D", "", num or "")

    dispute_map: dict[str, set[str]] = {}
    for content in bureau_data.values():
        for acc in content.get("disputes", []):
            action = str(acc.get("action_tag") or acc.get("recommended_action") or "").lower()
            if action != "dispute":
                continue
            name = acc.get("name") or acc.get("שם החשבון")
            if not name:
                continue
            name_norm = normalize_creditor_name(name)
            dispute_map.setdefault(name_norm, set()).add(clean_num(acc.get("account_number")))

    def consider_account(account):
        status_text = str(
            account.get("status")
            or account.get("account_status")
            or account.get("payment_status")
            or ""
        ).lower()
        if any(
            kw in status_text
            for kw in (
                "collection",
                "chargeoff",
                "charge-off",
                "charge off",
                "repossession",
                "repos",
                "delinquent",
                "late payments",
            )
        ):
            return
        name_norm = normalize_creditor_name(account.get("name") or account.get("שם החשבון") or "")
        acct_num = clean_num(account.get("account_number") or account.get("acct_number"))
        dispute_nums = dispute_map.get(name_norm)
        if dispute_nums is not None:
            for dn in dispute_nums:
                if not dn or not acct_num or dn == acct_num:
                    print(f"[🚫] Goodwill skipped for disputed account: '{account.get('name')}'")
                    return
        if account.get("goodwill_candidate") or account.get("goodwill_on_closed"):
            flagged_candidates.setdefault(name_norm, account.get("name") or account.get("שם החשבון") or name_norm)
        late_info = account.get("late_payments")

        def _total_lates(info) -> int:
            total = 0
            if isinstance(info, dict):
                for bureau_vals in info.values():
                    if isinstance(bureau_vals, dict):
                        for v in bureau_vals.values():
                            try:
                                total += int(v)
                            except (TypeError, ValueError):
                                continue
            return total

        if _total_lates(late_info) == 0 and not has_late_indicator(account):
            return

        creditor = account.get("name") or account.get("שם החשבון")
        if not creditor:
            return

        key = name_norm
        detected_late_accounts[key] = creditor
        print(f"[🔍] Detected late payments for: '{creditor}' → {late_info or {}}")

        if key not in seen_creditors:
            goodwill_accounts.setdefault(creditor, []).append(account)
            seen_creditors.add(key)
        else:
            goodwill_accounts[creditor].append(account)

    for bureau, content in bureau_data.items():
        candidate_sections = [
            content.get("goodwill", []),
            content.get("disputes", []),
            content.get("high_utilization", []),
        ]

        for section in candidate_sections:
            for account in section:
                consider_account(account)

    # Scan all accounts from the full analysis for missed late history
    for section in [
        "all_accounts",
        "positive_accounts",
        "open_accounts_with_issues",
        "negative_accounts",
        "high_utilization_accounts",
    ]:
        for account in client_info.get(section, []):
            consider_account(account)

    for creditor, accounts in goodwill_accounts.items():
        generate_goodwill_letter_with_ai(creditor, accounts, client_info, output_path, run_date)

    for norm, raw in detected_late_accounts.items():
        if norm not in {normalize_creditor_name(c) for c in goodwill_accounts}:
            print(f"[⚠️] Goodwill skipped for: '{raw}' despite having late payments")

    included_norms = {normalize_creditor_name(c) for c in goodwill_accounts}
    for norm, raw in flagged_candidates.items():
        if norm not in included_norms:
            print(f"[⚠️] Goodwill candidate '{raw}' skipped – reason: no late payment data")
