import os
import html as html_utils
import json
import random
import base64
from openai import OpenAI
from logic.generate_goodwill_letters import normalize_creditor_name
from logic.utils import analyze_custom_notes
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
import pdfkit
from logic.analyze_report import validate_analysis_sanity

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
env = Environment(loader=FileSystemLoader("templates"))


def get_logo_base64() -> str:
    """Return the Credit Impact logo encoded as a base64 data URI."""
    logo_path = Path("templates/Logo_CreditImpact.png")
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    return ""

def extract_clean_name(full_name: str) -> str:
    parts = full_name.strip().split()
    seen = set()
    unique_parts = []
    for part in parts:
        if part.lower() not in seen:
            unique_parts.append(part)
            seen.add(part.lower())
    return " ".join(unique_parts)

def render_html_to_pdf(html_string: str, output_path: Path):
    config = pdfkit.configuration(wkhtmltopdf=os.getenv("WKHTMLTOPDF_PATH", "wkhtmltopdf"))
    options = {"quiet": ""}
    try:
        pdfkit.from_string(html_string, str(output_path), configuration=config, options=options)
        print(f"[📄] PDF rendered: {output_path}")
    except Exception as e:
        print(f"[❌] Failed to render PDF: {e}")

def generate_account_action(account: dict) -> str:
    """Return a human-readable action sentence for an account using GPT."""
    try:
        prompt = (
            "You are a friendly credit repair coach speaking in plain English. "
            "Write one short sentence explaining what the client should do next "
            "for the account below. Keep it simple and avoid jargon like 'utilization' or 'negatively impacts.' "
            "If no action is needed, give a quick reassuring note.\n\n"
            f"Account data:\n{json.dumps(account, indent=2)}\n\n"
            "Respond with only the sentence."
        )
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.replace("```", "").strip()
        return content
    except Exception as e:
        print(f"[⚠️] GPT action generation failed: {e}")
        return "Review the attached letter and follow the standard mailing steps." 

def render_instruction_html(context: dict) -> str:
    template = env.get_template("instruction_template.html")
    return template.render(**context)

def generate_instruction_file(
    client_info,
    bureau_data,
    is_identity_theft: bool,
    output_path: Path,
    run_date: str | None = None,
    strategy: dict | None = None,
):
    """Generate the instruction PDF and JSON context for the client."""
    run_date = run_date or datetime.now().strftime("%B %d, %Y")
    logo_base64 = get_logo_base64()

    html, all_accounts = generate_html(
        client_info,
        bureau_data,
        is_identity_theft,
        run_date,
        logo_base64,
        strategy,
    )

    render_pdf_from_html(html, output_path)
    save_json_output(all_accounts, output_path)

    print("[✅] Instructions file generated successfully.")


def generate_html(
    client_info,
    bureau_data,
    is_identity_theft: bool,
    run_date: str,
    logo_base64: str,
    strategy: dict | None = None,
):
    client_name = extract_clean_name(client_info.get("name") or "Client")

    all_accounts: list[dict] = []

    def sanitize_number(num: str | None) -> str:
        if not num:
            return ""
        return re.sub(r"\D", "", num)

    def parse_date(date_str: str | None) -> datetime | None:
        if not date_str:
            return None
        for fmt in ("%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str, fmt)
            except Exception:
                continue
        return None

    def can_merge(existing: dict, new: dict) -> bool:
        """Return True if the two account records likely refer to the same account."""
        name1 = normalize_creditor_name(existing.get("name", "")).lower()
        name2 = normalize_creditor_name(new.get("name", "")).lower()
        if name1 != name2:
            return False

        num1 = sanitize_number(existing.get("account_number"))
        num2 = sanitize_number(new.get("account_number"))
        if num1 and num2 and num1[-4:] != num2[-4:]:
            return False
        if not num1 and not num2:
            status1 = (existing.get("status") or "").lower()
            status2 = (new.get("status") or "").lower()
            return status1 == status2

        return True

    for bureau, section in bureau_data.items():
        for acc in section.get("all_accounts", []):
            acc_copy = acc.copy()
            acc_copy.setdefault("bureaus", acc.get("bureaus", [bureau]))
            acc_copy.setdefault("categories", set(acc.get("categories", [])))

            merged = False
            for existing in all_accounts:
                if can_merge(existing, acc_copy):
                    existing["bureaus"].update(acc_copy.get("bureaus", []))
                    existing["categories"].update(acc_copy.get("categories", []))
                    for field in [
                        "action_tag",
                        "recommended_action",
                        "advisor_comment",
                        "status",
                        "utilization",
                        "dispute_type",
                        "goodwill_candidate",
                        "letter_type",
                        "custom_letter_note",
                    ]:
                        if not existing.get(field) and acc_copy.get(field):
                            existing[field] = acc_copy[field]
                    if acc_copy.get("duplicate_suspect"):
                        existing["duplicate_suspect"] = True
                    merged = True
                    break
            if not merged:
                acc_copy["bureaus"] = set(acc_copy.get("bureaus", []))
                acc_copy["categories"] = set(acc_copy.get("categories", []))
                all_accounts.append(acc_copy)

    # Additional de-duplication across bureaus using creditor name + bureau key
    deduped: dict[tuple[str, str], dict] = {}
    for acc in all_accounts:
        name_key = normalize_creditor_name(acc.get("name", ""))
        for b in acc.get("bureaus", []):
            key = (name_key, b)
            existing = deduped.get(key)
            if existing:
                existing["bureaus"].update(acc.get("bureaus", []))
                existing["categories"].update(acc.get("categories", []))
                for field in [
                    "action_tag",
                    "recommended_action",
                    "advisor_comment",
                    "status",
                    "utilization",
                    "dispute_type",
                    "goodwill_candidate",
                    "letter_type",
                    "custom_letter_note",
                ]:
                    if not existing.get(field) and acc.get(field):
                        existing[field] = acc[field]
                if acc.get("duplicate_suspect"):
                    existing["duplicate_suspect"] = True
            else:
                deduped[key] = acc

    all_accounts = list({id(v): v for v in deduped.values()}.values())

    has_dupes = any(acc.get("duplicate_suspect") for acc in all_accounts)

    sections_rows = {
        "problematic": [],
        "improve": [],
        "positive": [],
    }
    raw_notes = client_info.get("custom_dispute_notes", {}) or {}
    specific_notes, _ = analyze_custom_notes(
        raw_notes, [a.get("name", "") for a in all_accounts]
    )
    note_map = {normalize_creditor_name(k): v for k, v in specific_notes.items()}
    for acc in all_accounts:
        name = acc.get("name", "Unknown")
        advisor_comment = acc.get("advisor_comment", "")
        action_tag = acc.get("action_tag", "")
        recommended_action = acc.get("recommended_action") or (
            action_tag.replace("_", " ").title() if action_tag else None
        )
        personal_note = note_map.get(normalize_creditor_name(name))
        bureaus = ", ".join(sorted(acc.get("bureaus", [])))
        status = acc.get("reported_status") or acc.get("status") or ""
        utilization = acc.get("utilization")
        dispute_type = acc.get("dispute_type", "")
        categories = {c.lower() for c in acc.get("categories", [])}

        def get_group():
            util_pct = None
            if utilization:
                try:
                    util_pct = int(utilization.replace("%", ""))
                except Exception:
                    pass
            status_l = status.lower()
            if (
                "negative_accounts" in categories
                or any(
                    kw in status_l
                    for kw in (
                        "chargeoff",
                        "charge-off",
                        "charge off",
                        "collection",
                        "repossession",
                        "repos",
                        "delinquent",
                        "late payments",
                    )
                )
                or dispute_type
                or acc.get("goodwill_candidate")
            ):
                return "problematic"
            if (
                "open_accounts_with_issues" in categories
                or "high_utilization_accounts" in categories
                or (util_pct is not None and util_pct > 30)
            ):
                return "improve"
            return "positive"

        group = get_group()

        action_lines = [f"<strong class='account-title'>{html_utils.escape(name)}</strong> ({bureaus})"]
        clean_status = (status or "").strip()
        if clean_status:
            status_line = f"<strong>Status:</strong> {html_utils.escape(clean_status)}"
        else:
            status_line = "<strong>Status:</strong> No status available from the bureaus"
        action_lines.append(status_line)
        late = acc.get("late_payments")
        if isinstance(late, dict):
            parts = []
            for bureau, vals in late.items():
                sub = " ".join([f"{k}:{v}" for k, v in vals.items() if v])
                if sub:
                    parts.append(f"{bureau}: {sub}")
            if parts:
                action_lines.append("<em>Late history - " + "; ".join(parts) + "</em>")
        if recommended_action:
            action_lines.append(
                f"<strong>Strategist Action:</strong> {html_utils.escape(recommended_action)}"
            )

        letters = []
        needs_dispute = action_tag.lower() == "dispute"
        if needs_dispute:
            letters.append("Dispute")

        needs_goodwill = action_tag.lower() == "goodwill"
        if needs_goodwill:
            letters.append("Goodwill")

        if acc.get("letter_type") == "custom" or action_tag.lower() == "custom_letter":
            letters.append("Custom")

        if letters:
            action_lines.append(f"<strong>Letters Generated:</strong> {', '.join(letters)}")

        if dispute_type == "identity_theft":
            action_lines.append("⚠️ This account is reported as identity theft.")
        elif dispute_type == "unauthorized_or_unverified":
            action_lines.append("⚠️ This account doesn't look familiar and is being disputed.")
        elif dispute_type == "inaccurate_reporting":
            action_lines.append("⚠️ The information on this account appears incorrect.")

        # Custom letter note handled in letter summary above

        if utilization:
            try:
                percent = int(utilization.replace("%", ""))
                if percent > 30:
                    action_lines.append(
                        f"💳 You're using about {percent}% of your limit. Paying this down will help your score."
                    )
            except Exception:
                pass

        if not any(x.startswith(("📄", "⚠️", "💳")) for x in action_lines):
            if not advisor_comment:
                advisor_comment = random.choice(
                    [
                        "This account is in good standing and supports your credit profile.",
                        "Keep this account open and continue making on-time payments to strengthen your credit.",
                        "Avoid closing this account — older positive accounts help your score.",
                        "This account reflects positively on your report. Maintain low usage and regular activity.",
                    ]
                )
            action_lines.append(
                "✅ No immediate action needed — keep this account healthy."
            )

        if advisor_comment:
            action_lines.append(f"💬 <em>{advisor_comment}</em>")

        if personal_note:
            action_lines.append(f"📝 <em>{html_utils.escape(personal_note)}</em>")

        action_context = {
            "name": name,
            "bureaus": list(acc.get("bureaus", [])),
            "status": clean_status,
            "utilization": utilization,
            "dispute_type": dispute_type,
            "goodwill_candidate": acc.get("goodwill_candidate"),
            "categories": list(categories),
            "action_tag": action_tag,
            "recommended_action": recommended_action,
            "advisor_comment": advisor_comment,
        }
        action_sentence = generate_account_action(action_context)
        action_lines.append(
            f"<strong>Your Action:</strong> {html_utils.escape(action_sentence)}"
        )

        # Remove duplicate lines while preserving order
        deduped: list[str] = []
        seen: set[str] = set()
        for line in action_lines[2:]:
            if line not in seen:
                deduped.append(line)
                seen.add(line)

        action_html = "<br>".join(deduped)
        row_html = (
            f"<tr><td>{html_utils.escape(name)}</td>"
            f"<td>{html_utils.escape(bureaus)}</td>"
            f"<td>{html_utils.escape(clean_status) if clean_status else 'N/A'}</td>"
            f"<td>{action_html}</td></tr>"
        )

        sections_rows[group].append(row_html)

    html_intro = """
    <h2>What You Received</h2>
    <p>This package includes dispute letters for credit bureaus, goodwill letters for creditors, and a detailed breakdown of your credit report.</p>
    <p>Carefully review the summary below and follow the instructions for each account. Take action by printing and mailing the appropriate letters.</p>
    """

    duplicates_block = ""
    if has_dupes:
        duplicates_block = """
<div class='advisory'>
<h2>⚠️ Potential Duplicate Negative Reporting Detected</h2>
<p>We noticed that your report might contain duplicate negative entries for the same debt (e.g., reported both as Charge-Off and Collection). This situation is more complex and may require manual review and a personalized dispute strategy. We recommend that you contact us directly so we can assist you with a tailored approach to address this properly.</p>
</div>
"""

    def build_table(rows: list[str]) -> str:
        if not rows:
            return "<p>None</p>"
        header = (
            "<table class='account-table'>"
            "<tr><th>Account Name</th><th>Bureaus</th><th>Status</th><th>Action</th></tr>"
        )
        return header + "".join(rows) + "</table>"

    html_block = f"""
    <div class='category problematic'>
      <h2 class='category-title problematic-title'>🟥 Problematic Accounts to Remove</h2>
      {build_table(sections_rows['problematic'])}
    </div>
    <div class='category improve'>
      <h2 class='category-title improve-title'>🟡 Accounts to Improve</h2>
      {build_table(sections_rows['improve'])}
    </div>
    <div class='category positive'>
      <h2 class='category-title positive-title'>🟢 Positive Accounts to Maintain</h2>
      {build_table(sections_rows['positive'])}
    </div>
    """

    tips_block = """
    <h2>General Credit Tips</h2>
    <ul>
        <li>📆 Pay all bills on time — payment history is the most important factor in your credit score.</li>
        <li>📉 Keep your credit usage below 30%, ideally under 10%.</li>
        <li>🧾 Do not close old positive accounts — they help your average credit age.</li>
    </ul>
    """

    strategy_block = ""
    if strategy:
        items = []
        for rec in strategy.get("global_recommendations", []):
            items.append(f"<li>{html_utils.escape(rec)}</li>")
        account_tips = []
        for acc in strategy.get("accounts", []):
            tip = acc.get("recommendation") or acc.get("recommended_action")
            if tip:
                name = acc.get("name", "Account")
                account_tips.append(
                    f"<li><strong>{html_utils.escape(name)}:</strong> {html_utils.escape(tip)}</li>"
                )
        if items or account_tips:
            joined = "".join(items)
            extra = "".join(account_tips)
            strategy_block = (
                "<h2>Strategist Recommendations</h2><ul>" + joined + extra + "</ul>"
            )

    closing_block = (
        "<p><strong>You’re in control of your credit journey — "
        "every step brings you closer to financial freedom!</strong></p>"
        "<div class='support'>"
        "💬 Feeling overwhelmed? If any of this feels confusing or too much "
        "— you're not alone. We're here to help. Our team can take care of "
        "the whole process for you, including mailing the letters — just "
        "reach out and ask about our <strong>Done-For-You</strong> service."
        "</div>"
    )

    html = render_instruction_html(
        {
            "date": run_date,
            "client_name": client_name,
            "instructions": html_intro
            + duplicates_block
            + html_block
            + tips_block
            + strategy_block
            + closing_block,
            "is_identity_theft": is_identity_theft,
            "logo_base64": logo_base64,
        }
    )

    return html, all_accounts


def render_pdf_from_html(html: str, output_path: Path) -> Path:
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / "Start_Here - Instructions.pdf"
    render_html_to_pdf(html, filepath)
    return filepath


def save_json_output(all_accounts: list[dict], output_path: Path):
    def sanitize_for_json(data):
        if isinstance(data, dict):
            return {k: sanitize_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [sanitize_for_json(i) for i in data]
        elif isinstance(data, set):
            return list(data)
        else:
            return data

    sanitized_accounts = sanitize_for_json(all_accounts)
    with open(output_path / "instructions_context.json", "w") as f:
        json.dump({"accounts": sanitized_accounts}, f, indent=2)

