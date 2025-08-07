"""HTML rendering utilities for instruction generation.

This module consumes the structured context produced by
``instruction_data_preparation`` and builds the final HTML used
for PDF rendering.
"""

from __future__ import annotations

import html as html_utils
import random
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader("templates"))


def render_instruction_html(context: dict) -> str:
    """Render the Jinja2 template with the provided context."""
    template = env.get_template("instruction_template.html")
    return template.render(**context)


def build_instruction_html(context: dict) -> str:
    """Build the full instruction HTML string from prepared data."""
    sections = context.get("sections", {})

    html_intro = """
    <h2>What You Received</h2>
    <p>This package includes dispute letters for credit bureaus, goodwill letters for creditors, and a detailed breakdown of your credit report.</p>
    <p>Carefully review the summary below and follow the instructions for each account. Take action by printing and mailing the appropriate letters.</p>
    """

    duplicates_block = ""
    if context.get("has_duplicates"):
        duplicates_block = """
<div class='advisory'>
<h2>⚠️ Potential Duplicate Negative Reporting Detected</h2>
<p>We noticed that your report might contain duplicate negative entries for the same debt (e.g., reported both as Charge-Off and Collection). This situation is more complex and may require manual review and a personalized dispute strategy. We recommend that you contact us directly so we can assist you with a tailored approach to address this properly.</p>
</div>
"""

    def build_account_lines(acc: dict) -> list[str]:
        name = acc.get("name", "Unknown")
        bureaus = ", ".join(sorted(acc.get("bureaus", [])))
        status = acc.get("status") or ""
        action_lines = [
            f"<strong class='account-title'>{html_utils.escape(name)}</strong> ({bureaus})",
        ]
        clean_status = status.strip()
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

        if acc.get("recommended_action"):
            action_lines.append(
                f"<strong>Strategist Action:</strong> {html_utils.escape(acc['recommended_action'])}"
            )

        letters = acc.get("letters", [])
        if letters:
            action_lines.append(
                f"<strong>Letters Generated:</strong> {', '.join(letters)}"
            )

        dispute_type = acc.get("dispute_type")
        if dispute_type == "identity_theft":
            action_lines.append("⚠️ This account is reported as identity theft.")
        elif dispute_type == "unauthorized_or_unverified":
            action_lines.append("⚠️ This account doesn't look familiar and is being disputed.")
        elif dispute_type == "inaccurate_reporting":
            action_lines.append("⚠️ The information on this account appears incorrect.")

        utilization = acc.get("utilization")
        if utilization:
            try:
                percent = int(utilization.replace("%", ""))
                if percent > 30:
                    action_lines.append(
                        f"💳 You're using about {percent}% of your limit. Paying this down will help your score."
                    )
            except Exception:
                pass

        if not any(x.startswith(("📄", "⚠️", "💳")) for x in action_lines) and not acc.get("advisor_comment"):
            acc["advisor_comment"] = random.choice(
                [
                    "This account is in good standing and supports your credit profile.",
                    "Keep this account open and continue making on-time payments to strengthen your credit.",
                    "Avoid closing this account — older positive accounts help your score.",
                    "This account reflects positively on your report. Maintain low usage and regular activity.",
                ]
            )

        if acc.get("advisor_comment"):
            action_lines.append(f"💬 <em>{html_utils.escape(acc['advisor_comment'])}</em>")

        if acc.get("personal_note"):
            action_lines.append(f"📝 <em>{html_utils.escape(acc['personal_note'])}</em>")

        action_lines.append(
            f"<strong>Your Action:</strong> {html_utils.escape(acc.get('action_sentence', ''))}"
        )

        # Remove duplicate lines while preserving order
        deduped: list[str] = []
        seen: set[str] = set()
        for line in action_lines[2:]:
            if line not in seen:
                deduped.append(line)
                seen.add(line)

        return [action_lines[0], action_lines[1], *deduped]

    def build_table(accounts: list[dict]) -> str:
        if not accounts:
            return "<p>None</p>"
        rows = []
        for acc in accounts:
            lines = build_account_lines(acc)
            action_html = "<br>".join(lines[2:])
            rows.append(
                f"<tr><td>{html_utils.escape(acc['name'])}</td>"
                f"<td>{html_utils.escape(', '.join(acc.get('bureaus', [])))}</td>"
                f"<td>{html_utils.escape(acc.get('status') or '') or 'N/A'}</td>"
                f"<td>{action_html}</td></tr>"
            )
        header = (
            "<table class='account-table'>"
            "<tr><th>Account Name</th><th>Bureaus</th><th>Status</th><th>Action</th></tr>"
        )
        return header + "".join(rows) + "</table>"

    html_block = f"""
    <div class='category problematic'>
      <h2 class='category-title problematic-title'>🟥 Problematic Accounts to Remove</h2>
      {build_table(sections.get('problematic', []))}
    </div>
    <div class='category improve'>
      <h2 class='category-title improve-title'>🟡 Accounts to Improve</h2>
      {build_table(sections.get('improve', []))}
    </div>
    <div class='category positive'>
      <h2 class='category-title positive-title'>🟢 Positive Accounts to Maintain</h2>
      {build_table(sections.get('positive', []))}
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
    strategy = context.get("strategy")
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

    final_html = render_instruction_html(
        {
            "date": context.get("date"),
            "client_name": context.get("client_name"),
            "instructions": html_intro
            + duplicates_block
            + html_block
            + tips_block
            + strategy_block
            + closing_block,
            "is_identity_theft": context.get("is_identity_theft"),
            "logo_base64": context.get("logo_base64"),
        }
    )

    return final_html
