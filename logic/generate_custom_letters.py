import os
from openai import OpenAI
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
import pdfkit
from logic.utils import gather_supporting_docs

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

WKHTMLTOPDF_PATH = os.getenv("WKHTMLTOPDF_PATH", "wkhtmltopdf")
pdf_config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)

env = Environment(loader=FileSystemLoader("templates"))
template = env.get_template("general_letter_template.html")


def call_gpt_for_custom_letter(
    client_name: str,
    recipient_name: str,
    purpose: str,
    account_name: str,
    account_number: str,
    docs_text: str,
) -> str:
    docs_line = f"Supporting documents summary:\n{docs_text}" if docs_text else ""
    prompt = f"""
You are a professional credit repair assistant helping a client draft a formal custom letter. Write the body **in the first person** as if the client wrote it. Provide a clear 1-2 paragraph message with a polite tone and direct request for action.
Client name: {client_name}
Recipient: {recipient_name}
Purpose: {purpose}
Account: {account_name} {account_number}
{docs_line}
Respond only with the letter body.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    body = response.choices[0].message.content.strip()
    if body.startswith("```"):
        body = body.replace("```", "").strip()
    return body


def generate_custom_letter(account: dict, client_info: dict, output_path: Path, run_date: str | None = None) -> None:
    client_name = client_info.get("legal_name") or client_info.get("name", "Client")
    date_str = run_date or datetime.now().strftime("%B %d, %Y")
    recipient = account.get("name", "")
    note = account.get("custom_letter_note", "")
    if account.get("advisor_comment"):
        note += f"\nAdvisor: {account['advisor_comment']}"
    if account.get("action_tag"):
        note += f"\nStrategist tag: {account['action_tag']}"
    if account.get("recommended_action"):
        note += f"\nStrategist recommends: {account['recommended_action']}"
    acc_name = account.get("name", "")
    acc_number = account.get("account_number", "")
    session_id = client_info.get("session_id", "")

    docs_text, doc_names, _ = gather_supporting_docs(session_id)
    if docs_text:
        print(f"[📎] Including supplemental docs for custom letter to {recipient}.")

    body_paragraph = call_gpt_for_custom_letter(
        client_name,
        recipient,
        note,
        acc_name,
        acc_number,
        docs_text,
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
    pdfkit.from_string(html, str(full_path), configuration=pdf_config, options=options)
    print(f"[📝] Custom letter generated: {full_path}")

    response_path = output_path / f"{safe_recipient}_custom_gpt_response.txt"
    with open(response_path, "w", encoding="utf-8") as f:
        f.write(body_paragraph)


def generate_custom_letters(
    client_info: dict,
    bureau_data: dict,
    output_path: Path,
    run_date: str | None = None,
    log_messages: list[str] | None = None,
) -> None:
    if log_messages is None:
        log_messages = []
    for bureau, content in bureau_data.items():
        for acc in content.get("all_accounts", []):
            action = str(acc.get("action_tag") or acc.get("recommended_action") or "").lower()
            if acc.get("letter_type") == "custom" or action == "custom_letter":
                generate_custom_letter(acc, client_info, output_path, run_date)
            else:
                log_messages.append(
                    f"[{bureau}] No custom letter for '{acc.get('name')}' — not marked for custom correspondence"
                )
