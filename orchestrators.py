"""High-level orchestration routines for the credit repair pipeline.

ARCH: This module acts as the single entry point for coordinating the
intake, analysis, strategy generation, letter creation and finalization
steps of the credit repair workflow.  All core orchestration lives here;
``main.py`` only provides thin CLI wrappers.
"""

import os
from pathlib import Path
from datetime import datetime
from shutil import copyfile

from audit import AuditLevel
from logic.extract_info import extract_bureau_info_column_refined
from logic.utils.pdf_ops import convert_txts_to_pdfs, gather_supporting_docs_text
from logic.utils.report_sections import (
    filter_sections_by_bureau,
    extract_summary_from_sections,
)
from logic.summary_classifier import classify_client_summary
from logic.constants import StrategistFailureReason
from analytics_tracker import save_analytics_snapshot
from analytics.strategist_failures import tally_failure_reasons
from email_sender import send_email_with_attachment
from services.ai_client import AIClient
from config import AppConfig, get_app_config


def process_client_intake(client_info, audit):
    """Prepare client intake information.

    Returns:
        tuple[str, dict, dict]: session id, structured summaries and raw notes.
    """
    from session_manager import get_intake

    if "email" not in client_info or not client_info["email"]:
        raise ValueError("Client email is missing.")

    session_id = client_info.get("session_id", "session")
    audit.log_step("session_initialized", {"session_id": session_id})

    intake = get_intake(session_id) or {}
    structured = client_info.get("structured_summaries") or {}
    structured_map: dict[str, dict] = {}
    if isinstance(structured, list):
        for idx, item in enumerate(structured):
            if isinstance(item, dict):
                key = str(item.get("account_id") or idx)
                structured_map[key] = item
    elif isinstance(structured, dict):
        for key, item in structured.items():
            if isinstance(item, dict):
                structured_map[str(key)] = item

    raw_map = {
        str(r.get("account_id")): r.get("text")
        for r in intake.get("raw_explanations", [])
        if isinstance(r, dict)
    }
    return session_id, structured_map, raw_map


def classify_client_responses(
    structured_map, raw_map, client_info, audit, ai_client: AIClient
):
    """Classify client summaries for each account."""
    classification_map: dict[str, dict] = {}
    for acc_id, struct in structured_map.items():
        cls = classify_client_summary(
            struct, ai_client, client_info.get("state")
        )
        classification_map[acc_id] = cls
        audit.log_account(
            acc_id,
            {
                "stage": "explanation",
                "raw_explanation": raw_map.get(acc_id, ""),
                "structured_summary": struct,
                "classification": cls,
            },
        )
    return classification_map


def analyze_credit_report(
    proofs_files,
    session_id,
    client_info,
    audit,
    log_messages,
    ai_client: AIClient,
):
    """Ingest and analyze the client's credit report."""
    from logic.upload_validator import is_safe_pdf, move_uploaded_file
    from session_manager import update_session
    from logic.analyze_report import analyze_credit_report as analyze_report_logic
    from logic.bootstrap import get_current_month

    uploaded_path = proofs_files.get("smartcredit_report")
    if not uploaded_path or not os.path.exists(uploaded_path):
        raise FileNotFoundError(
            "SmartCredit report file not found at path: " + str(uploaded_path)
        )

    pdf_path = move_uploaded_file(Path(uploaded_path), session_id)
    update_session(session_id, file_path=str(pdf_path))
    if not is_safe_pdf(pdf_path):
        raise ValueError("Uploaded file failed PDF safety checks.")

    print("📄 Extracting client info from report...")
    client_personal_info = extract_bureau_info_column_refined(
        pdf_path, ai_client=ai_client
    )
    client_info.update(client_personal_info.get("data", {}))
    log_messages.append("📄 Personal info extracted.")
    if audit.level == AuditLevel.VERBOSE:
        audit.log_step("personal_info_extracted", client_personal_info)

    print("🔍 Analyzing report with GPT...")
    analyzed_json_path = Path("output/analyzed_report.json")
    sections = analyze_report_logic(
        pdf_path, analyzed_json_path, client_info, ai_client=ai_client
    )
    client_info.update(sections)
    log_messages.append("🔍 Report analyzed.")
    audit.log_step(
        "report_analyzed",
        {
            "negative_accounts": sections.get("negative_accounts", []),
            "open_accounts_with_issues": sections.get("open_accounts_with_issues", []),
            "unauthorized_inquiries": sections.get("unauthorized_inquiries", []),
        },
    )

    safe_name = (
        (client_info.get("name") or "Client").replace(" ", "_").replace("/", "_")
    )
    today_folder = Path(f"Clients/{get_current_month()}/{safe_name}_{session_id}")
    today_folder.mkdir(parents=True, exist_ok=True)
    log_messages.append(f"📁 Client folder created at: {today_folder}")
    if audit.level == AuditLevel.VERBOSE:
        audit.log_step("client_folder_created", {"path": str(today_folder)})

    for file in today_folder.glob("*.pdf"):
        file.unlink()
    for file in today_folder.glob("*_gpt_response.json"):
        file.unlink()

    original_pdf_copy = today_folder / "Original SmartCredit Report.pdf"
    copyfile(pdf_path, original_pdf_copy)
    log_messages.append("📁 Original report saved to client folder.")

    if analyzed_json_path.exists():
        copyfile(analyzed_json_path, today_folder / "analyzed_report.json")
        log_messages.append("📁 Analyzed report JSON saved.")

    detailed_logs = []
    bureau_data = {
        bureau: filter_sections_by_bureau(sections, bureau, detailed_logs)
        for bureau in ["Experian", "Equifax", "TransUnion"]
    }
    log_messages.extend(detailed_logs)
    if audit.level == AuditLevel.VERBOSE:
        audit.log_step("sections_split_by_bureau", bureau_data)

    return pdf_path, sections, bureau_data, today_folder


def generate_strategy_plan(
    client_info,
    bureau_data,
    classification_map,
    session_id,
    audit,
    log_messages,
    ai_client: AIClient,
):
    """Generate and merge the strategy plan."""
    from logic.strategy_merger import merge_strategy_data
    from logic.generate_strategy_report import StrategyGenerator

    docs_text = gather_supporting_docs_text(session_id)
    strat_gen = StrategyGenerator(ai_client=ai_client)
    audit.log_step(
        "strategist_invocation",
        {
            "client_info": client_info,
            "bureau_data": bureau_data,
            "classification_map": classification_map or {},
            "supporting_docs_text": docs_text,
        },
    )
    strategy = strat_gen.generate(
        client_info,
        bureau_data,
        docs_text,
        classification_map=classification_map,
        audit=audit,
    )
    if not strategy or not strategy.get("accounts"):
        audit.log_step(
            "strategist_failure",
            {"failure_reason": StrategistFailureReason.EMPTY_OUTPUT},
        )
    strat_gen.save_report(strategy, client_info, datetime.now().strftime("%Y-%m-%d"))
    audit.log_step("strategy_generated", strategy)

    merge_strategy_data(
        strategy, bureau_data, classification_map, audit, log_list=log_messages
    )
    audit.log_step("strategy_merged", bureau_data)
    for bureau, payload in bureau_data.items():
        for section, items in payload.items():
            if isinstance(items, list):
                for acc in items:
                    acc_id = acc.get("account_id") or acc.get("name")
                    audit.log_account(
                        acc_id,
                        {
                            "bureau": bureau,
                            "section": section,
                            "recommended_action": acc.get("recommended_action"),
                            "action_tag": acc.get("action_tag"),
                        },
                    )
    return strategy


def generate_letters(
    client_info,
    bureau_data,
    sections,
    today_folder,
    is_identity_theft,
    strategy,
    audit,
    log_messages,
    ai_client: AIClient,
    app_config: AppConfig | None = None,
):
    """Create all client letters and supporting files."""
    from logic.bootstrap import extract_all_accounts
    from logic.letter_generator import generate_all_dispute_letters_with_ai
    from logic.instructions_generator import generate_instruction_file
    from logic.generate_goodwill_letters import generate_goodwill_letters
    from logic.generate_custom_letters import generate_custom_letters

    print("📄 Generating dispute letters...")
    generate_all_dispute_letters_with_ai(
        client_info,
        bureau_data,
        today_folder,
        is_identity_theft,
        audit,
        log_messages=log_messages,
        ai_client=ai_client,
        rulebook_fallback_enabled=(
            app_config.rulebook_fallback_enabled if app_config else True
        ),
        wkhtmltopdf_path=app_config.wkhtmltopdf_path if app_config else None,
    )
    log_messages.append("📄 Dispute letters generated.")
    if audit.level is AuditLevel.VERBOSE:
        audit.log_step("dispute_letters_generated")

    if not is_identity_theft:
        print("💌 Generating goodwill letters...")
        generate_goodwill_letters(
            client_info, bureau_data, today_folder, audit, ai_client=ai_client
        )
        log_messages.append("💌 Goodwill letters generated.")
        if audit.level is AuditLevel.VERBOSE:
            audit.log_step("goodwill_letters_generated")
    else:
        print("🔒 Identity theft case - skipping goodwill letters.")
        log_messages.append("🚫 Goodwill letters skipped due to identity theft.")
        if audit.level is AuditLevel.VERBOSE:
            audit.log_step("goodwill_letters_skipped")

    all_accounts = extract_all_accounts(sections)
    for bureau in bureau_data:
        bureau_data[bureau]["all_accounts"] = all_accounts

    print("📝 Generating custom letters...")
    generate_custom_letters(
        client_info,
        bureau_data,
        today_folder,
        audit,
        log_messages=log_messages,
        ai_client=ai_client,
        wkhtmltopdf_path=app_config.wkhtmltopdf_path if app_config else None,
    )
    log_messages.append("📝 Custom letters generated.")
    if audit.level is AuditLevel.VERBOSE:
        audit.log_step("custom_letters_generated")

    print("📋 Generating instructions file for client...")
    generate_instruction_file(
        client_info,
        bureau_data,
        is_identity_theft,
        today_folder,
        strategy=strategy,
        ai_client=ai_client,
        wkhtmltopdf_path=app_config.wkhtmltopdf_path if app_config else None,
    )
    log_messages.append("📋 Instruction file created.")
    if audit.level is AuditLevel.VERBOSE:
        audit.log_step("instructions_generated")

    print("🌀 Converting letters to PDF...")
    convert_txts_to_pdfs(today_folder)
    log_messages.append("🌀 All letters converted to PDF.")
    if audit.level is AuditLevel.VERBOSE:
        audit.log_step("letters_converted_to_pdf")

    if is_identity_theft:
        print("📎 Adding FCRA rights PDF...")
        frca_source_path = "templates/FTC_FCRA_605b.pdf"
        frca_target_path = today_folder / "Your Rights - FCRA.pdf"
        if os.path.exists(frca_source_path):
            copyfile(frca_source_path, frca_target_path)
            print(f"📎 FCRA rights PDF copied to: {frca_target_path}")
            log_messages.append("📎 FCRA document added.")
        else:
            print("⚠️ FCRA rights file not found!")
            log_messages.append("⚠️ FCRA file missing.")
            if audit.level is AuditLevel.VERBOSE:
                audit.log_step("fcra_file_missing")
    else:
        log_messages.append("ℹ️ Identity theft not indicated — FCRA PDF skipped.")
        if audit.level is AuditLevel.VERBOSE:
            audit.log_step("fcra_skipped")


def finalize_outputs(
    client_info,
    today_folder,
    sections,
    audit,
    log_messages,
    app_config: AppConfig,
):
    """Send final outputs to the client and record analytics."""
    print("📧 Sending email with all documents to client...")
    output_files = [str(p) for p in today_folder.glob("*.pdf")]
    raw_name = (client_info.get("name") or "").strip()
    first_name = raw_name.split()[0] if raw_name else "Client"
    send_email_with_attachment(
        receiver_email=client_info["email"],
        subject="Your Credit Repair Package is Ready",
        body=f"""Hi {first_name},

We’ve successfully completed your credit analysis and prepared your customized repair package — it’s attached to this email.

🗂 Inside your package:
- Dispute letters prepared for each credit bureau
- Goodwill letters (if applicable)
- Your full SmartCredit report
- A personalized instruction guide with legal backup
- Your official rights under the FCRA (Fair Credit Reporting Act)

✅ Please print, sign, and mail each dispute letter to the bureaus at their addresses (included in the letters), along with:
- A copy of your government-issued ID
- A utility bill with your current address
- (Optional) FTC Identity Theft Report if applicable

In your **instruction file**, you'll also find:
- A breakdown of which accounts are hurting your score the most
- Recommendations like adding authorized users (we can help you do this!)
- When and how to follow up with SmartCredit

If you’d like our team to help you with the next steps — including adding an authorized user, tracking disputes, or escalating — we’re just one click away.

We're proud to support you on your journey to financial freedom.

Best regards,
**CREDIT IMPACT**
""",
        files=output_files,
        smtp_server=app_config.smtp_server,
        smtp_port=app_config.smtp_port,
        sender_email=app_config.smtp_username,
        sender_password=app_config.smtp_password,
    )
    log_messages.append("📧 Email sent to client.")
    if audit.level is AuditLevel.VERBOSE:
        audit.log_step("email_sent", {"files": output_files})

    failure_counts = tally_failure_reasons(audit)
    save_analytics_snapshot(
        client_info,
        extract_summary_from_sections(sections),
        strategist_failures=failure_counts,
    )
    log_messages.append("📊 Analytics snapshot saved.")
    if audit.level is AuditLevel.VERBOSE:
        audit.log_step("analytics_saved", {"strategist_failures": failure_counts})

    print("\n🎯 Credit Repair Process completed successfully!")
    print(f"📂 All output saved to: {today_folder}")
    log_messages.append("🎯 Process completed successfully.")
    audit.log_step("process_completed")


def save_log_file(client_info, is_identity_theft, output_folder, log_lines):
    """Persist a human-readable log of pipeline activity."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    client_name = client_info.get("name", "Unknown").replace(" ", "_")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_filename = f"{timestamp}_{client_name}.txt"
    log_path = logs_dir / log_filename

    header = [
        f"🕒 Run time: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"👤 Client: {client_info.get('name', '')}",
        f"🏠 Address: {client_info.get('address', '')}",
        f"🎯 Goal: {client_info.get('goal', '')}",
        f"🛠️ Treatment Type: {'Identity Theft' if is_identity_theft else 'Standard Dispute'}",
        f"📁 Output folder: {output_folder}",
        "",
    ]

    with open(log_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(header + log_lines))
    print(f"[📝] Log saved: {log_path}")


def run_credit_repair_process(
    client_info,
    proofs_files,
    is_identity_theft,
    *,
    app_config: AppConfig | None = None,
):
    """Execute the full credit repair pipeline for a single client."""
    app_config = app_config or get_app_config()
    log_messages: list[str] = []
    today_folder: Path | None = None
    pdf_path: Path | None = None
    session_id = client_info.get("session_id", "session")
    from audit import create_audit_logger
    from services.ai_client import build_ai_client

    audit = create_audit_logger(session_id)
    ai_client = build_ai_client(app_config.ai)
    strategy = None

    try:
        print("\n✅ Starting Credit Repair Process (B2C Mode)...")
        log_messages.append("✅ Process started.")
        audit.log_step("process_started", {"is_identity_theft": is_identity_theft})

        session_id, structured_map, raw_map = process_client_intake(client_info, audit)
        classification_map = classify_client_responses(
            structured_map, raw_map, client_info, audit, ai_client
        )
        pdf_path, sections, bureau_data, today_folder = analyze_credit_report(
            proofs_files, session_id, client_info, audit, log_messages, ai_client
        )
        strategy = generate_strategy_plan(
            client_info,
            bureau_data,
            classification_map,
            session_id,
            audit,
            log_messages,
            ai_client,
        )
        generate_letters(
            client_info,
            bureau_data,
            sections,
            today_folder,
            is_identity_theft,
            strategy,
            audit,
            log_messages,
            ai_client,
            app_config,
        )
        finalize_outputs(
            client_info, today_folder, sections, audit, log_messages, app_config
        )

    except Exception as e:  # pragma: no cover - surface for higher-level handling
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        log_messages.append(error_msg)
        audit.log_error(error_msg)
        raise

    finally:
        save_log_file(client_info, is_identity_theft, today_folder, log_messages)
        if today_folder:
            audit.save(today_folder)
            if app_config.export_trace_file:
                from trace_exporter import export_trace_file, export_trace_breakdown

                export_trace_file(audit, session_id)
                export_trace_breakdown(
                    audit,
                    strategy,
                    (
                        strategy.get("accounts")
                        if isinstance(strategy, dict)
                        else getattr(strategy, "accounts", None)
                    ),
                    Path("client_output"),
                )
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
                print(f"[🧹] Deleted uploaded PDF: {pdf_path}")
            except Exception as delete_error:  # pragma: no cover - best effort
                print(f"[⚠️] Failed to delete uploaded PDF: {delete_error}")


def extract_problematic_accounts_from_report(
    file_path: str, session_id: str | None = None
) -> dict:
    """Return problematic accounts extracted from the report for user review."""
    from logic.upload_validator import is_safe_pdf, move_uploaded_file
    from session_manager import update_session

    session_id = session_id or "session"
    pdf_path = move_uploaded_file(Path(file_path), session_id)
    update_session(session_id, file_path=str(pdf_path))
    if not is_safe_pdf(pdf_path):
        raise ValueError("Uploaded file failed PDF safety checks.")

    analyzed_json_path = Path("output/analyzed_report.json")
    from logic.analyze_report import analyze_credit_report as analyze_report_logic

    sections = analyze_report_logic(pdf_path, analyzed_json_path, {})

    return {
        "negative_accounts": sections.get("negative_accounts", []),
        "open_accounts_with_issues": sections.get("open_accounts_with_issues", []),
        "unauthorized_inquiries": sections.get("unauthorized_inquiries", []),
    }
