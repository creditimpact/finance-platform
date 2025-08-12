"""High-level orchestration routines for the credit repair pipeline.

ARCH: This module acts as the single entry point for coordinating the
intake, analysis, strategy generation, letter creation and finalization
steps of the credit repair workflow.  All core orchestration lives here;
``main.py`` only provides thin CLI wrappers.
"""

import os
import warnings
from pathlib import Path
from datetime import datetime
from shutil import copyfile
from typing import Any, Mapping

from backend.audit.audit import AuditLevel
from backend.core.logic.extract_info import extract_bureau_info_column_refined
from backend.core.logic.utils.pdf_ops import (
    convert_txts_to_pdfs,
    gather_supporting_docs_text,
)
from backend.core.logic.utils.report_sections import (
    filter_sections_by_bureau,
    extract_summary_from_sections,
)
from backend.core.logic.summary_classifier import classify_client_summary
from backend.core.logic.constants import StrategistFailureReason
from backend.analytics.analytics_tracker import save_analytics_snapshot
from backend.analytics.analytics.strategist_failures import tally_failure_reasons
from backend.core.email_sender import send_email_with_attachment
from backend.core.services.ai_client import AIClient
from backend.api.config import AppConfig, get_app_config
from backend.assets.paths import templates_path
from backend.core.models import (
    ClientInfo,
    ProofDocuments,
    BureauPayload,
    BureauAccount,
    Inquiry,
)


def process_client_intake(client_info, audit):
    """Prepare client intake information.

    Returns:
        tuple[str, dict, dict]: session id, structured summaries and raw notes.
    """
    from backend.api.session_manager import get_intake

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
        cls = classify_client_summary(struct, ai_client, client_info.get("state"))
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
    from backend.core.logic.upload_validator import is_safe_pdf, move_uploaded_file
    from backend.api.session_manager import update_session
    from backend.core.logic.analyze_report import (
        analyze_credit_report as analyze_report_logic,
    )
    from backend.core.logic.bootstrap import get_current_month

    uploaded_path = proofs_files.get("smartcredit_report")
    if not uploaded_path or not os.path.exists(uploaded_path):
        raise FileNotFoundError(
            "SmartCredit report file not found at path: " + str(uploaded_path)
        )

    pdf_path = move_uploaded_file(Path(uploaded_path), session_id)
    update_session(session_id, file_path=str(pdf_path))
    if not is_safe_pdf(pdf_path):
        raise ValueError("Uploaded file failed PDF safety checks.")

    print("[INFO] Extracting client info from report...")
    client_personal_info = extract_bureau_info_column_refined(
        pdf_path, ai_client=ai_client
    )
    client_info.update(client_personal_info.get("data", {}))
    log_messages.append("[INFO] Personal info extracted.")
    if audit.level == AuditLevel.VERBOSE:
        audit.log_step("personal_info_extracted", client_personal_info)

    print("[INFO] Analyzing report with GPT...")
    analyzed_json_path = Path("output/analyzed_report.json")
    sections = analyze_report_logic(
        pdf_path, analyzed_json_path, client_info, ai_client=ai_client
    )
    client_info.update(sections)
    log_messages.append("[INFO] Report analyzed.")
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
    log_messages.append(f"[INFO] Client folder created at: {today_folder}")
    if audit.level == AuditLevel.VERBOSE:
        audit.log_step("client_folder_created", {"path": str(today_folder)})

    for file in today_folder.glob("*.pdf"):
        file.unlink()
    for file in today_folder.glob("*_gpt_response.json"):
        file.unlink()

    original_pdf_copy = today_folder / "Original SmartCredit Report.pdf"
    copyfile(pdf_path, original_pdf_copy)
    log_messages.append("[INFO] Original report saved to client folder.")

    if analyzed_json_path.exists():
        copyfile(analyzed_json_path, today_folder / "analyzed_report.json")
        log_messages.append("[INFO] Analyzed report JSON saved.")

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
    from backend.core.logic.strategy_merger import merge_strategy_data
    from backend.core.logic.generate_strategy_report import StrategyGenerator

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
    from backend.core.logic.bootstrap import extract_all_accounts
    from backend.core.logic.letter_generator import generate_all_dispute_letters_with_ai
    from backend.core.logic.instructions_generator import generate_instruction_file
    from backend.core.logic.generate_goodwill_letters import generate_goodwill_letters
    from backend.core.logic.generate_custom_letters import generate_custom_letters

    print("[INFO] Generating dispute letters...")
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
    log_messages.append("[INFO] Dispute letters generated.")
    if audit.level is AuditLevel.VERBOSE:
        audit.log_step("dispute_letters_generated")

    if not is_identity_theft:
        print("[INFO] Generating goodwill letters...")
        generate_goodwill_letters(
            client_info, bureau_data, today_folder, audit, ai_client=ai_client
        )
        log_messages.append("[INFO] Goodwill letters generated.")
        if audit.level is AuditLevel.VERBOSE:
            audit.log_step("goodwill_letters_generated")
    else:
        print("[INFO] Identity theft case - skipping goodwill letters.")
        log_messages.append("[INFO] Goodwill letters skipped due to identity theft.")
        if audit.level is AuditLevel.VERBOSE:
            audit.log_step("goodwill_letters_skipped")

    all_accounts = extract_all_accounts(sections)
    for bureau in bureau_data:
        bureau_data[bureau]["all_accounts"] = all_accounts

    print("[INFO] Generating custom letters...")
    generate_custom_letters(
        client_info,
        bureau_data,
        today_folder,
        audit,
        log_messages=log_messages,
        ai_client=ai_client,
        wkhtmltopdf_path=app_config.wkhtmltopdf_path if app_config else None,
    )
    log_messages.append("[INFO] Custom letters generated.")
    if audit.level is AuditLevel.VERBOSE:
        audit.log_step("custom_letters_generated")

    print("[INFO] Generating instructions file for client...")
    generate_instruction_file(
        client_info,
        bureau_data,
        is_identity_theft,
        today_folder,
        strategy=strategy,
        ai_client=ai_client,
        wkhtmltopdf_path=app_config.wkhtmltopdf_path if app_config else None,
    )
    log_messages.append("[INFO] Instruction file created.")
    if audit.level is AuditLevel.VERBOSE:
        audit.log_step("instructions_generated")

    print("[INFO] Converting letters to PDF...")
    if os.getenv("DISABLE_PDF_RENDER", "").lower() not in ("1", "true", "yes"):
        convert_txts_to_pdfs(today_folder)
        log_messages.append("[INFO] All letters converted to PDF.")
        if audit.level is AuditLevel.VERBOSE:
            audit.log_step("letters_converted_to_pdf")
    else:
        print(
            "[INFO] PDF rendering disabled via DISABLE_PDF_RENDER – skipping conversion."
        )

    if is_identity_theft:
        print("[INFO] Adding FCRA rights PDF...")
        frca_source_path = templates_path("FTC_FCRA_605b.pdf")
        frca_target_path = today_folder / "Your Rights - FCRA.pdf"
        if os.path.exists(frca_source_path):
            copyfile(frca_source_path, frca_target_path)
            print(f"[INFO] FCRA rights PDF copied to: {frca_target_path}")
            log_messages.append("[INFO] FCRA document added.")
        else:
            print("[WARN] FCRA rights file not found!")
            log_messages.append("[WARN] FCRA file missing.")
            if audit.level is AuditLevel.VERBOSE:
                audit.log_step("fcra_file_missing")
    else:
        log_messages.append("[INFO] Identity theft not indicated - FCRA PDF skipped.")
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
    print("[INFO] Sending email with all documents to client...")
    output_files = [str(p) for p in today_folder.glob("*.pdf")]
    raw_name = (client_info.get("name") or "").strip()
    first_name = raw_name.split()[0] if raw_name else "Client"
    send_email_with_attachment(
        receiver_email=client_info["email"],
        subject="Your Credit Repair Package is Ready",
        body=f"""Hi {first_name},

WeÃ¢Â€Â™ve successfully completed your credit analysis and prepared your customized repair package Ã¢Â€Â" itÃ¢Â€Â™s attached to this email.

Ã°ÂŸÂ-Â‚ Inside your package:
- Dispute letters prepared for each credit bureau
- Goodwill letters (if applicable)
- Your full SmartCredit report
- A personalized instruction guide with legal backup
- Your official rights under the FCRA (Fair Credit Reporting Act)

Ã¢ÂœÂ... Please print, sign, and mail each dispute letter to the bureaus at their addresses (included in the letters), along with:
- A copy of your government-issued ID
- A utility bill with your current address
- (Optional) FTC Identity Theft Report if applicable

In your **instruction file**, you'll also find:
- A breakdown of which accounts are hurting your score the most
- Recommendations like adding authorized users (we can help you do this!)
- When and how to follow up with SmartCredit

If youÃ¢Â€Â™d like our team to help you with the next steps Ã¢Â€Â" including adding an authorized user, tracking disputes, or escalating Ã¢Â€Â" weÃ¢Â€Â™re just one click away.

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
    log_messages.append("[INFO] Email sent to client.")
    if audit.level is AuditLevel.VERBOSE:
        audit.log_step("email_sent", {"files": output_files})

    failure_counts = tally_failure_reasons(audit)
    save_analytics_snapshot(
        client_info,
        extract_summary_from_sections(sections),
        strategist_failures=failure_counts,
    )
    log_messages.append("[INFO] Analytics snapshot saved.")
    if audit.level is AuditLevel.VERBOSE:
        audit.log_step("analytics_saved", {"strategist_failures": failure_counts})

    print("\n[INFO] Credit Repair Process completed successfully!")
    print(f"[INFO] All output saved to: {today_folder}")
    log_messages.append("[INFO] Process completed successfully.")
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
        f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Client: {client_info.get('name', '')}",
        f"Address: {client_info.get('address', '')}",
        f"Goal: {client_info.get('goal', '')}",
        f"Treatment Type: {'Identity Theft' if is_identity_theft else 'Standard Dispute'}",
        f"Output folder: {output_folder}",
        "",
    ]

    with open(log_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(header + log_lines))
    print(f"[INFO] Log saved: {log_path}")


def run_credit_repair_process(
    client: ClientInfo,
    proofs: ProofDocuments,
    is_identity_theft: bool,
    *,
    app_config: AppConfig | None = None,
):
    """Execute the full credit repair pipeline for a single client.

    ``client`` and ``proofs`` should be instances of :class:`ClientInfo` and
    :class:`ProofDocuments` respectively. Passing plain dictionaries is
    deprecated and will be removed in a future release.
    """
    app_config = app_config or get_app_config()
    if isinstance(client, dict):  # pragma: no cover - backward compat
        client = ClientInfo.from_dict(client)
    if isinstance(proofs, dict):  # pragma: no cover - backward compat
        proofs = ProofDocuments.from_dict(proofs)
    client_info = client.to_dict()
    proofs_files = proofs.to_dict()
    log_messages: list[str] = []
    today_folder: Path | None = None
    pdf_path: Path | None = None
    session_id = client_info.get("session_id", "session")
    from backend.audit.audit import create_audit_logger
    from backend.core.services.ai_client import build_ai_client

    audit = create_audit_logger(session_id)
    ai_client = build_ai_client(app_config.ai)
    strategy = None

    try:
        print("\n[INFO] Starting Credit Repair Process (B2C Mode)...")
        log_messages.append("[INFO] Process started.")
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
        error_msg = f"[ERROR] Error: {str(e)}"
        print(error_msg)
        log_messages.append(error_msg)
        audit.log_error(error_msg)
        raise

    finally:
        save_log_file(client_info, is_identity_theft, today_folder, log_messages)
        if today_folder:
            audit.save(today_folder)
            if app_config.export_trace_file:
                from backend.audit.trace_exporter import (
                    export_trace_file,
                    export_trace_breakdown,
                )

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
            print(f"[INFO] Deleted uploaded PDF: {pdf_path}")
        except Exception as delete_error:  # pragma: no cover - best effort
            print(f"[WARN] Failed to delete uploaded PDF: {delete_error}")


def extract_problematic_accounts_from_report(
    file_path: str, session_id: str | None = None
) -> "BureauPayload":
    """Return problematic accounts extracted from the report for user review."""
    from backend.core.logic.upload_validator import is_safe_pdf, move_uploaded_file
    from backend.api.session_manager import update_session
    from backend.core.logic.analyze_report import (
        analyze_credit_report as analyze_report_logic,
    )

    session_id = session_id or "session"
    pdf_path = move_uploaded_file(Path(file_path), session_id)
    update_session(session_id, file_path=str(pdf_path))
    if not is_safe_pdf(pdf_path):
        raise ValueError("Uploaded file failed PDF safety checks.")

    analyzed_json_path = Path("output/analyzed_report.json")

    sections = analyze_report_logic(pdf_path, analyzed_json_path, {})

    return BureauPayload(
        disputes=[
            BureauAccount.from_dict(d) for d in sections.get("negative_accounts", [])
        ],
        goodwill=[
            BureauAccount.from_dict(d)
            for d in sections.get("open_accounts_with_issues", [])
        ],
        inquiries=[
            Inquiry.from_dict(d) for d in sections.get("unauthorized_inquiries", [])
        ],
    )


def extract_problematic_accounts_from_report_dict(
    file_path: str, session_id: str | None = None
) -> Mapping[str, Any]:
    """Deprecated adapter returning ``dict`` for old clients."""
    warnings.warn(
        "extract_problematic_accounts_from_report_dict is deprecated;"
        " use extract_problematic_accounts_from_report instead",
        DeprecationWarning,
        stacklevel=2,
    )
    payload = extract_problematic_accounts_from_report(file_path, session_id)
    return {
        "negative_accounts": [a.to_dict() for a in payload.disputes],
        "open_accounts_with_issues": [a.to_dict() for a in payload.goodwill],
        "unauthorized_inquiries": [i.to_dict() for i in payload.inquiries],
    }
