"""High-level orchestration routines for the credit repair pipeline.

ARCH: This module acts as the single entry point for coordinating the
intake, analysis, strategy generation, letter creation and finalization
steps of the credit repair workflow.  All core orchestration lives here;
``main.py`` only provides thin CLI wrappers.
"""

import os
import random
import time
import warnings
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from typing import Any, Mapping

import tactical
from backend.analytics.analytics.strategist_failures import tally_failure_reasons
from backend.analytics.analytics_tracker import emit_counter, save_analytics_snapshot
from backend.api.config import (
    ENABLE_FIELD_POPULATION,
    ENABLE_PLANNER,
    ENABLE_PLANNER_PIPELINE,
    FIELD_POPULATION_CANARY_PERCENT,
    PLANNER_CANARY_PERCENT,
    PLANNER_PIPELINE_CANARY_PERCENT,
    AppConfig,
    env_bool,
    get_app_config,
)
from backend.api.session_manager import update_session
from backend.assets.paths import templates_path
from backend.audit.audit import AuditLevel
from backend.core.email_sender import send_email_with_attachment
from backend.core.letters.field_population import apply_field_fillers
from backend.core.logic.compliance.constants import StrategistFailureReason
from backend.core.logic.report_analysis.extract_info import (
    extract_bureau_info_column_refined,
)
from backend.core.logic.strategy.normalizer_2_5 import normalize_and_tag
from backend.core.logic.strategy.summary_classifier import (
    RULES_VERSION,
    ClassificationRecord,
    classify_client_summaries,
    summary_hash,
)
from backend.core.logic.utils.pdf_ops import (
    convert_txts_to_pdfs,
    gather_supporting_docs_text,
)
from backend.core.logic.utils.report_sections import (
    extract_summary_from_sections,
    filter_sections_by_bureau,
)
from backend.core.models import (
    Account,
    BureauAccount,
    BureauPayload,
    ClientInfo,
    Inquiry,
    ProofDocuments,
)
from backend.core.services.ai_client import AIClient, get_ai_client, _StubAIClient
from backend.policy.policy_loader import load_rulebook
from planner import plan_next_step


def plan_and_generate_letters(session: dict, action_tags: list[str]) -> list[str]:
    """Optionally run the planner before generating letters.

    The planner pipeline is enabled when ``ENABLE_PLANNER_PIPELINE`` is true and
    a random draw for each account is below ``PLANNER_PIPELINE_CANARY_PERCENT``.
    For accounts outside this canary slice the planner is bypassed and the
    legacy router order is used.  When the pipeline is enabled, the planner
    executes only if ``ENABLE_PLANNER`` is true and the session passes the
    ``PLANNER_CANARY_PERCENT`` gate.  Otherwise the tactical pipeline runs with
    the original ``action_tags`` to preserve legacy behavior.

    Args:
        session: Session context passed to the tactical layer.
        action_tags: Proposed tags for this run.

    Returns:
        The tags passed to ``tactical.generate_letters``.
    """

    use_pipeline = ENABLE_PLANNER_PIPELINE
    pipeline_tags: list[str] = []
    legacy_tags: list[str] = []
    if use_pipeline:
        for tag in action_tags:
            if (
                PLANNER_PIPELINE_CANARY_PERCENT >= 100
                or random.random() < PLANNER_PIPELINE_CANARY_PERCENT / 100
            ):
                pipeline_tags.append(tag)
            else:
                legacy_tags.append(tag)
    else:
        legacy_tags = list(action_tags)

    use_planner = ENABLE_PLANNER
    if use_planner and PLANNER_CANARY_PERCENT < 100:
        if random.random() >= PLANNER_CANARY_PERCENT / 100:
            use_planner = False

    allowed: list[str] = []
    if pipeline_tags:
        planned = (
            plan_next_step(session, pipeline_tags) if use_planner else pipeline_tags
        )
        allowed.extend(planned)
    if legacy_tags:
        allowed.extend(legacy_tags)

    tactical.generate_letters(session, allowed)
    return allowed


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
    """Classify client summaries for each account.

    Results are cached in the session store keyed by a hash of the structured
    summary for each account.  Subsequent calls with the same summary skip the
    expensive classification step and reuse the stored data.
    """
    from backend.api.session_manager import get_session, update_session

    classification_map: dict[str, ClassificationRecord] = {}
    session_id = client_info.get("session_id")
    session = get_session(session_id or "") or {}
    cache = session.get("summary_classifications", {}) if session_id else {}
    state = client_info.get("state")

    updated = False
    to_process: list[tuple[str, dict, str]] = []
    for acc_id, struct in structured_map.items():
        struct_hash = summary_hash(struct)
        cached = cache.get(acc_id) if isinstance(cache, dict) else None
        if (
            cached
            and cached.get("summary_hash") == struct_hash
            and cached.get("state") == state
            and cached.get("rules_version") == RULES_VERSION
        ):
            cls = cached.get("classification", {})
            classification_map[acc_id] = ClassificationRecord(
                summary=struct,
                classification=cls,
                summary_hash=struct_hash,
                state=state,
                rules_version=RULES_VERSION,
            )
        else:
            enriched = dict(struct)
            enriched.setdefault("account_id", acc_id)
            to_process.append((acc_id, enriched, struct_hash))

    for i in range(0, len(to_process), 10):
        batch = to_process[i : i + 10]
        summaries = [item[1] for item in batch]
        batch_results = classify_client_summaries(
            summaries,
            ai_client,
            client_info.get("state"),
            session_id=session_id,
        )
        for acc_id, _summary, struct_hash in batch:
            cls = batch_results.get(acc_id, {})
            classification_map[acc_id] = ClassificationRecord(
                summary=_summary,
                classification=cls,
                summary_hash=struct_hash,
                state=state,
                rules_version=RULES_VERSION,
            )
            if session_id:
                cache[acc_id] = {
                    "summary_hash": struct_hash,
                    "classified_at": time.time(),
                    "classification": cls,
                    "state": state,
                    "rules_version": RULES_VERSION,
                }
                updated = True

    for acc_id, struct in structured_map.items():
        record = classification_map.get(acc_id)
        audit.log_account(
            acc_id,
            {
                "stage": "explanation",
                "raw_explanation": raw_map.get(acc_id, ""),
                "structured_summary": struct,
                "classification": record.classification if record else {},
            },
        )

    if session_id and updated:
        update_session(session_id, summary_classifications=cache)
    return classification_map


def analyze_credit_report(
    proofs_files,
    session_id,
    client_info,
    audit,
    log_messages,
    ai_client: AIClient | None = None,
):
    """Ingest and analyze the client's credit report."""
    from backend.api.session_manager import update_session
    from backend.core.logic.compliance.upload_validator import (
        is_safe_pdf,
        move_uploaded_file,
    )
    from backend.core.logic.report_analysis.analyze_report import (
        analyze_credit_report as analyze_report_logic,
    )
    from backend.core.logic.utils.bootstrap import get_current_month

    uploaded_path = proofs_files.get("smartcredit_report")
    if not uploaded_path or not os.path.exists(uploaded_path):
        raise FileNotFoundError(
            "SmartCredit report file not found at path: " + str(uploaded_path)
        )

    ai_client = ai_client or get_ai_client()
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
        pdf_path,
        analyzed_json_path,
        client_info,
        ai_client=ai_client,
        request_id=session_id,
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
    stage_2_5_data,
    session_id,
    audit,
    log_messages,
    ai_client: AIClient,
):
    """Generate and merge the strategy plan."""
    from backend.core.logic.strategy.generate_strategy_report import StrategyGenerator
    from backend.core.logic.strategy.strategy_merger import merge_strategy_data

    docs_text = gather_supporting_docs_text(session_id)
    strat_gen = StrategyGenerator(ai_client=ai_client)
    audit.log_step(
        "strategist_invocation",
        {
            "client_info": client_info,
            "bureau_data": bureau_data,
            "classification_map": {
                k: v.classification for k, v in (classification_map or {}).items()
            },
            "supporting_docs_text": docs_text,
        },
    )
    strategy = strat_gen.generate(
        client_info,
        bureau_data,
        docs_text,
        classification_map={
            k: v.classification for k, v in (classification_map or {}).items()
        },
        stage_2_5_data=stage_2_5_data,
        audit=audit,
    )
    if not strategy or not strategy.get("accounts"):
        audit.log_step(
            "strategist_failure",
            {"failure_reason": StrategistFailureReason.EMPTY_OUTPUT},
        )
    strat_gen.save_report(
        strategy,
        client_info,
        datetime.now().strftime("%Y-%m-%d"),
        stage_2_5_data=stage_2_5_data,
    )
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
    classification_map,
    ai_client: AIClient,
    app_config: AppConfig | None = None,
):
    """Create all client letters and supporting files."""
    from backend.core.logic.letters.generate_custom_letters import (
        generate_custom_letters,
    )
    from backend.core.logic.letters.generate_goodwill_letters import (
        generate_goodwill_letters,
    )
    from backend.core.logic.letters.letter_generator import (
        generate_all_dispute_letters_with_ai,
    )
    from backend.core.logic.rendering.instructions_generator import (
        generate_instruction_file,
    )
    from backend.core.logic.utils.bootstrap import extract_all_accounts

    print("[INFO] Generating dispute letters...")
    generate_all_dispute_letters_with_ai(
        client_info,
        bureau_data,
        today_folder,
        is_identity_theft,
        audit,
        classification_map=classification_map,
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
            client_info,
            bureau_data,
            today_folder,
            audit,
            ai_client=ai_client,
            classification_map=classification_map,
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
        classification_map=classification_map,
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
        os.environ["SESSION_ID"] = session_id
        classification_map = classify_client_responses(
            structured_map, raw_map, client_info, audit, ai_client
        )
        rulebook = load_rulebook()
        pdf_path, sections, bureau_data, today_folder = analyze_credit_report(
            proofs_files, session_id, client_info, audit, log_messages, ai_client
        )
        try:
            from services.outcome_ingestion.ingest_report import (
                ingest_report as ingest_outcome_report,
            )

            ingest_outcome_report(None, bureau_data)
        except Exception:
            pass
        tri_merge_map: dict[str, dict[str, Any]] = {}
        if env_bool("ENABLE_TRI_MERGE", False):
            from backend.api.session_manager import get_session
            from backend.core.logic.report_analysis.tri_merge import (
                compute_mismatches,
                normalize_and_match,
            )
            from backend.core.logic.report_analysis.tri_merge_models import Tradeline

            tradelines: list[Tradeline] = []
            for bureau, payload in bureau_data.items():
                for section, items in payload.items():
                    if section == "inquiries" or not isinstance(items, list):
                        continue
                    for acc in items:
                        tradelines.append(
                            Tradeline(
                                creditor=str(acc.get("name") or ""),
                                bureau=bureau,
                                account_number=acc.get("account_number"),
                                data=acc,
                            )
                        )
            _start = time.perf_counter()
            families = normalize_and_match(tradelines)
            emit_counter(
                "tri_merge.process_time_ms", (time.perf_counter() - _start) * 1000
            )
            compute_mismatches(families)
            tri_session = get_session(session_id) if session_id else None
            tri_evidence = (
                (tri_session.get("tri_merge") or {}).get("evidence", {})
                if tri_session
                else {}
            )
            for fam in families:
                family_id = getattr(fam, "family_id", None)
                mismatch_types = [m.field for m in getattr(fam, "mismatches", [])]
                evidence_id = family_id
                evidence = tri_evidence.get(evidence_id)
                for tl in fam.tradelines.values():
                    acc_id = str(tl.data.get("account_id") or "")
                    if acc_id and family_id:
                        tri_merge_map[acc_id] = {
                            "family_id": family_id,
                            "mismatch_types": mismatch_types,
                            "evidence_snapshot_id": evidence_id,
                        }
                        if evidence:
                            tri_merge_map[acc_id]["evidence"] = evidence
        facts_map: dict[str, dict[str, Any]] = {}
        for key in (
            "negative_accounts",
            "open_accounts_with_issues",
            "high_utilization_accounts",
            "positive_accounts",
            "all_accounts",
        ):
            for acc in sections.get(key, []):
                acc_id = str(acc.get("account_id") or "")
                if acc_id:
                    tri_info = tri_merge_map.get(acc_id)
                    if tri_info:
                        acc["tri_merge"] = tri_info
                    facts_map[acc_id] = acc
        stage_2_5: dict[str, Any] = {}
        for acc_id in set(facts_map) | set(classification_map):
            record = classification_map.get(acc_id)
            account_cls = {**record.summary, **record.classification} if record else {}
            stage_2_5[acc_id] = normalize_and_tag(
                account_cls, facts_map.get(acc_id, {}), rulebook, account_id=acc_id
            )
        if session_id:
            update_session(session_id, stage_2_5=stage_2_5)
        from backend.core.letters import router as letters_router

        for acc_id, acc_ctx in stage_2_5.items():
            tag = acc_ctx.get("action_tag")
            decision = letters_router.select_template(tag, acc_ctx, phase="candidate")
            emit_counter(
                "router.candidate_selected",
                {"tag": tag, "template": decision.template_path},
            )
        strategy = generate_strategy_plan(
            client_info,
            bureau_data,
            classification_map,
            stage_2_5,
            session_id,
            audit,
            log_messages,
            ai_client,
        )
        session_ctx = {
            "session_id": session_id,
            "client_info": client_info,
            "bureau_data": bureau_data,
            "sections": sections,
            "today_folder": today_folder,
            "is_identity_theft": is_identity_theft,
            "strategy": strategy,
            "audit": audit,
            "log_messages": log_messages,
            "classification_map": classification_map,
            "ai_client": ai_client,
            "app_config": app_config,
        }
        action_tags = [
            ctx.get("action_tag") for ctx in stage_2_5.values() if ctx.get("action_tag")
        ]
        plan_and_generate_letters(session_ctx, action_tags)
        strategy_accounts = {
            str(acc.get("account_id")): acc for acc in strategy.get("accounts", [])
        }
        for acc_id, acc_ctx in stage_2_5.items():
            tag = acc_ctx.get("action_tag")
            acc_strat = strategy_accounts.get(acc_id, {})
            do_population = ENABLE_FIELD_POPULATION
            if do_population and FIELD_POPULATION_CANARY_PERCENT < 100:
                do_population = random.random() < FIELD_POPULATION_CANARY_PERCENT / 100
            if do_population:
                apply_field_fillers(acc_ctx, strategy=acc_strat, profile=client_info)
                if tag:
                    for _ in acc_ctx.get("missing_fields", []):
                        emit_counter(
                            "finalize.missing_fields_after_population", {"tag": tag}
                        )
            letters_router.select_template(tag, acc_ctx, phase="finalize")
        if session_id:
            update_session(session_id, stage_2_5=stage_2_5)
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
                    export_trace_breakdown,
                    export_trace_file,
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
    from backend.api.session_manager import update_session
    from backend.core.logic.compliance.upload_validator import (
        is_safe_pdf,
        move_uploaded_file,
    )
    from backend.core.logic.report_analysis.analyze_report import (
        analyze_credit_report as analyze_report_logic,
    )

    session_id = session_id or "session"
    pdf_path = move_uploaded_file(Path(file_path), session_id)
    update_session(session_id, file_path=str(pdf_path))
    if not is_safe_pdf(pdf_path):
        raise ValueError("Uploaded file failed PDF safety checks.")

    analyzed_json_path = Path("output/analyzed_report.json")

    ai_client = get_ai_client()
    run_ai = not isinstance(ai_client, _StubAIClient)
    sections = analyze_report_logic(
        pdf_path,
        analyzed_json_path,
        {},
        ai_client=ai_client if run_ai else None,
        run_ai=run_ai,
        request_id=session_id,
    )
    sections.setdefault("negative_accounts", [])
    sections.setdefault("open_accounts_with_issues", [])
    update_session(session_id, status="awaiting_user_explanations")

    return BureauPayload(
        disputes=[
            BureauAccount.from_dict(d) for d in sections.get("negative_accounts", [])
        ],
        goodwill=[
            BureauAccount.from_dict(d)
            for d in sections.get("open_accounts_with_issues", [])
        ],
        inquiries=[
            Inquiry.from_dict(d)
            for d in sections.get(
                "unauthorized_inquiries", sections.get("inquiries", [])
            )
        ],
        high_utilization=[
            Account.from_dict(d) for d in sections.get("high_utilization_accounts", [])
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
