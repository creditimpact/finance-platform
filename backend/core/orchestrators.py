"""High-level orchestration routines for the credit repair pipeline.

ARCH: This module acts as the single entry point for coordinating the
intake, analysis, strategy generation, letter creation and finalization
steps of the credit repair workflow.  All core orchestration lives here;
``main.py`` only provides thin CLI wrappers.
"""

import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from typing import Any, Mapping

import backend.config as config
import tactical
from backend.analytics.analytics.strategist_failures import tally_failure_reasons
from backend.analytics.analytics_tracker import emit_counter, save_analytics_snapshot
from backend.api.config import (
    ENABLE_FIELD_POPULATION,
    ENABLE_PLANNER,
    ENABLE_PLANNER_PIPELINE,
    EXCLUDE_PARSER_AGGREGATED_ACCOUNTS,
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
from backend.core.case_store.api import get_account_case, list_accounts
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
    BureauAccount,
    BureauPayload,
    ClientInfo,
    Inquiry,
    ProblemAccount,
    ProofDocuments,
)
from backend.core.services.ai_client import AIClient, _StubAIClient, get_ai_client
from backend.core.telemetry.emit import emit
from backend.policy.policy_loader import load_rulebook
from planner import plan_next_step

logger = logging.getLogger(__name__)


def _emit_stageA_events(session_id: str, accounts: list[Mapping[str, Any]]) -> None:
    """Emit telemetry events for Stage A decisions."""
    for acc in accounts or []:
        emit(
            "stageA_problem_decision",
            {
                "session_id": session_id,
                "normalized_name": acc.get("normalized_name"),
                "account_id": acc.get("account_number_last4")
                or acc.get("account_fingerprint"),
                "decision_source": acc.get("decision_source"),
                "primary_issue": acc.get("primary_issue"),
                "confidence": acc.get("confidence", 0.0),
                "tier": acc.get("tier", 0),
                "reasons_count": len(acc.get("problem_reasons", [])),
            },
        )


def collect_stageA_problem_accounts(
    session_id: str, all_accounts: list[Mapping[str, Any]] | None = None
) -> list[Mapping[str, Any]]:
    """Return problem accounts for Stage A.

    When ``ENABLE_CASESTORE_STAGEA`` is true the decisions are read from the
    Case Store ``stageA_detection`` artifacts.  Otherwise ``all_accounts`` is
    filtered using the legacy ``_detector_is_problem`` flag.
    """

    problems: list[Mapping[str, Any]] = []
    if config.ENABLE_CASESTORE_STAGEA:
        for acc_id in list_accounts(session_id):  # type: ignore[operator]
            try:
                case = get_account_case(session_id, acc_id)  # type: ignore[operator]
            except Exception:  # pragma: no cover - defensive
                continue
            art = case.artifacts.get("stageA_detection")
            if not art:
                logger.warning(
                    "stageA_artifact_missing session=%s account=%s", session_id, acc_id
                )
                continue
            data = art.model_dump()
            tier = str(data.get("tier", "none"))
            source = data.get("decision_source", "rules")
            reasons = data.get("problem_reasons", [])
            include = False
            if config.ENABLE_AI_ADJUDICATOR and source == "ai":
                if tier in {"Tier1", "Tier2", "Tier3"}:
                    include = True
            elif reasons:
                include = True
            if include:
                acc = {"account_id": acc_id}
                acc.update(
                    {
                        "primary_issue": data.get("primary_issue", "unknown"),
                        "issue_types": data.get("issue_types", []),
                        "problem_reasons": reasons,
                        "confidence": data.get("confidence", 0.0),
                        "tier": data.get("tier", 0),
                        "decision_source": source,
                    }
                )
                problems.append(acc)
    else:
        for acc in all_accounts or []:
            if acc.get("_detector_is_problem"):
                problems.append(acc)

    _emit_stageA_events(session_id, problems)
    return problems


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
    _emit_stageA_events(session_id, sections.get("problem_accounts", []))
    if (
        os.getenv("DEFER_ASSIGN_ISSUE_TYPES") == "1"
        and not sections.get("negative_accounts")
        and not sections.get("open_accounts_with_issues")
    ):
        all_accounts = sections.get("all_accounts", [])
        sections["negative_accounts"] = list(all_accounts)
        sections["open_accounts_with_issues"] = list(all_accounts)
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


def _annotate_with_tri_merge(sections: Mapping[str, Any]) -> None:
    """Annotate accounts in ``sections`` with tri-merge mismatch details."""
    if not env_bool("ENABLE_TRI_MERGE", False):
        return

    import copy

    from backend.api.session_manager import get_session
    from backend.audit.audit import emit_event
    from backend.core.logic.report_analysis.tri_merge import (
        compute_mismatches,
        normalize_and_match,
    )
    from backend.core.logic.report_analysis.tri_merge_models import Tradeline
    from backend.core.logic.utils.report_sections import filter_sections_by_bureau

    tracked_keys = [
        "negative_accounts",
        "open_accounts_with_issues",
        "high_utilization_accounts",
        "positive_accounts",
        "all_accounts",
    ]
    before = {k: copy.deepcopy(sections.get(k, [])) for k in tracked_keys}
    counts_before = {k: len(v) for k, v in before.items()}
    primary_before: dict[str, Any] = {}
    for lst in before.values():
        for acc in lst:
            acc_id = str(acc.get("account_id") or id(acc))
            primary_before.setdefault(acc_id, acc.get("primary_issue"))

    bureau_data = {
        bureau: filter_sections_by_bureau(sections, bureau, [])
        for bureau in ["Experian", "Equifax", "TransUnion"]
    }

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

    if not tradelines:
        return

    _start = time.perf_counter()
    families = normalize_and_match(tradelines)
    emit_counter("tri_merge.process_time_ms", (time.perf_counter() - _start) * 1000)
    compute_mismatches(families)

    session_id = os.getenv("SESSION_ID", "")
    tri_session = get_session(session_id) if session_id else None
    tri_evidence = (
        (tri_session.get("tri_merge") or {}).get("evidence", {}) if tri_session else {}
    )

    tri_merge_map: dict[str, dict[str, Any]] = {}
    for fam in families:
        family_id = getattr(fam, "family_id", None)
        mismatch_types = [m.field for m in getattr(fam, "mismatches", [])]
        evidence_id = family_id
        evidence = tri_evidence.get(evidence_id)
        for tl in fam.tradelines.values():
            acc_id = str(tl.data.get("account_id") or "")
            if acc_id and family_id:
                info = {
                    "family_id": family_id,
                    "mismatch_types": mismatch_types,
                    "evidence_snapshot_id": evidence_id,
                }
                if evidence:
                    info["evidence"] = evidence
                tri_merge_map[acc_id] = info

    for key in (
        "negative_accounts",
        "open_accounts_with_issues",
        "high_utilization_accounts",
        "positive_accounts",
        "all_accounts",
    ):
        for acc in sections.get(key, []):
            acc_id = str(acc.get("account_id") or "")
            tri_info = tri_merge_map.get(acc_id)
            if not tri_info:
                continue
            acc["tri_merge"] = tri_info

            evidence = (
                tri_info.get("evidence")
                if isinstance(tri_info.get("evidence"), dict)
                else {}
            )
            # Aggregate flags from mismatch types and any explicit evidence flags
            flags: list[str] = []
            if tri_info.get("mismatch_types"):
                flags.append("tri_merge_mismatch")
            flags.extend(
                evidence.get("flags", []) if isinstance(evidence, dict) else []
            )
            if flags:
                existing = acc.setdefault("flags", [])
                for flag in flags:
                    if flag not in existing:
                        existing.append(flag)

            # Populate bureau-level statuses from tri-merge evidence when missing
            if isinstance(evidence, dict) and not acc.get("bureau_statuses"):
                tradelines = evidence.get("tradelines", {})
                statuses: dict[str, str] = {}
                if isinstance(tradelines, dict):
                    for bureau, data in tradelines.items():
                        if not isinstance(data, Mapping):
                            continue
                        status = data.get("status") or data.get("account_status")
                        if status:
                            statuses[bureau] = status
                if statuses:
                    acc["bureau_statuses"] = statuses

    # Ensure tri-merge remains purely annotative.
    violation_reason: str | None = None
    for key in tracked_keys:
        if len(sections.get(key, [])) != counts_before.get(key, 0):
            violation_reason = "account_count_changed"
            break

    if violation_reason is None:
        primary_after: dict[str, Any] = {}
        for key in tracked_keys:
            for acc in sections.get(key, []):
                acc_id = str(acc.get("account_id") or id(acc))
                if acc_id not in primary_after:
                    primary_after[acc_id] = acc.get("primary_issue")
        for acc_id, before_issue in primary_before.items():
            if primary_after.get(acc_id) != before_issue:
                violation_reason = "primary_issue_changed"
                break
        else:
            for acc_id in primary_after:
                if acc_id not in primary_before:
                    violation_reason = "account_count_changed"
                    break

    if violation_reason:
        emit_event("trimerge_violation", {"reason": violation_reason})
        for key, val in before.items():
            sections[key] = val


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
        if os.getenv("DISABLE_TRI_MERGE_PRECONFIRM") != "1":
            _annotate_with_tri_merge(sections)
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
) -> BureauPayload | Mapping[str, Any]:
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

    force_parser = os.getenv("ANALYSIS_FORCE_PARSER_ONLY") == "1"
    if force_parser or sections.get("ai_failed"):
        logger.info("analysis_falling_back_to_parser_only force=%s", force_parser)
        sections = analyze_report_logic(
            pdf_path,
            analyzed_json_path,
            {},
            ai_client=None,
            run_ai=False,
            request_id=session_id,
        )
        sections["needs_human_review"] = True
        sections["ai_failed"] = True
    if (
        os.getenv("DEFER_ASSIGN_ISSUE_TYPES") == "1"
        and not sections.get("negative_accounts")
        and not sections.get("open_accounts_with_issues")
    ):
        all_accounts = sections.get("all_accounts", [])
        sections["negative_accounts"] = list(all_accounts)
        sections["open_accounts_with_issues"] = list(all_accounts)
    sections.setdefault("negative_accounts", [])
    sections.setdefault("open_accounts_with_issues", [])
    sections.setdefault("all_accounts", [])
    sections.setdefault("high_utilization_accounts", [])
    from backend.core.logic.report_analysis.report_postprocessing import (
        _inject_missing_late_accounts,
        enrich_account_metadata,
    )

    def _log_account_snapshot(label: str) -> None:
        all_acc = sections.get("all_accounts") or []
        neg = sections.get("negative_accounts") or []
        open_acc = sections.get("open_accounts_with_issues") or []
        sample_src = all_acc or (neg + open_acc)
        sample = [
            {
                "normalized_name": a.get("normalized_name"),
                "primary_issue": a.get("primary_issue"),
                "issue_types": a.get("issue_types"),
                "status": a.get("status"),
                "source_stage": a.get("source_stage"),
            }
            for a in sample_src[:3]
        ]
        logger.info(
            "%s all_accounts=%d negative_accounts=%d open_accounts_with_issues=%d sample=%s",
            label,
            len(all_acc),
            len(neg),
            len(open_acc),
            sample,
        )

    _log_account_snapshot("post_analyze_report")
    _inject_missing_late_accounts(sections, {}, {}, {})
    _log_account_snapshot("post_inject_missing_late_accounts")

    from backend.core.logic.utils.names_normalization import normalize_creditor_name

    parser_only = {
        normalize_creditor_name(a.get("name", ""))
        for a in sections.get("all_accounts", [])
        if a.get("source_stage") == "parser_aggregated"
    }

    suppress_accounts_without_issue_types = env_bool(
        "SUPPRESS_ACCOUNTS_WITHOUT_ISSUE_TYPES", False
    )

    for cat in ["negative_accounts", "open_accounts_with_issues"]:
        filtered = []
        for acc in sections.get(cat, []):
            if suppress_accounts_without_issue_types and not acc.get("issue_types"):
                logger.info(
                    "suppressed_account %s",
                    {
                        "suppression_reason": "missing_issue_types",
                        "name": acc.get("name"),
                        "category": cat,
                    },
                )
                continue
            norm = normalize_creditor_name(acc.get("name", ""))
            if EXCLUDE_PARSER_AGGREGATED_ACCOUNTS and norm in parser_only:
                logger.info(
                    "suppressed_account %s",
                    {
                        "suppression_reason": "parser_aggregated_only",
                        "name": acc.get("name"),
                        "category": cat,
                    },
                )
                continue
            enriched = enrich_account_metadata(acc)
            remarks_contains_co = acc.get("remarks_contains_co")
            if remarks_contains_co is None:
                remarks = acc.get("remarks")
                remarks_lower = remarks.lower() if isinstance(remarks, str) else ""
                remarks_contains_co = (
                    "charge" in remarks_lower and "off" in remarks_lower
                ) or "collection" in remarks_lower
            logger.info(
                "emitted_account name=%s primary_issue=%s status=%s "
                "last4=%s orig_cred=%s issues=%s bureaus=%s stage=%s "
                "payment_statuses=%s has_co_marker=%s co_bureaus=%s has_remarks=%s "
                "remarks_contains_co=%s",
                enriched.get("normalized_name"),
                enriched.get("primary_issue"),
                enriched.get("status"),
                enriched.get("account_number_last4"),
                enriched.get("original_creditor"),
                enriched.get("issue_types"),
                list((enriched.get("bureau_statuses") or {}).keys()),
                enriched.get("source_stage"),
                acc.get("payment_statuses") or acc.get("payment_status"),
                acc.get("has_co_marker"),
                acc.get("co_bureaus"),
                bool(acc.get("remarks")),
                remarks_contains_co,
            )
            filtered.append(enriched)
        sections[cat] = filtered
    update_session(session_id, status="awaiting_user_explanations")
    _log_account_snapshot("pre_bureau_payload")
    for cat in (
        "negative_accounts",
        "open_accounts_with_issues",
        "high_utilization_accounts",
    ):
        for acc in sections.get(cat, []):
            acc.setdefault("primary_issue", "unknown")
            acc.setdefault("issue_types", [])
            acc.setdefault("status", acc.get("account_status") or "")
            acc.setdefault("late_payments", {})
            acc.setdefault("payment_statuses", {})
            acc.setdefault("has_co_marker", False)
            acc.setdefault("co_bureaus", [])
            acc.setdefault("remarks_contains_co", False)
            acc.setdefault("bureau_statuses", {})
            acc.setdefault("account_number_last4", None)
            acc.setdefault("account_fingerprint", None)
            acc.setdefault("original_creditor", None)
            acc.setdefault("source_stage", acc.get("source_stage") or "")
            acc.setdefault("bureau_details", {})
    if os.getenv("ANALYSIS_TRACE"):
        for acc in sections.get("negative_accounts", []) + sections.get(
            "open_accounts_with_issues", []
        ):
            remarks_contains_co = acc.get("remarks_contains_co")
            if remarks_contains_co is None:
                remarks = acc.get("remarks")
                remarks_lower = remarks.lower() if isinstance(remarks, str) else ""
                remarks_contains_co = (
                    "charge" in remarks_lower and "off" in remarks_lower
                ) or "collection" in remarks_lower
            statuses = acc.get("payment_statuses")
            payment_status_texts: list[str] = []
            if isinstance(statuses, dict):
                payment_status_texts.extend(str(v or "") for v in statuses.values())
            elif isinstance(statuses, (list, tuple, set)):
                payment_status_texts.extend(str(v or "") for v in statuses)
            else:
                payment_status_texts.append(str(statuses or ""))
            single_status = acc.get("payment_status")
            if single_status:
                payment_status_texts.append(str(single_status))
            status_lower = " ".join(payment_status_texts).lower()
            status_contains_co = "collection" in status_lower or (
                "charge" in status_lower and "off" in status_lower
            )
            trace_missing_reasons: list[str] = []
            if not statuses:
                trace_missing_reasons.append("no_payment_status_line")
            grid_history = acc.get("grid_history_raw")
            if grid_history:
                if isinstance(grid_history, dict):
                    grid_values = " ".join(str(v or "") for v in grid_history.values())
                else:
                    grid_values = str(grid_history)
                if "CO" not in grid_values:
                    trace_missing_reasons.append("no_co_in_grid")
            remarks_val = acc.get("remarks")
            if not remarks_val:
                trace_missing_reasons.append("no_remarks")
            if acc.get("heading_join_miss") or acc.get("heading_join_misses"):
                trace_missing_reasons.append("heading_join_miss")
            status_texts_field = acc.get("status_texts")
            if status_texts_field:
                texts: list[str] = []
                if isinstance(status_texts_field, dict):
                    texts.extend(str(v or "") for v in status_texts_field.values())
                elif isinstance(status_texts_field, (list, tuple, set)):
                    texts.extend(str(v or "") for v in status_texts_field)
                else:
                    texts.append(str(status_texts_field))
                combined = " ".join(texts).lower()
                if "collection" not in combined and not (
                    "charge" in combined and "off" in combined
                ):
                    trace_missing_reasons.append("no_collection_in_status_texts")
            details_hint: dict[str, dict[str, Any]] = {}
            details_contains_co = False
            for bureau, fields in (acc.get("bureau_details") or {}).items():
                code = {
                    "TransUnion": "TU",
                    "Experian": "EX",
                    "Equifax": "EQ",
                }.get(bureau, bureau[:2].upper())
                hint: dict[str, Any] = {}
                status_val = fields.get("account_status") or fields.get(
                    "payment_status"
                )
                if status_val:
                    hint["status"] = status_val
                    status_lower = str(status_val).lower()
                    if "collection" in status_lower or (
                        "charge" in status_lower and "off" in status_lower
                    ):
                        details_contains_co = True
                past_due_val = fields.get("past_due_amount")
                if past_due_val not in (None, "", 0):
                    hint["past_due"] = past_due_val
                if hint:
                    details_hint[code] = hint
            trace = {
                "name": acc.get("normalized_name"),
                "source_stage": acc.get("source_stage"),
                "primary_issue": acc.get("primary_issue"),
                "issue_types": acc.get("issue_types"),
                "status": acc.get("status") or acc.get("account_status"),
                "payment_statuses": acc.get("payment_statuses"),
                "payment_status": acc.get("payment_status"),
                "has_co_marker": acc.get("has_co_marker"),
                "remarks_contains_co": remarks_contains_co,
                "late_payments": acc.get("late_payments"),
                "bureau_statuses": acc.get("bureau_statuses"),
                "account_number_last4": acc.get("account_number_last4"),
                "account_fingerprint": acc.get("account_fingerprint"),
                "original_creditor": acc.get("original_creditor"),
            }
            co_bureaus = acc.get("co_bureaus")
            if co_bureaus:
                trace["co_bureaus"] = co_bureaus
            if trace_missing_reasons:
                trace["trace_missing_reasons"] = trace_missing_reasons
            if details_hint:
                trace["details_hint"] = details_hint
            if acc.get("primary_issue") in {"charge_off", "collection"} and not (
                acc.get("has_co_marker")
                or status_contains_co
                or remarks_contains_co
                or details_contains_co
            ):
                logger.info("account_trace_bug %s", json.dumps(trace, sort_keys=True))
            logger.info("account_trace %s", json.dumps(trace, sort_keys=True))
    if config.PROBLEM_DETECTION_ONLY:
        problem_accounts = sections.get("problem_accounts") or []
        return {"problem_accounts": problem_accounts}
    payload = BureauPayload(
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
            ProblemAccount.from_dict(d)
            for d in sections.get("high_utilization_accounts", [])
        ],
    )
    logger.debug(
        "constructed_bureau_payload disputes=%d goodwill=%d inquiries=%d high_utilization=%d",
        len(payload.disputes),
        len(payload.goodwill),
        len(payload.inquiries),
        len(payload.high_utilization),
    )
    payload.needs_human_review = sections.get("needs_human_review", False)
    return payload


def extract_problematic_accounts_from_report_dict(
    file_path: str, session_id: str | None = None
) -> Mapping[str, Any]:
    """Deprecated adapter returning ``dict`` for old clients."""
    logger.debug(
        "extract_problematic_accounts_from_report_dict is deprecated; use extract_problematic_accounts_from_report instead"
    )
    payload = extract_problematic_accounts_from_report(file_path, session_id)
    if isinstance(payload, Mapping):
        return payload
    return {
        "negative_accounts": [a.to_dict() for a in payload.disputes],
        "open_accounts_with_issues": [a.to_dict() for a in payload.goodwill],
        "unauthorized_inquiries": [i.to_dict() for i in payload.inquiries],
    }
