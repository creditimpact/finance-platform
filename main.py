import os
import logging
import config
from pathlib import Path
from datetime import datetime
from shutil import copyfile
from logic.constants import StrategistFailureReason
from email_sender import send_email_with_attachment
from analytics_tracker import save_analytics_snapshot
from analytics.strategist_failures import tally_failure_reasons
from audit import start_audit, clear_audit, AuditLevel
from orchestrators import (
    process_client_intake,
    analyze_credit_report,
    classify_client_responses,
    generate_strategy_plan,
    generate_letters,
    finalize_outputs,
)

logger = logging.getLogger(__name__)
logger.info("Main process starting with OPENAI_BASE_URL=%s", config.OPENAI_BASE_URL)
logger.info("Main process OPENAI_API_KEY present=%s", bool(config.OPENAI_API_KEY))


def validate_env_variables():
    defaults = {
        "SMTP_SERVER": "localhost",
        "SMTP_PORT": "1025",
        "SMTP_USERNAME": "noreply@example.com",
        "SMTP_PASSWORD": "",
    }

    print("🔍 Validating environment configuration...\n")
    base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    os.environ["OPENAI_BASE_URL"] = base_url
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is missing")
    print(f"✅ OPENAI_BASE_URL: {base_url}")
    print("✅ OPENAI_API_KEY is set.")
    for var, default in defaults.items():
        if not os.getenv(var):
            print(f"⚠️ {var} not set, using default '{default}'")
            os.environ[var] = default
        else:
            print(f"✅ {var} is set.")
    print("✅ Environment variables configured.\n")


def merge_strategy_data(strategy_obj: dict, bureau_data_obj: dict, classification_map: dict, audit=None, log_list=None):
    from logic.constants import normalize_action_tag, FallbackReason, StrategistFailureReason
    from logic.utils import normalize_creditor_name
    from logic.fallback_manager import determine_fallback_action
    import re

    def norm_key(name: str, number: str) -> tuple[str, str]:
        norm_name = normalize_creditor_name(name)
        digits = re.sub(r"\D", "", number or "")
        last4 = digits[-4:] if digits else ""
        return norm_name, last4

    index = {}
    for item in strategy_obj.get("accounts", []):
        key = norm_key(item.get("name", ""), item.get("account_number", ""))
        index[key] = item

    for bureau, payload in bureau_data_obj.items():
        for section, items in payload.items():
            if not isinstance(items, list):
                continue
            for acc in items:
                key = norm_key(acc.get("name", ""), acc.get("account_number", ""))
                src = index.get(key)
                raw_action = None
                tag = ""
                failure_reason = None
                acc_id = acc.get("account_id") or acc.get("name")
                if src is None:
                    failure_reason = StrategistFailureReason.MISSING_INPUT
                    if audit:
                        audit.log_account(
                            acc_id,
                            {
                                "stage": "strategist_failure",
                                "failure_reason": failure_reason.value,
                            },
                        )
                    if log_list is not None:
                        log_list.append(
                            f"[{bureau}] No strategist entry for '{acc.get('name')}' ({acc.get('account_number')})"
                        )
                else:
                    raw_action = src.get("recommended_action") or src.get("recommendation")
                    tag, action = normalize_action_tag(raw_action)
                    if raw_action is None:
                        failure_reason = StrategistFailureReason.EMPTY_OUTPUT
                        if audit:
                            audit.log_account(
                                acc_id,
                                {
                                    "stage": "strategist_failure",
                                    "failure_reason": failure_reason.value,
                                },
                            )
                    elif raw_action and not tag:
                        failure_reason = StrategistFailureReason.UNRECOGNIZED_FORMAT
                        if audit:
                            audit.log_account(
                                acc_id,
                                {
                                    "stage": "strategist_failure",
                                    "failure_reason": failure_reason.value,
                                    "raw_action": raw_action,
                                },
                            )
                        print(f"[⚠️] Unrecognised strategist action '{raw_action}' for {src.get('name')}")
                        acc["fallback_unrecognized_action"] = True
                    if tag:
                        acc["action_tag"] = tag
                        acc["recommended_action"] = action
                    elif raw_action:
                        acc["recommended_action"] = raw_action

                    if "advisor_comment" in src:
                        acc["advisor_comment"] = src["advisor_comment"]
                    elif "analysis" in src:
                        acc["advisor_comment"] = src["analysis"]
                    if src.get("flags"):
                        acc["flags"] = src["flags"]

                if not acc.get("action_tag"):
                    strategist_action = raw_action if raw_action else None
                    if raw_action is None:
                        fallback_reason = FallbackReason.NO_RECOMMENDATION
                    else:
                        raw_key = str(raw_action).strip().lower().replace(" ", "_")
                        fallback_reason = (
                            FallbackReason.KEYWORD_MATCH
                            if raw_key == FallbackReason.KEYWORD_MATCH.value
                            else FallbackReason.UNRECOGNIZED_TAG
                        )

                    fallback_action = determine_fallback_action(acc)
                    keywords_trigger = fallback_action == "dispute"

                    if keywords_trigger:
                        acc["action_tag"] = "dispute"
                        if raw_action:
                            acc["recommended_action"] = "Dispute"
                        else:
                            acc.setdefault("recommended_action", "Dispute")

                        if log_list is not None and (raw_action is None or not tag):
                            if raw_action:
                                log_list.append(
                                    f"[{bureau}] Fallback dispute overriding '{raw_action}' for '{acc.get('name')}' ({acc.get('account_number')})",
                                )
                            else:
                                log_list.append(
                                    f"[{bureau}] Fallback dispute (no recommendation) for '{acc.get('name')}' ({acc.get('account_number')})",
                                )
                    else:
                        if log_list is not None and (raw_action is None or not tag):
                            log_list.append(
                                f"[{bureau}] Evaluated fallback for '{acc.get('name')}' ({acc.get('account_number')})",
                            )

                    overrode_strategist = bool(raw_action) and bool(keywords_trigger)

                    if audit:
                        audit.log_account(
                            acc_id,
                            {
                                "stage": "strategy_fallback",
                                "fallback_reason": fallback_reason.value,
                                "strategist_action": strategist_action,
                                **(
                                    {"raw_action": strategist_action}
                                    if acc.get("fallback_unrecognized_action") and strategist_action
                                    else {}
                                ),
                                "overrode_strategist": overrode_strategist,
                                **(
                                    {"failure_reason": failure_reason.value}
                                    if failure_reason
                                    else {}
                                ),
                            },
                        )

                if audit:
                    cls = classification_map.get(str(acc.get("account_id")))
                    audit.log_account(
                        acc_id,
                        {
                            "stage": "strategy_decision",
                            "action": acc.get("action_tag") or None,
                            "recommended_action": acc.get("recommended_action"),
                            "flags": acc.get("flags"),
                            "reason": acc.get("advisor_comment")
                            or acc.get("analysis")
                            or raw_action,
                            "classification": cls,
                        },
                    )

def run_credit_repair_process(client_info, proofs_files, is_identity_theft):
    """High-level controller for the credit repair pipeline."""
    log_messages = []
    today_folder = None
    pdf_path = None
    audit = start_audit()
    session_id = None

    try:
        print("\n✅ Starting Credit Repair Process (B2C Mode)...")
        log_messages.append("✅ Process started.")
        audit.log_step("process_started", {"is_identity_theft": is_identity_theft})

        session_id, structured_map, raw_map = process_client_intake(client_info, audit)
        classification_map = classify_client_responses(structured_map, raw_map, client_info, audit)
        pdf_path, sections, bureau_data, today_folder = analyze_credit_report(
            proofs_files, session_id, client_info, audit, log_messages
        )
        strategy = generate_strategy_plan(
            client_info, bureau_data, classification_map, session_id, audit, log_messages
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
        )
        finalize_outputs(client_info, today_folder, sections, audit, log_messages)

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        log_messages.append(error_msg)
        audit.log_error(error_msg)
        raise

    finally:
        save_log_file(client_info, is_identity_theft, today_folder, log_messages)
        if today_folder:
            audit.save(today_folder)
            if config.EXPORT_TRACE_FILE:
                from trace_exporter import export_trace_file
                export_trace_file(audit, session_id)
        clear_audit()
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
                print(f"[🧹] Deleted uploaded PDF: {pdf_path}")
            except Exception as delete_error:
                print(f"[⚠️] Failed to delete uploaded PDF: {delete_error}")
def extract_problematic_accounts_from_report(file_path: str, session_id: str | None = None) -> dict:
    """Return problematic accounts extracted from the report for user review."""
    validate_env_variables()

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

def get_current_month():
    return datetime.now().strftime("%Y-%m")

def extract_all_accounts(sections: dict) -> list:
    """Return a de-duplicated list of all accounts with source categories.

    Accounts are considered the same only when key fields match. This prevents
    different accounts from the same creditor from being merged together.
    """

    from datetime import datetime
    import re
    from logic.generate_goodwill_letters import normalize_creditor_name

    def sanitize_number(num: str | None) -> str:
        if not num:
            return ""
        digits = "".join(c for c in num if c.isdigit())
        return digits[-4:] if len(digits) >= 4 else digits

    def parse_date(date_str: str | None) -> datetime | None:
        if not date_str:
            return None
        for fmt in ("%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str, fmt)
            except Exception:
                continue
        return None

    accounts: list[dict] = []
    categories = [
        "negative_accounts",
        "open_accounts_with_issues",
        "high_utilization_accounts",
        "positive_accounts",
        "all_accounts",
    ]

    for key in categories:
        for acc in sections.get(key, []):
            acc_copy = acc.copy()
            acc_copy.setdefault("categories", set()).add(key)

            norm_name = normalize_creditor_name(acc_copy.get("name", "")).lower()
            last4 = sanitize_number(acc_copy.get("account_number"))
            bureaus = tuple(sorted(acc_copy.get("bureaus", [])))
            status = (acc_copy.get("status") or "").strip().lower()
            opened = parse_date(acc_copy.get("opened_date"))
            closed = parse_date(acc_copy.get("closed_date"))

            found = None
            for existing in accounts:
                if (
                    normalize_creditor_name(existing.get("name", "")).lower() == norm_name
                    and sanitize_number(existing.get("account_number")) == last4
                    and tuple(sorted(existing.get("bureaus", []))) == bureaus
                    and (existing.get("status") or "").strip().lower() == status
                    and parse_date(existing.get("opened_date")) == opened
                    and parse_date(existing.get("closed_date")) == closed
                ):
                    found = existing
                    break

            if found:
                found.setdefault("categories", set()).add(key)
            else:
                accounts.append(acc_copy)

    # Flag potential duplicate negatives across all bureaus
    from difflib import SequenceMatcher

    def _is_negative(acc: dict) -> bool:
        cats = {c.lower() for c in acc.get("categories", [])}
        status = str(acc.get("status") or acc.get("reported_status") or "").lower()
        if "negative_accounts" in cats:
            return True
        return any(
            kw in status
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

    def _acct_suffix(num: str | None) -> str:
        if not num:
            return ""
        digits = re.sub(r"\D", "", str(num))
        return digits[-4:]

    def _similar_name(a: str, b: str) -> bool:
        n1 = normalize_creditor_name(a or "").lower()
        n2 = normalize_creditor_name(b or "").lower()
        if n1 == n2 or n1.startswith(n2) or n2.startswith(n1):
            return True
        return SequenceMatcher(None, n1, n2).ratio() >= 0.8

    def _parse_amount(val: str | None) -> float | None:
        if not val:
            return None
        clean = re.sub(r"[^0-9.]+", "", str(val))
        try:
            return float(clean)
        except Exception:
            return None

    def _potential_dupe(a: dict, b: dict) -> bool:
        if not _similar_name(a.get("name"), b.get("name")):
            return False

        s1 = _acct_suffix(a.get("account_number"))
        s2 = _acct_suffix(b.get("account_number"))
        if s1 and s2 and s1 != s2:
            return False

        bal1 = _parse_amount(a.get("balance"))
        bal2 = _parse_amount(b.get("balance"))
        if bal1 is not None and bal2 is not None:
            diff = abs(bal1 - bal2)
            if diff > max(100, 0.1 * min(bal1, bal2)):
                return False

        d1 = parse_date(a.get("opened_date"))
        d2 = parse_date(b.get("opened_date"))
        if d1 and d2 and abs((d1 - d2).days) > 90:
            return False
        d1c = parse_date(a.get("closed_date"))
        d2c = parse_date(b.get("closed_date"))
        if d1c and d2c and abs((d1c - d2c).days) > 90:
            return False

        return True

    dupe_indices: set[int] = set()
    for i, acc_a in enumerate(accounts):
        if not _is_negative(acc_a):
            continue
        for j in range(i + 1, len(accounts)):
            acc_b = accounts[j]
            if not _is_negative(acc_b):
                continue
            if _potential_dupe(acc_a, acc_b):
                dupe_indices.update({i, j})

    for idx in dupe_indices:
        accounts[idx]["duplicate_suspect"] = True

    return accounts


def save_log_file(client_info, is_identity_theft, output_folder, log_lines):
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
        ""
    ]

    with open(log_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(header + log_lines))
    print(f"[📝] Log saved: {log_path}")
