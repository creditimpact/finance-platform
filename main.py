import os
import logging
import config
from pathlib import Path
from datetime import datetime
from shutil import copyfile
from logic.extract_info import extract_bureau_info_column_refined
from logic.analyze_report import analyze_credit_report
from logic.utils import convert_txts_to_pdfs, gather_supporting_docs_text
from logic.letter_generator import generate_dispute_letters_for_all_bureaus
from logic.instructions_generator import generate_instruction_file
from logic.generate_goodwill_letters import generate_goodwill_letters
from logic.generate_custom_letters import generate_custom_letters
from logic.generate_strategy_report import StrategyGenerator
from logic.summary_classifier import classify_client_summary
from logic.constants import StrategistFailureReason
from email_sender import send_email_with_attachment
from analytics_tracker import save_analytics_snapshot
from analytics.strategist_failures import tally_failure_reasons
from audit import start_audit, get_audit, clear_audit, AuditLevel

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
    validate_env_variables()

    log_messages = []
    today_folder = None
    pdf_path = None
    audit = start_audit()
    session_id = client_info.get("session_id", "session")

    try:
        print("\n✅ Starting Credit Repair Process (B2C Mode)...")
        log_messages.append("✅ Process started.")
        audit.log_step("process_started", {"is_identity_theft": is_identity_theft})

        from logic.upload_validator import is_safe_pdf, move_uploaded_file
        from session_manager import update_session, get_intake

        if "email" not in client_info or not client_info["email"]:
            raise ValueError("Client email is missing.")

        audit.log_step("session_initialized", {"session_id": session_id})

        # Record raw and structured client explanations for traceability
        intake = get_intake(session_id) or {}
        structured = client_info.get("structured_summaries") or {}
        structured_map: dict[str, dict] = {}
        classification_map: dict[str, dict] = {}
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
        for acc_id, struct in structured_map.items():
            cls = classify_client_summary(struct, client_info.get("state"))
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
        uploaded_path = proofs_files.get("smartcredit_report")
        if not uploaded_path or not os.path.exists(uploaded_path):
            raise FileNotFoundError("SmartCredit report file not found at path: " + str(uploaded_path))

        pdf_path = move_uploaded_file(Path(uploaded_path), session_id)
        update_session(session_id, file_path=str(pdf_path))
        if not is_safe_pdf(pdf_path):
            raise ValueError("Uploaded file failed PDF safety checks.")

        print("📄 Extracting client info from report...")
        client_personal_info = extract_bureau_info_column_refined(pdf_path)
        client_info.update(client_personal_info.get("data", {}))
        log_messages.append("📄 Personal info extracted.")
        # Avoid logging personal details unless verbose auditing is enabled
        if audit.level == AuditLevel.VERBOSE:
            audit.log_step("personal_info_extracted", client_personal_info)

        print("🔍 Analyzing report with GPT...")
        analyzed_json_path = Path("output/analyzed_report.json")
        sections = analyze_credit_report(pdf_path, analyzed_json_path, client_info)
        client_info.update(sections)
        log_messages.append("🔍 Report analyzed.")
        audit.log_step(
            "report_analyzed",
            {
                "negative_accounts": sections.get("negative_accounts", []),
                "open_accounts_with_issues": sections.get(
                    "open_accounts_with_issues", []
                ),
                "unauthorized_inquiries": sections.get(
                    "unauthorized_inquiries", []
                ),
            },
        )

        safe_name = (client_info.get("name") or "Client").replace(" ", "_").replace("/", "_")
        today_folder = Path(f"Clients/{get_current_month()}/{safe_name}_{session_id}")
        today_folder.mkdir(parents=True, exist_ok=True)
        log_messages.append(f"📁 Client folder created at: {today_folder}")
        if audit.level == AuditLevel.VERBOSE:
            audit.log_step("client_folder_created", {"path": str(today_folder)})

                # 🧹 מחיקת PDFים ו־JSON ישנים (רק תגובות GPT)
        for file in today_folder.glob("*.pdf"):
            file.unlink()
        for file in today_folder.glob("*_gpt_response.json"):
            file.unlink()


        # 🧾 Copy original report into client folder
        original_pdf_copy = today_folder / "Original SmartCredit Report.pdf"
        copyfile(pdf_path, original_pdf_copy)
        log_messages.append("📁 Original report saved to client folder.")

        # 🧠 Copy analyzed JSON into folder
        if analyzed_json_path.exists():
            copyfile(analyzed_json_path, today_folder / "analyzed_report.json")
            log_messages.append("📁 Analyzed report JSON saved.")

        from logic.utils import filter_sections_by_bureau
        detailed_logs = []
        bureau_data = {
            bureau: filter_sections_by_bureau(sections, bureau, detailed_logs)
            for bureau in ["Experian", "Equifax", "TransUnion"]
        }
        log_messages.extend(detailed_logs)
        if audit.level == AuditLevel.VERBOSE:
            audit.log_step("sections_split_by_bureau", bureau_data)

        print("🧠 Generating strategy report...")
        docs_text = gather_supporting_docs_text(session_id)
        strat_gen = StrategyGenerator()
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

        merge_strategy_data(strategy, bureau_data, classification_map, audit, log_list=log_messages)
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


        print("📄 Generating dispute letters...")
        generate_dispute_letters_for_all_bureaus(
            client_info,
            bureau_data,
            today_folder,
            is_identity_theft,
            log_messages=log_messages,
        )
        log_messages.append("📄 Dispute letters generated.")
        if audit.level is AuditLevel.VERBOSE:
            audit.log_step("dispute_letters_generated")

        if not is_identity_theft:
            print("💌 Generating goodwill letters...")
            generate_goodwill_letters(client_info, bureau_data, today_folder)
            log_messages.append("💌 Goodwill letters generated.")
            if audit.level is AuditLevel.VERBOSE:
                audit.log_step("goodwill_letters_generated")
        else:
            print("🔒 Identity theft case - skipping goodwill letters.")
            log_messages.append("🚫 Goodwill letters skipped due to identity theft.")
            if audit.level is AuditLevel.VERBOSE:
                audit.log_step("goodwill_letters_skipped")

        # 🧠 הזרקת all_accounts להוראות
        all_accounts = extract_all_accounts(sections)
        for bureau in bureau_data:
            bureau_data[bureau]["all_accounts"] = all_accounts

        print("📝 Generating custom letters...")
        generate_custom_letters(
            client_info,
            bureau_data,
            today_folder,
            log_messages=log_messages,
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

        print("📧 Sending email with all documents to client...")
        output_files = [str(p) for p in today_folder.glob("*.pdf")]
        raw_name = (client_info.get("name") or "").strip()
        first_name = raw_name.split()[0] if raw_name else "Client"
        send_email_with_attachment(
            receiver_email=client_info["email"],
            subject="Your Credit Repair Package is Ready",
            body=f"""
Hi {first_name},

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
            files=output_files
        )
        log_messages.append("📧 Email sent to client.")
        if audit.level is AuditLevel.VERBOSE:
            audit.log_step("email_sent", {"files": output_files})

        from logic.utils import extract_summary_from_sections
        failure_counts = tally_failure_reasons(audit)
        save_analytics_snapshot(
            client_info,
            extract_summary_from_sections(sections),
            strategist_failures=failure_counts,
        )
        log_messages.append("📊 Analytics snapshot saved.")
        if audit.level is AuditLevel.VERBOSE:
            audit.log_step("analytics_saved", {"strategist_failures": failure_counts})

        print(f"\n🎯 Credit Repair Process completed successfully!")
        print(f"📂 All output saved to: {today_folder}")
        log_messages.append("🎯 Process completed successfully.")
        audit.log_step("process_completed")

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
    sections = analyze_credit_report(pdf_path, analyzed_json_path, {})

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
