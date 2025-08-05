import os
import json
import fitz  # pymupdf
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import re
from .generate_goodwill_letters import normalize_creditor_name
from logic.utils import (
    extract_late_history_blocks,
    extract_pdf_text_safe,
    extract_inquiries,
)
from .utils import normalize_bureau_name, has_late_indicator, enforce_collection_status

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(pdf_path):
    """Extract text using a robust multi-engine approach."""
    return extract_pdf_text_safe(Path(pdf_path), max_chars=150000)

def call_ai_analysis(text, client_goal, is_identity_theft, output_json_path):
    late_blocks, late_raw_map = extract_late_history_blocks(text, return_raw_map=True)
    if late_blocks:
        late_summary_text = "Late payment history extracted from report:\n"
        for account, bureaus in late_blocks.items():
            raw = late_raw_map.get(account, account)
            for bureau, counts in bureaus.items():
                parts = [f"{k} days late: {v}" for k, v in counts.items()]
                goodwill = False
                thirty = counts.get("30", 0)
                sixty = counts.get("60", 0)
                ninety = counts.get("90", 0)
                if (thirty == 1 and not sixty and not ninety) or (
                    thirty == 2 and not sixty and not ninety
                ):
                    goodwill = True
                line = f"- {account} (raw: {raw}) ({bureau}): {', '.join(parts)}"
                if goodwill:
                    line += " \u2192 Possible goodwill candidate"
                late_summary_text += line + "\n"
    else:
        late_summary_text = (
            "No late payment blocks were detected by parsing logic. "
            "Do not infer or guess late payment history unless clearly shown."
        )

    inquiry_list = extract_inquiries(text)
    if inquiry_list:
        inquiry_summary = "\nParsed inquiries from report:\n" + "\n".join(
            f"- {normalize_creditor_name(i['creditor_name'])} (raw: {i['creditor_name']}) {i['date']} ({i['bureau']})"
            for i in inquiry_list
        )
    else:
        inquiry_summary = "\nNo inquiries were detected by the parser."

    prompt = f"""
You are a senior credit repair expert with deep knowledge of credit reports, FCRA regulations, dispute strategies, and client psychology.

Your task: deeply analyze the following SmartCredit report text and extract a structured JSON summary for use in automated dispute and goodwill letters, and personalized client instructions.

Client goal: "{client_goal}"
Identity theft case: {"Yes" if is_identity_theft else "No"}

Return this exact JSON structure:

1. personal_info_issues: List of inconsistencies or mismatches in personal info (name, address, DOB) across bureaus.

2. negative_accounts: Accounts marked as Chargeoff or in Collections.
   For each account, include:
   - name
   - balance (e.g., "$2,356")
   - status ("Chargeoff" or "Collections")
   - account_status ("Open" or "Closed")
   - account_number (if visible)
   - bureaus (["Experian", "Equifax", "TransUnion"])
   - is_suspected_identity_theft: true/false
   - dispute_type: "identity_theft" / "unauthorized_or_unverified" / "inaccurate_reporting"
   - impact: short sentence describing how this negatively affects the client
   - advisor_comment: 1–2 sentences explaining what we’re doing about this account and why.

3. open_accounts_with_issues: Non-collection accounts with any late payment history or remarks like "past due", "derogatory", etc. Look carefully for text such as "30 days late", "60 days late", or "past due" even if formatting is odd. If an account is otherwise in good standing with only one or two late payments, mark ``goodwill_candidate`` true.
   Include:
   - name
   - status
   - account_status
   - account_number (if available)
   - bureaus
   - goodwill_candidate: true/false
   - is_suspected_identity_theft: true/false
   - dispute_type
   - hardship_reason: (e.g. "temporary job loss")
   - recovery_summary
   - advisor_comment

4. high_utilization_accounts: Revolving accounts with balance over 30% of limit.
   Include:
   - name
   - balance
   - limit
   - utilization (e.g., "76%")
   - bureaus
   - is_suspected_identity_theft: true/false
   - advisor_comment
   - recommended_payment_amount

5. positive_accounts: Accounts in good standing that support the credit profile.
   Include:
   - name
   - account_number (if available)
   - opened_date (if available)
   - balance
   - status
   - utilization (if revolving)
   - bureaus
   - advisor_comment

6. inquiries: Hard inquiries from the last 2 years that are NOT tied to any existing account.
   Exclude inquiries that clearly relate to open or closed accounts listed. Treat names with minor spelling differences as the same creditor when deciding to exclude.
   Include:
   - creditor_name
   - date (MM/YYYY)
   - bureau

7. account_inquiry_matches: Inquiries clearly matching existing accounts (to be excluded from disputes).
   Include:
   - creditor_name
   - matched_account_name

8. summary_metrics:
   - num_collections
   - num_late_payments
   - high_utilization (true/false)
   - recent_inquiries
   - total_inquiries
   - num_negative_accounts
   - num_accounts_over_90_util
   - account_types_in_problem

9. strategic_recommendations: Clear action items based on the client goal, such as:
   - "Open secured card"
   - "Lower utilization on Capital One"
   - "Dispute Midland account with Experian"

10. all_accounts: ✅ For every account (positive or negative), return:
   - name
   - bureaus
   - status
   - utilization (if available)
   - advisor_comment: 1–2 sentence explanation of the account’s effect and what the client should do (dispute, pay down, keep healthy, goodwill, etc.)

⚠️ Rules:
- Output valid JSON only
- No markdown, no explanations, no headers — only pure JSON
- Use proper casing, punctuation, and clean formatting
- Never guess — only include facts that are visible
Use the following late payment data to help you accurately tag late accounts, even if the report formatting is inconsistent:

{late_summary_text}

{inquiry_summary}

Report text:
===
{text}
===
"""

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    content = response.choices[0].message.content.strip()

    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    raw_path = output_json_path.with_name(output_json_path.stem + "_raw.txt")
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(content)
        f.write("\n\n---\n[Debug] Late Payment Summary Used in Prompt:\n")
        f.write(late_summary_text)
        f.write("\n\n---\n[Debug] Inquiry Summary Used in Prompt:\n")
        f.write(inquiry_summary)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("\u26a0\ufe0f The AI returned invalid JSON. Here's the raw response:")
        print(content)
        raise

def validate_analysis_sanity(analysis: dict):
    warnings = []

    if not analysis.get("negative_accounts") and not analysis.get("open_accounts_with_issues"):
        warnings.append("\u26a0\ufe0f No dispute/goodwill accounts found.")

    total_inquiries = analysis.get("summary_metrics", {}).get("total_inquiries")
    if isinstance(total_inquiries, list):
        if len(total_inquiries) > 50:
            warnings.append("\u26a0\ufe0f Too many inquiries detected — may indicate parsing issue.")
    elif isinstance(total_inquiries, int):
        if total_inquiries > 50:
            warnings.append("\u26a0\ufe0f Too many inquiries detected — may indicate parsing issue.")

    if not analysis.get("strategic_recommendations"):
        warnings.append("\u26a0\ufe0f No strategic recommendations provided.")

    for section in ["negative_accounts", "open_accounts_with_issues", "all_accounts"]:
        for account in analysis.get(section, []):
            comment = account.get("advisor_comment", "")
            if len(comment.split()) < 4:
                warnings.append(f"\u26a0\ufe0f Advisor comment too short for account: {account.get('name')}")

    if warnings:
        print("\n[!] ANALYSIS QA WARNINGS:")
        for warn in warnings:
            print(warn)

    return warnings


def _merge_parser_inquiries(result: dict, parsed: list[dict]):
    """Merge parser-detected inquiries, preferring them over GPT output.

    Any inquiries present in ``parsed`` but missing from the AI result are
    injected with an ``advisor_comment`` note so downstream code can track the
    source.
    """
    cleaned: list[dict] = []
    seen = set()

    gpt_set = {
        (
            normalize_creditor_name(i.get("creditor_name")),
            i.get("date"),
            normalize_bureau_name(i.get("bureau")),
        )
        for i in result.get("inquiries", [])
    }

    for inq in parsed:
        key = (
            normalize_creditor_name(inq.get("creditor_name")),
            inq.get("date"),
            normalize_bureau_name(inq.get("bureau")),
        )
        if key in seen:
            continue
        entry = {
            "creditor_name": inq.get("creditor_name"),
            "date": inq.get("date"),
            "bureau": normalize_bureau_name(inq.get("bureau")),
        }
        if key not in gpt_set:
            entry["advisor_comment"] = "Detected by parser; missing from AI output"
        cleaned.append(entry)
        seen.add(key)

    for inq in result.get("inquiries", []):
        key = (
            normalize_creditor_name(inq.get("creditor_name")),
            inq.get("date"),
            normalize_bureau_name(inq.get("bureau")),
        )
        if key not in seen:
            cleaned.append(inq)
            seen.add(key)

    if cleaned:
        result["inquiries"] = cleaned
    elif "inquiries" in result:
        # Ensure field exists even if empty for downstream code
        result["inquiries"] = []


def _sanitize_late_counts(history: dict) -> None:
    """Remove unrealistic late payment numbers from parsed history."""
    for acc, bureaus in list(history.items()):
        for bureau, counts in list(bureaus.items()):
            for key, val in list(counts.items()):
                if val > 12:
                    print(
                        f"[~] Dropping unrealistic count {val}x{key} for {acc} ({bureau})"
                    )
                    counts.pop(key)
            if not counts:
                bureaus.pop(bureau)
        if not bureaus:
            history.pop(acc)


def _cleanup_unverified_late_text(result: dict, verified: set[str]):
    """Remove GPT late references for accounts without verified history."""

    def clean(acc: dict):
        norm = normalize_creditor_name(acc.get("name", ""))
        if norm in verified:
            return
        if "flags" in acc:
            acc["flags"] = [f for f in acc["flags"] if "late" not in f.lower()]
            if not acc["flags"]:
                acc.pop("flags")
        comment = acc.get("advisor_comment")
        if comment and re.search(r"late|delinqu", comment, re.I):
            acc.pop("advisor_comment", None)

    for sec in [
        "all_accounts",
        "negative_accounts",
        "open_accounts_with_issues",
        "positive_accounts",
        "high_utilization_accounts",
    ]:
        for a in result.get(sec, []):
            clean(a)

def _inject_missing_late_accounts(result: dict, history: dict, raw_map: dict) -> None:
    """Add accounts detected by the parser but missing from the AI output."""
    existing = {
        normalize_creditor_name(acc.get("name", "")) for acc in result.get("all_accounts", [])
    }

    for norm_name, bureaus in history.items():
        if norm_name in existing:
            continue
        entry = {
            "name": raw_map.get(norm_name, norm_name),
            "bureaus": list(bureaus.keys()),
            "status": "Unknown",
            "advisor_comment": "Detected by parser; missing from AI output",
            "late_payments": bureaus,
            "flags": ["Late Payments"],
        }
        result.setdefault("all_accounts", []).append(entry)
        print(f"[⚠️] Added missing account from parser: {entry['name']}")

def analyze_credit_report(pdf_path, output_json_path, client_info):
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        raise ValueError("\u274c No text extracted from PDF")

    def detected_late_phrases(txt: str) -> bool:
        import re
        return bool(re.search(r"late|past due", txt, re.I))

    raw_goal = client_info.get("goal", "").strip().lower()
    if raw_goal in ["", "not specified", "improve credit", "repair credit"]:
        client_goal = (
            "Improve credit score significantly within the next 3–6 months using strategies such as authorized users, "
            "credit building tools, and removal of negative items."
        )
    else:
        client_goal = client_info.get("goal", "Not specified")

    is_identity_theft = client_info.get("is_identity_theft", False)
    result = call_ai_analysis(text, client_goal, is_identity_theft, output_json_path)

    parsed_inquiries = extract_inquiries(text)
    if parsed_inquiries:
        print(f"[🔎] Parser found {len(parsed_inquiries)} inquiries in text.")
    else:
        print("[⚠️] Parser did not find any inquiries in the report text.")

    if result.get("inquiries"):
        print(f"[🔍] GPT found {len(result['inquiries'])} inquiries:")
        for inq in result["inquiries"]:
            print(f"  • {inq['creditor_name']} - {inq['date']} ({inq['bureau']})")
    else:
        print("[⚠️] No inquiries found in GPT result.")

    try:
        account_names = {acc.get("name", "") for acc in result.get("all_accounts", [])}
        history_all, raw_map = extract_late_history_blocks(text, return_raw_map=True)
        _sanitize_late_counts(history_all)
        history = extract_late_history_blocks(text, account_names)
        _sanitize_late_counts(history)

        if history:
            print(f"[✅] Found {len(history)} late payment block(s):")
            for creditor, bureaus in history.items():
                print(
                    f"[🔍] Detected late payments for: '{creditor.title()}' → {bureaus}"
                )
        else:
            print("[❌] No late payment history blocks detected.")

        existing_norms = set()
        for acc in result.get("all_accounts", []):
            raw_name = acc.get("name", "")
            norm = normalize_creditor_name(raw_name)
            if raw_name and norm != raw_name.lower().strip():
                print(f"[~] Normalized account heading '{raw_name}' -> '{norm}'")
            existing_norms.add(norm)
            if norm in history:
                acc["late_payments"] = history[norm]
                if any(v >= 1 for vals in history[norm].values() for v in vals.values()):
                    acc.setdefault("flags", []).append("Late Payments")
                    status_text = str(acc.get("status") or acc.get("account_status") or "").strip().lower()
                    if status_text == "closed":
                        acc["goodwill_on_closed"] = True

        for section in ["negative_accounts", "open_accounts_with_issues", "positive_accounts", "high_utilization_accounts"]:
            for acc in result.get(section, []):
                raw_name = acc.get("name", "")
                norm = normalize_creditor_name(raw_name)
                if raw_name and norm != raw_name.lower().strip():
                    print(f"[~] Normalized account heading '{raw_name}' -> '{norm}'")
                if norm in history:
                    acc["late_payments"] = history[norm]
                    if any(v >= 1 for vals in history[norm].values() for v in vals.values()):
                        acc.setdefault("flags", []).append("Late Payments")
                        if section not in ["negative_accounts", "open_accounts_with_issues"]:
                            acc["goodwill_candidate"] = True
                        status_text = str(acc.get("status") or acc.get("account_status") or "").strip().lower()
                        if status_text == "closed":
                            acc["goodwill_on_closed"] = True

        for raw_norm, bureaus in history_all.items():
            pretty = {b: v for b, v in bureaus.items()}
            linked = raw_norm in history
            if linked:
                print(f"[📝] Linked late payment block '{raw_map.get(raw_norm, raw_norm)}' to account '{raw_norm.title()}'")
            else:
                snippet = raw_map.get(raw_norm, raw_norm)
                print(f"[⚠️] Unlinked late-payment block detected near: '{snippet}'")

        # Remove any late_payment fields that were not verified by parser
        verified_names = set(history.keys())
        def strip_unverified(acc_list):
            for acc in acc_list:
                norm = normalize_creditor_name(acc.get("name", ""))
                if "late_payments" in acc and norm not in verified_names:
                    acc.pop("late_payments", None)
        for sec in ["all_accounts", "negative_accounts", "open_accounts_with_issues", "positive_accounts", "high_utilization_accounts"]:
            strip_unverified(result.get(sec, []))

        _cleanup_unverified_late_text(result, verified_names)

        _inject_missing_late_accounts(result, history_all, raw_map)

        _merge_parser_inquiries(result, parsed_inquiries)

        for section in [
            "all_accounts",
            "negative_accounts",
            "open_accounts_with_issues",
            "positive_accounts",
            "high_utilization_accounts",
        ]:
            for acc in result.get(section, []):
                enforce_collection_status(acc)

        # Check that GPT returned all parser-detected inquiries
        found_pairs = {
            (normalize_creditor_name(i.get("creditor_name")), i.get("date"), normalize_bureau_name(i.get("bureau")))
            for i in result.get("inquiries", [])
        }
        for parsed in parsed_inquiries:
            key = (
                normalize_creditor_name(parsed["creditor_name"]),
                parsed["date"],
                normalize_bureau_name(parsed["bureau"]),
            )
            if key not in found_pairs:
                print(f"[⚠️] Inquiry missing from GPT output: {parsed['creditor_name']} {parsed['date']} ({parsed['bureau']})")


    except Exception as e:
        print(f"[⚠️] Late history parsing failed: {e}")

    warnings = validate_analysis_sanity(result)
    if not result.get("open_accounts_with_issues") and detected_late_phrases(text):
        msg = "⚠️ Late payment terms found in text but no accounts marked with issues."
        warnings.append(msg)
        print(msg)

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    return result
