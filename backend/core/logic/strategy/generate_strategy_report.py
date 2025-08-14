import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from backend.audit.audit import AuditLogger
from backend.core.logic.compliance.constants import StrategistFailureReason
from backend.core.logic.guardrails import fix_draft_with_guardrails
from backend.core.logic.utils.json_utils import parse_json
from backend.core.services.ai_client import AIClient


class StrategyGenerator:
    """Generate an internal strategic analysis using GPT-4."""

    def __init__(self, ai_client: AIClient):
        self.ai_client = ai_client

    def generate(
        self,
        client_info: dict,
        bureau_data: dict,
        supporting_docs_text: str = "",
        run_date: str | None = None,
        classification_map: Dict[str, Dict[str, Any]] | None = None,
        stage_2_5_data: Dict[str, Dict[str, Any]] | None = None,
        audit: AuditLogger | None = None,
    ) -> dict:
        """Return a strategy JSON object for internal analyst review."""
        run_date = run_date or datetime.now().strftime("%B %d, %Y")
        client_name = client_info.get("name", "Client")

        docs_section = (
            f"\nSupporting documents summary:\n{supporting_docs_text}"
            if supporting_docs_text
            else ""
        )


        policy_context: Dict[str, Dict[str, Any]] = {}
        if classification_map:
            for acc_id, cls in classification_map.items():
                policy_context.setdefault(acc_id, {}).update(
                    {
                        "category": cls.get("category"),
                        "legal_tag": cls.get("legal_tag"),
                        "dispute_approach": cls.get("dispute_approach"),
                        "tone": cls.get("tone"),
                    }
                )
        if stage_2_5_data:
            for acc_id, data in stage_2_5_data.items():
                policy_context.setdefault(acc_id, {}).update(
                    {
                        "legal_safe_summary": data.get("legal_safe_summary"),
                        "suggested_dispute_frame": data.get("suggested_dispute_frame", ""),
                        "rule_hits": data.get("rule_hits", []),
                        "needs_evidence": data.get("needs_evidence", []),
                        "red_flags": data.get("red_flags", []),
                        "prohibited_admission_detected": data.get(
                            "prohibited_admission_detected", False
                        ),
                        "rulebook_version": data.get("rulebook_version", ""),
                    }
                )
        policy_section = (
            "\nAccount policy context:\n" + json.dumps(policy_context, indent=2)
            if policy_context
            else ""
        )

        prompt = f"""
You are a credit repair strategist. Analyze the client's credit report data and propose a concise plan of action. Base all recommendations on the supplied classification and rule hits; do not contradict them.
Client name: {client_name}
Run date: {run_date}
{policy_section}
Credit report data:
{json.dumps(bureau_data, indent=2)}
{docs_section}

Return only a JSON object with this structure:
{{
  "overview": "...",
  "accounts": [{{
    "account_id": "",
    "name": "",
    "account_number": "",
    "status": "",
    "analysis": "",
    "recommendation": "",
    "alternative_options": [],
    "flags": [],
    "legal_safe_summary": "",
    "suggested_dispute_frame": "",
    "rule_hits": [],
    "needs_evidence": [],
    "red_flags": []
  }}],
  "global_recommendations": []
}}
Ensure the response is strictly valid JSON: all property names and strings in double quotes, no trailing commas or comments, and no text outside the JSON.
"""
        response = self.ai_client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        raw_content = response.choices[0].message.content
        if audit:
            audit.log_step("strategist_raw_output", {"content": raw_content})
        content = (raw_content or "").strip()
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
        report, error_reason = parse_json(content)
        expected_keys = {"overview", "accounts", "global_recommendations"}
        failure_reason = None
        if not raw_content:
            failure_reason = StrategistFailureReason.EMPTY_OUTPUT
        else:
            if error_reason is not None or not isinstance(report, dict) or not report:
                failure_reason = StrategistFailureReason.UNRECOGNIZED_FORMAT
            elif not expected_keys.issubset(report):
                failure_reason = StrategistFailureReason.SCHEMA_ERROR
        if audit and failure_reason:
            audit.log_step(
                "strategist_failure",
                {"failure_reason": failure_reason.value},
            )
        if isinstance(report, dict) and stage_2_5_data:
            for acc in report.get("accounts", []):
                acc_id = str(acc.get("account_id", ""))
                data = stage_2_5_data.get(acc_id)
                if not data:
                    continue
                acc.setdefault("legal_safe_summary", data.get("legal_safe_summary"))
                acc.setdefault(
                    "suggested_dispute_frame", data.get("suggested_dispute_frame", "")
                )
                acc.setdefault("rule_hits", data.get("rule_hits", []))
                acc.setdefault("needs_evidence", data.get("needs_evidence", []))
                acc.setdefault("red_flags", data.get("red_flags", []))
                acc.setdefault(
                    "prohibited_admission_detected",
                    data.get("prohibited_admission_detected", False),
                )
                acc.setdefault(
                    "rulebook_version", data.get("rulebook_version", "")
                )

        fix_draft_with_guardrails(
            json.dumps(report, indent=2),
            client_info.get("state"),
            {},
            client_info.get("session_id", ""),
            "strategy",
        )
        return report

    def save_report(
        self,
        report: dict,
        client_info: dict,
        run_date: str,
        base_dir: str = "Clients",
        stage_2_5_data: Dict[str, Dict[str, Any]] | None = None,
    ) -> Path:
        """Save the strategy JSON under the client's folder and return the path."""
        safe_name = (
            (client_info.get("name") or "Client").replace(" ", "_").replace("/", "_")
        )
        session_id = client_info.get("session_id", "session")
        folder = (
            Path(base_dir)
            / datetime.now().strftime("%Y-%m")
            / f"{safe_name}_{session_id}"
        )
        folder.mkdir(parents=True, exist_ok=True)
        if stage_2_5_data:
            for acc in report.get("accounts", []):
                acc_id = str(acc.get("account_id", ""))
                data = stage_2_5_data.get(acc_id)
                if data:
                    acc.setdefault("legal_safe_summary", data.get("legal_safe_summary"))
                    acc.setdefault(
                        "suggested_dispute_frame", data.get("suggested_dispute_frame", "")
                    )
                    acc.setdefault("rule_hits", data.get("rule_hits", []))
                    acc.setdefault("needs_evidence", data.get("needs_evidence", []))
                    acc.setdefault("red_flags", data.get("red_flags", []))
                    acc.setdefault(
                        "prohibited_admission_detected",
                        data.get("prohibited_admission_detected", False),
                    )
                    acc.setdefault(
                        "rulebook_version", data.get("rulebook_version", "")
                    )
        path = folder / "strategy.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return path
