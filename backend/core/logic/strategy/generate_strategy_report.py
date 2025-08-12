import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from backend.core.services.ai_client import AIClient

from backend.audit.audit import AuditLogger
from backend.core.logic.compliance.constants import StrategistFailureReason
from backend.core.logic.utils.json_utils import parse_json
from backend.core.logic.guardrails import fix_draft_with_guardrails


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

        prompt = f"""
You are a credit repair strategist. Analyze the client's credit report data and propose a concise plan of action.
Client name: {client_name}
Run date: {run_date}

Credit report data:
{json.dumps(bureau_data, indent=2)}
{docs_section}

Return only a JSON object with this structure:
{{
  "overview": "...",
  "accounts": [{{
    "name": "",
    "account_number": "",
    "status": "",
    "analysis": "",
    "recommendation": "",
    "alternative_options": [],
    "flags": []
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
        path = folder / "strategy.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return path
