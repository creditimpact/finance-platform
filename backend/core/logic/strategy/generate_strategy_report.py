import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from jsonschema import Draft7Validator, ValidationError

from backend.audit.audit import AuditLogger, emit_event
from backend.analytics.analytics_tracker import emit_counter
from backend.core.logic.compliance.constants import (
    StrategistFailureReason,
    normalize_action_tag,
)
from backend.core.logic.guardrails import fix_draft_with_guardrails
from backend.core.logic.utils.json_utils import parse_json
from backend.core.services.ai_client import AIClient


# Mapping of rule hits to required or forbidden recommendations
_RULE_ACTIONS: Dict[str, Dict[str, list[str]]] = {
    "no_goodwill_on_collections": {"forbidden": ["Goodwill"]},
    "fraud_flow": {"required": ["Fraud dispute"]},
}

# Default recommendation if the model suggests a forbidden action
_SAFE_FALLBACK_RECOMMENDATION = "Dispute with bureau"


_SCHEMA_PATH = Path(__file__).with_name("strategy_schema.json")
_SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
_VALIDATOR = Draft7Validator(_SCHEMA)


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
        rule_policies: Dict[str, Dict[str, Dict[str, list[str]]]] = {}
        if classification_map:
            for acc_id, cls in classification_map.items():
                cls_data = getattr(cls, "classification", cls)
                policy_context.setdefault(acc_id, {}).update(
                    {
                        "category": cls_data.get("category"),
                        "legal_tag": cls_data.get("legal_tag"),
                        "dispute_approach": cls_data.get("dispute_approach"),
                        "tone": cls_data.get("tone"),
                    }
                )
        if stage_2_5_data:
            for acc_id, data in stage_2_5_data.items():
                allowed_actions: list[str] = []
                forbidden_actions: list[str] = []
                rule_action_map: Dict[str, Dict[str, list[str]]] = {}
                for hit in data.get("rule_hits", []):
                    mapping = _RULE_ACTIONS.get(hit)
                    if mapping:
                        rule_action_map[hit] = mapping
                        allowed_actions.extend(mapping.get("required", []))
                        forbidden_actions.extend(mapping.get("forbidden", []))
                rule_policies[acc_id] = rule_action_map
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
                        "allowed_actions": allowed_actions,
                        "forbidden_actions": forbidden_actions,
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
        expected_keys = {"overview", "accounts", "global_recommendations"}
        report = None
        failure_reason = None
        for attempt in range(2):
            response = self.ai_client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            raw_content = response.choices[0].message.content
            if audit:
                audit.log_step(
                    "strategist_raw_output",
                    {"content": raw_content, "attempt": attempt + 1},
                )
            content = (raw_content or "").strip()
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "").strip()
            parsed, error_reason = parse_json(content)
            if not raw_content:
                current_reason = StrategistFailureReason.EMPTY_OUTPUT
            elif (
                error_reason is not None
                or not isinstance(parsed, dict)
                or not parsed
            ):
                current_reason = StrategistFailureReason.UNRECOGNIZED_FORMAT
            elif not expected_keys.issubset(parsed):
                current_reason = StrategistFailureReason.SCHEMA_ERROR
            else:
                report = parsed
                failure_reason = None
                break
            if failure_reason is None:
                failure_reason = current_reason

        if report is None:
            if audit and failure_reason:
                audit.log_step(
                    "strategist_failure",
                    {"failure_reason": failure_reason.value},
                )
            return self._build_minimal_strategy(
                bureau_data,
                stage_2_5_data or {},
                classification_map or {},
            )

        if stage_2_5_data:
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

                policy = rule_policies.get(acc_id, {})
                allowed_actions: list[str] = []
                forbidden_actions: list[str] = []
                for details in policy.values():
                    allowed_actions.extend(details.get("required", []))
                    forbidden_actions.extend(details.get("forbidden", []))
                acc.setdefault("allowed_actions", allowed_actions)
                acc.setdefault("forbidden_actions", forbidden_actions)

                recommendation = acc.get("recommendation", "")
                rec_lower = recommendation.lower()
                override = False
                enforced_rules: list[str] = []
                reason = ""

                for rule_id, details in policy.items():
                    for forbidden in details.get("forbidden", []):
                        if forbidden.lower() in rec_lower:
                            acc["recommendation"] = _SAFE_FALLBACK_RECOMMENDATION
                            override = True
                            enforced_rules.append(rule_id)
                            reason = f"{forbidden} forbidden by {rule_id}"
                            break
                    if override:
                        break

                if not override:
                    for rule_id, details in policy.items():
                        for required in details.get("required", []):
                            if required.lower() not in rec_lower:
                                acc["recommendation"] = required
                                override = True
                                enforced_rules.append(rule_id)
                                reason = f"{required} required by {rule_id}"
                                break
                        if override:
                            break

                if override:
                    acc["policy_override"] = True
                    acc["policy_override_reason"] = reason
                    acc["enforced_rules"] = enforced_rules

                rule_hits = acc.get("rule_hits", [])
                emit_counter("strategy.rule_hit_total", len(rule_hits))
                if acc.get("policy_override"):
                    emit_counter("strategy.policy_override_total")

                if audit:
                    audit.log_account(
                        acc_id,
                        {
                            "stage": "strategy_rule_enforcement",
                            "rule_hits": rule_hits,
                            "applied_rules": enforced_rules,
                            "policy_override": acc.get("policy_override", False),
                            "suggested_dispute_frame": acc.get(
                                "suggested_dispute_frame", ""
                            ),
                        },
                    )

                emit_event(
                    "strategy_rule_enforcement",
                    {
                        "account_id": acc_id,
                        "rule_hits": rule_hits,
                        "applied_rules": enforced_rules,
                        "policy_override": acc.get("policy_override", False),
                        "suggested_dispute_frame": acc.get(
                            "suggested_dispute_frame", ""
                        ),
                        "final_recommendation": acc.get("recommendation", ""),
                    },
                )

        guardrails_result = fix_draft_with_guardrails(
            json.dumps(report, indent=2),
            client_info.get("state"),
            {},
            client_info.get("session_id", ""),
            "strategy",
            ai_client=self.ai_client,
        )
        if guardrails_result:
            fixed_text, _, _ = guardrails_result
            try:
                report = json.loads(fixed_text)
            except Exception:
                pass
        return report

    def _build_minimal_strategy(
        self,
        bureau_data: Dict[str, Any],
        stage_2_5_data: Dict[str, Dict[str, Any]],
        classification_map: Dict[str, Dict[str, Any]],
    ) -> dict:
        """Construct a minimal strategy object from deterministic inputs."""

        accounts: Dict[str, Dict[str, Any]] = {}
        for payload in bureau_data.values():
            for items in payload.values():
                if not isinstance(items, list):
                    continue
                for entry in items:
                    acc_id = str(entry.get("account_id", ""))
                    if not acc_id or acc_id in accounts:
                        continue
                    accounts[acc_id] = {
                        "account_id": acc_id,
                        "name": entry.get("name", ""),
                        "account_number": entry.get("account_number", ""),
                        "status": entry.get("status", ""),
                        "analysis": "",
                        "recommendation": "",
                        "alternative_options": [],
                        "flags": [],
                        "legal_safe_summary": "",
                        "suggested_dispute_frame": "",
                        "rule_hits": [],
                        "needs_evidence": [],
                        "red_flags": [],
                        "prohibited_admission_detected": False,
                        "rulebook_version": "",
                        "action_tag": "",
                        "priority": "",
                        "legal_notes": [],
                        "enforced_rules": [],
                        "policy_override_reason": "",
                    }

        for acc_id, acc in accounts.items():
            data_25 = stage_2_5_data.get(acc_id, {})
            for key in (
                "legal_safe_summary",
                "suggested_dispute_frame",
                "rule_hits",
                "needs_evidence",
                "red_flags",
                "prohibited_admission_detected",
                "rulebook_version",
            ):
                if key in data_25:
                    acc[key] = data_25.get(key, acc[key])

            cls_record = classification_map.get(acc_id)
            cls = getattr(cls_record, "classification", cls_record) or {}
            if cls.get("flags"):
                acc["flags"] = cls.get("flags", [])
            raw_tag = cls.get("action_tag")
            if raw_tag:
                tag, _ = normalize_action_tag(raw_tag)
                acc["action_tag"] = tag
            acc["priority"] = cls.get("priority", acc["priority"])
            acc["legal_notes"] = cls.get("legal_notes", acc["legal_notes"])

        return {
            "overview": "",
            "accounts": list(accounts.values()),
            "global_recommendations": [],
        }

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
        for acc in report.get("accounts", []):
            acc_id = str(acc.get("account_id", ""))
            if stage_2_5_data:
                data = stage_2_5_data.get(acc_id, {})
            else:
                data = {}
            acc.setdefault("legal_safe_summary", data.get("legal_safe_summary", ""))
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
            acc.setdefault("rulebook_version", data.get("rulebook_version", ""))

            # Stage 2 defaults
            acc.setdefault("account_number", "")
            acc.setdefault("status", "")
            acc.setdefault("analysis", "")
            acc.setdefault("recommendation", "")
            acc.setdefault("alternative_options", [])
            acc.setdefault("flags", [])

            # Enforcement metadata
            tag, _ = normalize_action_tag(acc.get("recommendation"))
            acc.setdefault("action_tag", tag)
            acc.setdefault("priority", "")
            acc.setdefault("legal_notes", [])
            acc.setdefault("enforced_rules", acc.get("enforced_rules", []))
            acc.setdefault("policy_override_reason", acc.get("policy_override_reason", ""))

            # Ensure optional enforcement flag exists
            acc.setdefault("policy_override", acc.get("policy_override", False))

        try:
            _VALIDATOR.validate(report)
        except ValidationError as exc:
            raise ValueError(f"strategy schema validation failed: {exc.message}") from exc
        path = folder / "strategy.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return path
