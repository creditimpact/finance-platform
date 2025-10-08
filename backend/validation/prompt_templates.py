"""Prompt templates for the validation AI stage."""

from __future__ import annotations

import json
from typing import Any, Sequence

_VALIDATION_PROMPT_TEMPLATE = """SYSTEM:
You are an adjudication assistant for credit-report disputes. Use ONLY the JSON pack provided.
Do not assume facts not present. Output STRICT JSON only (one object), suitable for JSONL.

USER:
You must decide how actionable this field is for a consumer dispute against the bureaus.

Practical meaning of decisions:
- "strong": This field alone provides a sufficient, material basis to open a dispute that compels a bureau reinvestigation.
- "supportive": Not sufficient alone, but meaningfully strengthens a dispute when bundled with at least one other strong field.
- "neutral": Low added value; may be included for context only.
- "no_case": Do not use this field for a dispute.

Decision policy (apply in order):
1) Prefer normalized values when available; otherwise use raw.
2) Treat C5 (all different) as material unless differences are non-semantic/noisy.
3) Treat C4 (two match, one differs) as supportive by default; upgrade to strong if the differing value meaningfully changes consumer treatment (e.g., account_type).
4) History/timeline: If there’s a consistent span ≥ 18 months that helps anchor chronology, set modifiers.time_anchor=true (this alone does NOT make the field "strong").
5) If specific documents are essential to make the field actionable, set modifiers.doc_dependency=true.

You MUST output exactly one JSON object with the following shape:
{
  "sid": string,
  "account_id": number,
  "id": string,
  "field": string,
  "decision": "strong" | "supportive" | "neutral" | "no_case",
  "rationale": string,   // ≤120 words and MUST include the exact reason_code
  "citations": string[], // ≥1 items, each "<bureau>: <normalized OR raw>"
  "reason_code": string,
  "reason_label": string,
  "modifiers": {
    "material_mismatch": boolean,
    "time_anchor": boolean,
    "doc_dependency": boolean
  },
  "confidence": number    // 0.0–1.0
}

Context:
- sid: {{sid}}
- reason_code: {{reason_code}}
- reason_label: {{reason_label}}
- documents_required: {{documents | join(", ")}}

Field finding (verbatim JSON):
{{finding_json}}

Hard constraints:
- Output JSON only (no prose), ONE object.
- Rationale MUST contain {{reason_code}} literally.
- citations MUST be non-empty and name at least one bureau you relied on, e.g. "equifax: conventional real estate mortgage".
- If normalized is present, cite normalized; otherwise cite raw.

Rendering details:

Use your existing Jinja/string formatting to inject sid, reason_code, reason_label, documents, and finding_json (the exact finding blob).
"""


def _normalize_documents(documents: Any) -> list[str]:
    if isinstance(documents, str):
        text = documents.strip()
        return [text] if text else []
    if isinstance(documents, Sequence) and not isinstance(documents, (bytes, bytearray, str)):
        result: list[str] = []
        for entry in documents:
            if entry is None:
                continue
            try:
                text = str(entry).strip()
            except Exception:
                continue
            if text:
                result.append(text)
        return result
    return []


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _stringify_finding(finding: Any) -> str:
    if isinstance(finding, str) and finding.strip():
        return finding.strip()
    try:
        return json.dumps(finding or {}, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return json.dumps({}, ensure_ascii=False)


def _replace(template: str, **values: str) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def render_validation_prompt(
    *,
    sid: Any,
    reason_code: Any,
    reason_label: Any,
    documents: Any,
    finding: Any,
) -> tuple[str, str]:
    """Render the validation prompt using the provided context."""

    sid_text = _stringify(sid)
    reason_code_text = _stringify(reason_code)
    reason_label_text = _stringify(reason_label)
    documents_list = _normalize_documents(documents)
    documents_text = ", ".join(documents_list)
    finding_json = _stringify_finding(finding)

    rendered = _replace(
        _VALIDATION_PROMPT_TEMPLATE,
        sid=sid_text,
        reason_code=reason_code_text,
        reason_label=reason_label_text,
        documents=documents_text,
        finding_json=finding_json,
    )

    system_marker = "SYSTEM:\n"
    user_marker = "\nUSER:\n"

    if not rendered.startswith(system_marker):
        raise ValueError("Rendered prompt missing SYSTEM header")
    try:
        system_part, user_part = rendered.split(user_marker, 1)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError("Rendered prompt missing USER section") from exc

    system_prompt = system_part[len(system_marker) :]
    user_prompt = user_part

    return system_prompt, user_prompt


__all__ = ["render_validation_prompt"]

