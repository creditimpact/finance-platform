import os
import logging
from typing import Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from .json_utils import parse_json

load_dotenv()

logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

_RULE_MAP = {
    "identity_theft": {
        "legal_tag": "FCRA §605B",
        "dispute_approach": "fraud_block",
        "tone": "urgent",
    },
    "not_mine": {
        "legal_tag": "FCRA §609(e)",
        "dispute_approach": "validation",
        "tone": "firm",
    },
    "goodwill": {
        "legal_tag": "FCRA §623(a)(1)",
        "dispute_approach": "goodwill_adjustment",
        "tone": "conciliatory",
    },
    "inaccurate_reporting": {
        "legal_tag": "FCRA §611",
        "dispute_approach": "reinvestigation",
        "tone": "professional",
    },
}

_STATE_HOOKS = {
    "CA": "California Consumer Credit Reporting Agencies Act",
    "NY": "New York FCRA Article 25",
}


def _heuristic_category(summary: Dict[str, Any]) -> str:
    text_bits = [
        summary.get("dispute_type", ""),
        summary.get("facts_summary", ""),
    ] + summary.get("claimed_errors", [])
    text = " ".join([t.lower() for t in text_bits if isinstance(t, str)])
    if "identity" in text or "stolen" in text:
        return "identity_theft"
    if "not mine" in text:
        return "not_mine"
    if "goodwill" in text:
        return "goodwill"
    return "inaccurate_reporting"


def classify_client_summary(summary: Dict[str, Any], state: str | None = None) -> Dict[str, str]:
    """Classify a structured summary into a dispute category and legal strategy.

    Attempts to use the OpenAI API when credentials are available; otherwise falls
    back to a lightweight keyword heuristic. The return value always contains the
    keys ``category``, ``legal_tag``, ``dispute_approach`` and ``tone``. A
    ``state_hook`` is included when a supported state modifier applies.
    """

    category = None
    if os.getenv("OPENAI_API_KEY"):
        prompt = (
            "Classify the following structured credit dispute summary into one of "
            "the categories: not_mine, inaccurate_reporting, identity_theft, goodwill. "
            "Return only JSON with a 'category' field. Summary: "
            f"{summary}"
        )
        try:
            resp = client.responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                input=prompt,
                response_format={"type": "json_object"},
            )
            content = resp.output[0].content[0].text
            data, _ = parse_json(content)
            data = data or {}
            category = data.get("category")
        except Exception:
            category = None
    if not category:
        category = _heuristic_category(summary)

    mapping = _RULE_MAP.get(category, _RULE_MAP["inaccurate_reporting"]).copy()
    result = {"category": category, **mapping}
    if state and state in _STATE_HOOKS:
        result["state_hook"] = _STATE_HOOKS[state]
    logger.info("Summary classification: %s -> %s", summary.get("account_id"), result)
    return result
