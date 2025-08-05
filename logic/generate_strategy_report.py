import os
import json
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class StrategyGenerator:
    """Generate an internal strategic analysis using GPT-4."""

    def generate(
        self,
        client_info: dict,
        bureau_data: dict,
        supporting_docs_text: str = "",
        run_date: str | None = None,
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
"""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = (
                content.replace("```json", "").replace("```", "").strip()
            )
        return json.loads(content)

    def save_report(
        self,
        report: dict,
        client_info: dict,
        run_date: str,
        base_dir: str = "Clients",
    ) -> Path:
        """Save the strategy JSON under the client's folder and return the path."""
        safe_name = (client_info.get("name") or "Client").replace(" ", "_").replace("/", "_")
        session_id = client_info.get("session_id", "session")
        folder = Path(base_dir) / datetime.now().strftime("%Y-%m") / f"{safe_name}_{session_id}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "strategy.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return path
