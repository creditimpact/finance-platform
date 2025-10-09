import os


def sanitize_openai_env() -> None:
    """
    Normalize OpenAI env vars for all processes (Flask & Celery) before any client is created:
    - Trim whitespace
    - Fill default OPENAI_BASE_URL
    - Enforce presence of OPENAI_PROJECT_ID when using sk-proj-* keys
    """
    key = (os.getenv("OPENAI_API_KEY", "") or "").strip()
    proj = (os.getenv("OPENAI_PROJECT_ID", "") or "").strip()
    base = (os.getenv("OPENAI_BASE_URL", "") or "").strip() or "https://api.openai.com/v1"

    # Re-write sanitized values back to the environment
    if key:
        os.environ["OPENAI_API_KEY"] = key
    if proj:
        os.environ["OPENAI_PROJECT_ID"] = proj
    os.environ["OPENAI_BASE_URL"] = base

    # Hard requirement: sk-proj-* keys must have a project id
    if key.startswith("sk-proj-") and not proj:
        raise RuntimeError(
            "OPENAI_PROJECT_ID is required when using sk-proj-* API keys"
        )
