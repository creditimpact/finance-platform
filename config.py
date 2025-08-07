import os
import logging

from services.ai_client import AIConfig

RULEBOOK_FALLBACK_ENABLED = os.getenv("RULEBOOK_FALLBACK_ENABLED", "1") != "0"
EXPORT_TRACE_FILE = os.getenv("EXPORT_TRACE_FILE", "1") != "0"
WKHTMLTOPDF_PATH = os.getenv("WKHTMLTOPDF_PATH", "wkhtmltopdf")

_logger = logging.getLogger("config")
if not _logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    _logger.addHandler(handler)
_logger.setLevel(logging.INFO)


def get_ai_config() -> AIConfig:
    """Return AI configuration loaded from the environment."""

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    os.environ.setdefault("OPENAI_BASE_URL", base_url)

    _logger.info("OPENAI_BASE_URL=%s", base_url)
    _logger.info("OPENAI_API_KEY present=%s", bool(api_key))
    _logger.info("RULEBOOK_FALLBACK_ENABLED=%s", RULEBOOK_FALLBACK_ENABLED)
    _logger.info("EXPORT_TRACE_FILE=%s", EXPORT_TRACE_FILE)

    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")
    if "localhost" in base_url:
        raise EnvironmentError("OPENAI_BASE_URL points to localhost")

    return AIConfig(
        api_key=api_key,
        base_url=base_url,
        chat_model=os.getenv("OPENAI_MODEL", "gpt-4"),
        response_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    )
