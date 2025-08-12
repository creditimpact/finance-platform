import os
import logging
from dataclasses import dataclass

from backend.core.services.ai_client import AIConfig

_logger = logging.getLogger("config")
if not _logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    _logger.addHandler(handler)
_logger.setLevel(logging.INFO)


@dataclass
class AppConfig:
    """Application configuration loaded from the environment."""

    ai: AIConfig
    wkhtmltopdf_path: str
    rulebook_fallback_enabled: bool
    export_trace_file: bool
    smtp_server: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    celery_broker_url: str
    admin_password: str | None = None
    secret_key: str = "change-me"


def get_app_config() -> AppConfig:
    """Load and validate application configuration from environment variables."""

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    os.environ.setdefault("OPENAI_BASE_URL", base_url)

    wkhtmltopdf_path = os.getenv("WKHTMLTOPDF_PATH", "wkhtmltopdf")
    rulebook_fallback_enabled = os.getenv("RULEBOOK_FALLBACK_ENABLED", "1") != "0"
    export_trace_file = os.getenv("EXPORT_TRACE_FILE", "1") != "0"
    smtp_server = os.getenv("SMTP_SERVER", "localhost")
    smtp_port = int(os.getenv("SMTP_PORT", "1025"))
    smtp_username = os.getenv("SMTP_USERNAME", "noreply@example.com")
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    celery_broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    admin_password = os.getenv("ADMIN_PASSWORD")
    secret_key = os.getenv("SECRET_KEY", "change-me")

    _logger.info("OPENAI_BASE_URL=%s", base_url)
    _logger.info("OPENAI_API_KEY present=%s", bool(api_key))
    _logger.info("RULEBOOK_FALLBACK_ENABLED=%s", rulebook_fallback_enabled)
    _logger.info("EXPORT_TRACE_FILE=%s", export_trace_file)

    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")
    if "localhost" in base_url:
        raise EnvironmentError("OPENAI_BASE_URL points to localhost")

    ai_conf = AIConfig(
        api_key=api_key,
        base_url=base_url,
        chat_model=os.getenv("OPENAI_MODEL", "gpt-4"),
        response_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    )

    return AppConfig(
        ai=ai_conf,
        wkhtmltopdf_path=wkhtmltopdf_path,
        rulebook_fallback_enabled=rulebook_fallback_enabled,
        export_trace_file=export_trace_file,
        smtp_server=smtp_server,
        smtp_port=smtp_port,
        smtp_username=smtp_username,
        smtp_password=smtp_password,
        celery_broker_url=celery_broker_url,
        admin_password=admin_password,
        secret_key=secret_key,
    )


def get_ai_config() -> AIConfig:
    """Backward compatible helper returning the AI sub-config."""

    return get_app_config().ai
