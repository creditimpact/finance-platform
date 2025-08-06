import os
import logging
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"

# Ensure the fallback is visible to later getenv calls
os.environ.setdefault("OPENAI_BASE_URL", OPENAI_BASE_URL)

# Basic logging to confirm configuration
_logger = logging.getLogger("config")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    _logger.addHandler(_handler)
_logger.setLevel(logging.INFO)
_logger.info("OPENAI_BASE_URL=%s", OPENAI_BASE_URL)
_logger.info("OPENAI_API_KEY present=%s", bool(OPENAI_API_KEY))

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is not set")
if "localhost" in OPENAI_BASE_URL:
    raise EnvironmentError("OPENAI_BASE_URL points to localhost")
