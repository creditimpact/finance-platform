import os
import sys
import types

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_BASE_URL", "https://example.com/v1")

# Stub optional heavy dependencies to keep tests lightweight
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=lambda *_, **__: None))
sys.modules.setdefault(
    "pdfkit",
    types.SimpleNamespace(
        configuration=lambda *_, **__: None, from_string=lambda *_, **__: None
    ),
)
sys.modules.setdefault("fpdf", types.SimpleNamespace(FPDF=object))
sys.modules.setdefault("pdfplumber", types.SimpleNamespace(open=lambda *_, **__: None))
sys.modules.setdefault("fitz", types.SimpleNamespace(open=lambda *_, **__: None))
