"""Telemetry helpers for parser audit instrumentation."""

from backend.core.case_store.telemetry import emit


def emit_parser_audit(
    *,
    session_id: str,
    pages_total: int,
    pages_with_text: int,
    pages_empty_text: int,
    extract_text_ms: int,
    call_ai_ms: int | None,
    fields_written: int | None,
    errors: str | None,
) -> None:
    """Emit a ``parser_audit`` telemetry event."""

    emit(
        "parser_audit",
        session_id=session_id,
        pages_total=pages_total,
        pages_with_text=pages_with_text,
        pages_empty_text=pages_empty_text,
        extract_text_ms=extract_text_ms,
        call_ai_ms=call_ai_ms,
        fields_written=fields_written,
        errors=errors,
    )

