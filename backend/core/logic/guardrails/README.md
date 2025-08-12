# Guardrails

## Purpose
Enforce content guardrails such as validation of summaries and generated text.

## Files
File | Role in this capability | Key imports / called by
--- | --- | ---
summary_validator.py | validate generated summaries for safety & tone | used by strategy and letters (TODO)

## Entry points
- `summary_validator.validate_summary` (TODO)

## Dependencies
- **Internal**: `backend.core.logic.utils`
- **External**: `pydantic` (TODO)

## Notes / Guardrails
- Guardrail checks must not leak sensitive data.
- Maintain neutral, non-adversarial messaging.
