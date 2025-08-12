# Compliance

## Purpose
Enforce regulatory rules and validate user-provided documents to ensure safe operations.

## Files
File | Role in this capability | Key imports / called by
--- | --- | ---
compliance_adapter.py | interface to external compliance services | compliance_pipeline
compliance_pipeline.py | orchestrates compliance checks across modules | rule_checker, rules_loader
constants.py | shared constants for compliance rules | --
rule_checker.py | evaluate data against compliance rules | rules_loader
rules_loader.py | load rule definitions and constants | constants
upload_validator.py | validate uploaded PDFs for size & safety | pdfplumber

## Entry points
- `compliance_pipeline.run_compliance_pipeline`
- `rule_checker.check_rules`
- `upload_validator.is_safe_pdf`

## Dependencies
- **Internal**: `backend.core.logic.utils`, `backend.core.logic.guardrails`
- **External**: `pdfplumber`

## Notes / Guardrails
- Ensure rules remain current with regulations.
- Never retain uploaded documents longer than necessary.
