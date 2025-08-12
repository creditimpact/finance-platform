# Utils

## Purpose
Shared helpers for parsing, normalization, and file operations used across the logic package.

## Files
File | Role in this capability | Key imports / called by
--- | --- | ---
bootstrap.py | environment bootstrap helpers | various modules
file_paths.py | standardize file path management | rendering, compliance
inquiries.py | utilities for credit inquiries | report_analysis, letters
json_utils.py | safe JSON parsing/helpers | explanations_normalizer, report_analysis
names_normalization.py | normalize creditor & bureau names | process_accounts
note_handling.py | utilities for note fields | report_postprocessing
pdf_ops.py | PDF manipulation helpers | rendering.pdf_renderer
report_sections.py | constants for report sections | report_parsing
text_parsing.py | generic text parsing helpers | process_accounts, disputes

## Entry points
- `json_utils.parse_json`
- `names_normalization.normalize_creditor_name`
- TODO: document other commonly used helpers

## Dependencies
- **Internal**: standard library
- **External**: `pdfplumber` (for PDF helpers)

## Notes / Guardrails
- Keep utilities side-effect free where possible.
- Handle PII cautiously, especially in text and PDF utilities.
