# Report Analysis

## Purpose
Extract structured sections from SmartCredit reports and categorize accounts for downstream strategy and letter generation.

## Pipeline position
Ingests the uploaded PDF and produces bureau‑specific sections (`disputes`, `goodwill`, `inquiries`, etc.) consumed by strategy modules.

## Files
- `__init__.py`: package marker.
- `analyze_report.py`: orchestrates parsing, prompting, and post‑processing.
  - Key function: `analyze_credit_report()` runs AI analysis and merges parsed results.
  - Internal deps: `.report_parsing`, `.report_prompting`, `.report_postprocessing`, `backend.core.logic.utils.text_parsing`, `backend.core.logic.utils.inquiries`.
- `extract_info.py`: pull identity columns from the report.
  - Key functions: `extract_clean_name()`, `normalize_name_order()`, `extract_bureau_info_column_refined()`.
  - Internal deps: `backend.core.logic.utils.names_normalization`, `pdfplumber`.
- `process_accounts.py`: convert analysis output into bureau payloads.
  - Key items: `Account` dataclass; functions `process_analyzed_report()` and `save_bureau_outputs()`; helpers `infer_hardship_reason()`, `infer_personal_impact()`, `infer_recovery_summary()`.
  - Internal deps: `backend.core.logic.utils.names_normalization`, `backend.core.logic.strategy.fallback_manager`, `backend.audit.audit`.
- `report_parsing.py`: read PDF text and helper for converting dicts.
  - Key functions: `extract_text_from_pdf()`, `bureau_data_from_dict()`.
  - Internal deps: `pdfplumber`.
- `report_postprocessing.py`: clean and augment AI results.
  - Key functions: `_merge_parser_inquiries()`, `_sanitize_late_counts()`, `_cleanup_unverified_late_text()`, `_inject_missing_late_accounts()`, `validate_analysis_sanity()`.
  - Internal deps: `backend.core.logic.utils` modules.
- `report_prompting.py`: build LLM prompts and call the AI client.
  - Key function: `call_ai_analysis()`.
  - Internal deps: `backend.core.services.ai_client`.

## Entry points
- `analyze_report.analyze_credit_report`
- `process_accounts.process_analyzed_report`

## Guardrails / constraints
- Sanitization helpers ensure PII is normalized and late-payment data is validated.
- Summaries should remain factual and neutral.

## Output fields
The JSON produced by this stage may include informational fields that are not
yet consumed by downstream modules:

- `confidence`: heuristic confidence score for AI parsing.
- `needs_human_review`: flag indicating analysis uncertainty.
- `missing_bureaus`: list of bureaus absent from the source report.

These fields are optional and safely ignored by letter generation and
instructions pipelines.
