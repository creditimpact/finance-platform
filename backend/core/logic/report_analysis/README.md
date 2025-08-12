# Report Analysis

## Purpose
Analyze credit reports to extract structured information and categorize accounts for downstream processing.

## Files
File | Role in this capability | Key imports / called by
--- | --- | ---
analyze_report.py | orchestrates end-to-end report analysis | uses utils.text_parsing, services.ai_client
extract_info.py | pull structured facts from parsed report data | utils.json_utils
process_accounts.py | categorize accounts by bureau and tag goodwill/dispute items | utils.names_normalization, fallback_manager
report_parsing.py | parse uploaded report into intermediate model | utils.text_parsing
report_postprocessing.py | clean and enrich parsed report results | utils.note_handling
report_prompting.py | build LLM prompts for analyzing report content | services.ai_client

## Entry points
- `analyze_report.analyze_report`
- `process_accounts.process_analyzed_report`
- TODO: confirm other external calls

## Dependencies
- **Internal**: `backend.core.logic.utils`, `backend.core.services.ai_client`
- **External**: `dataclasses`, `json`

## Notes / Guardrails
- Maintain neutral tone when summarizing reports.
- Avoid exposing PII; rely on sanitization helpers when needed.
