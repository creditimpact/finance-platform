# Strategy

## Purpose
Compute credit repair strategies from analyzed report data and other signals.

## Files
File | Role in this capability | Key imports / called by
--- | --- | ---
fallback_manager.py | selects fallback actions when strategy generation fails | compliance.constants
generate_strategy_report.py | high-level entry to assemble strategy report | strategy_engine, strategy_merger
strategy_engine.py | core engine deriving actions from analysis | rule_checker, utils
strategy_merger.py | combines outputs from multiple strategy components | strategy_engine
summary_classifier.py | classifies summaries to direct strategy paths | services.ai_client

## Entry points
- `generate_strategy_report.generate_strategy_report`
- TODO: expose `strategy_engine.run` or similar

## Dependencies
- **Internal**: `backend.core.logic.report_analysis`, `backend.core.logic.utils`, `backend.core.logic.compliance`
- **External**: standard library `typing`, `dataclasses`

## Notes / Guardrails
- Strategies must comply with regulatory requirements.
- Recommendations should remain factual and actionable.
