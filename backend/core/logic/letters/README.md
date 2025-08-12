# Letters

## Purpose
Prepare and generate dispute and goodwill letters using LLM prompting and templates.

## Files
File | Role in this capability | Key imports / called by
--- | --- | ---
dispute_preparation.py | preprocess account data for disputes | utils.text_parsing
generate_custom_letters.py | create dispute letters for various scenarios | letter_generator
generate_goodwill_letters.py | produce goodwill request letters | goodwill_preparation, goodwill_prompting
goodwill_preparation.py | enrich accounts with hardship context | utils.names_normalization
goodwill_prompting.py | craft LLM prompts for goodwill letters | services.ai_client
goodwill_rendering.py | finalize goodwill letter text | rendering.letter_rendering
gpt_prompting.py | shared GPT prompt helpers for letters | services.ai_client
letter_generator.py | orchestrate letter creation | rendering.pdf_renderer
outcomes_store.py | persist generated letter outcomes | storage layer (TODO)
explanations_normalizer.py | sanitize and structure user-provided explanations | utils.json_utils

## Entry points
- `generate_custom_letters.generate_custom_letters`
- `generate_goodwill_letters.generate_goodwill_letters`
- `letter_generator.generate_letters` (TODO)
- `dispute_preparation.prepare_disputes` (TODO)

## Dependencies
- **Internal**: `backend.core.logic.report_analysis`, `backend.core.logic.rendering`, `backend.core.logic.utils`
- **External**: AI client, standard library

## Notes / Guardrails
- Maintain neutral, non-admitting tone in generated letters.
- Redact or sanitize any PII before sending to models or outputs.
