# Rendering

## Purpose
Render letters and instruction documents into final text and PDF outputs.

## Files
File | Role in this capability | Key imports / called by
--- | --- | ---
instruction_data_preparation.py | gather data for instruction documents | utils.file_paths
instruction_renderer.py | build textual instructions | instructions_generator
instructions_generator.py | orchestrate instruction generation flow | compliance.rule_checker
letter_rendering.py | assemble letter text into templates | utils.pdf_ops
pdf_renderer.py | convert text content to PDF files | external PDF libraries

## Entry points
- `letter_rendering.render_letter`
- `pdf_renderer.render_pdf`
- `instructions_generator.generate_instructions` (TODO)

## Dependencies
- **Internal**: `backend.core.logic.utils`, `backend.core.logic.compliance`, `backend.core.logic.letters`
- **External**: `pdfplumber` or similar PDF libs

## Notes / Guardrails
- Ensure generated documents avoid PII exposure.
- Output files should be sanitized and stored securely.
