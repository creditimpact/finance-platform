# Assets

## Purpose
Central home for templates, static files, fonts, and data files.

## Subfolders
- `templates/`: HTML and PDF templates used for rendering letters, emails, and other documents.
- `static/`: CSS, images, and other static resources for rendering.
- `fonts/`: Font files used during PDF generation.
- `data/`: JSON fixtures and similar resources. **Do not store secrets here.**

## Path considerations
Existing code may still look for assets in their previous top-level locations. Modules that reference these paths include:
- `backend/core/orchestrators.py`
- `backend/core/logic/instructions_generator.py`
- `backend/core/logic/utils/pdf_ops.py`
- `backend/core/logic/goodwill_rendering.py`
- `docs/MODULE_GUIDE.md`

## Notes / guardrails
- Set `DISABLE_PDF_RENDER=1` (or `true`/`yes`) to skip font-dependent PDF rendering when fonts are unavailable.
- Keep `data/` free of credentials and other secrets.
