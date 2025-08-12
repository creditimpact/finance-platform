# Assets
## Purpose
Central home for templates, static files, fonts, and data fixtures.
## Subfolders / Key Files
- templates/ — HTML and PDF templates
- static/ — CSS, images, and other static resources
- fonts/ — font files used for PDF rendering
- data/ — JSON fixtures (no secrets)
## Entry Points
- TODO
## Internal Dependencies
- orchestrators and PDF utilities reference these paths (TODO)
## External Dependencies
- TODO
## Notes / Guardrails
- Set DISABLE_PDF_RENDER=1 to skip font-dependent PDF rendering when fonts are unavailable.
- Keep data/ free of credentials and other secrets.
