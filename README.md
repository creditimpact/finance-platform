# Finance Platform

This project ingests a PDF credit report and optional email, analyzes the report, highlights issues with explanations, builds a strategy, and generates guardrail-compliant letters as HTML/PDF alongside audit logs.

## Project Structure

```
backend/
  api/
  core/
    logic/
      report_analysis/
      strategy/
      letters/
      rendering/
      compliance/
      utils/
      guardrails/
  analytics/
  audit/
  assets/
    templates/
    static/
    fonts/
    data/
frontend/
tools/
scripts/
tests/
docs/
archive/
examples/
```

## Getting Started (Local)

### Backend (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
set FLASK_DEBUG=1
set DISABLE_PDF_RENDER=true
python -m backend.api.app
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Manual Scenario

1. Open http://localhost:5173.
2. Upload a credit report PDF (email optional).
3. Review the analysis and strategy.
4. Letters are written to the `examples/` folder or a configured output path.

## Environment Variables

Provide these in a `.env` file:

- `OPENAI_API_KEY` – required for LLM calls.
- `ADMIN_PASSWORD` – password for admin endpoints.
- `OPENAI_BASE_URL` – optional override for OpenAI's URL.

Secrets are never committed to the repository.

## PDF Rendering

PDF generation is disabled by default with `DISABLE_PDF_RENDER=true`. To enable PDFs, install [wkhtmltopdf](https://wkhtmltopdf.org/) and unset the flag.

## Folder READMEs

See the README files inside `backend/api`, `backend/core`, `backend/analytics`, `backend/audit`, `backend/assets`, and other folders for more details.

## Tests (optional for now)

```bash
DISABLE_PDF_RENDER=true python -m pytest -q
# or
python tools/import_sanity_check.py
```

## Contributing / Changelog

- [Contributing Guide](docs/CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## Pre-commit & Tests

- Install hooks: `pip install pre-commit && pre-commit install`
- Run import smoke: `python tools/import_sanity_check.py`
- Run tests: `DISABLE_PDF_RENDER=true python -m pytest -q`

