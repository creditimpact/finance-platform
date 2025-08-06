# Credit Repair Cloud (Demo)

This repository contains a simplified demo of a credit repair automation flow used for testing.

## Environment

All backend components read configuration from a `.env` file on startup via
[python-dotenv](https://pypi.org/project/python-dotenv/). Provide
`OPENAI_API_KEY` and optionally `OPENAI_BASE_URL` in this file. When
`OPENAI_BASE_URL` is absent, the default `https://api.openai.com/v1` is used.
The Flask server, Celery workers and any CLI scripts load this file
automatically when they start.

## AI JSON Handling

OpenAI responses are parsed using a repair utility backed by the
[`dirtyjson`](https://pypi.org/project/dirtyjson/) library. When the model
returns nearly valid JSON (e.g., with trailing commas or single quotes), the
utility attempts to clean it so processing can continue without manual
intervention.

## Action Tags

Accounts may contain an `action_tag` field used to control which letters are generated. The allowed values are:

- `dispute` – generate dispute letters for the bureaus
- `goodwill` – create goodwill request letters to creditors
- `custom_letter` – produce a one-off custom letter
- `ignore` – no letters are generated

`action_tag` is preferred over the older `recommended_action` field. When both are present, the tag takes priority.

### Automatic Tagging

Accounts with derogatory statuses are automatically tagged for dispute. Both `process_analyzed_report()` and `merge_strategy_data()` assign `action_tag: "dispute"` whenever the status text contains keywords such as "collection", "chargeoff"/"charge off", "repossession", "repos", "delinquent", or "late payments", or when a `dispute_type` is present. This prevents obvious dispute items from being skipped even if the strategist omits a tag.

Goodwill letters are only generated for late-payment accounts that are not in collections, chargeoff, repossession, or other clearly derogatory statuses.

## Frontend

A React-based client is available in the `frontend/` directory for uploading PDF credit reports and tracking processing status.

### Running the frontend

```bash
cd frontend
npm install
npm run dev
```

The app runs on http://localhost:5173 and communicates with the Flask backend at http://localhost:5000.

## Celery Workers

Always start Celery from the project root so that Python can locate the
local modules:

```bash
cd path/to/finance-platform
celery -A tasks worker --loglevel=info
```

The worker bootstrap ensures the repository root is added to
`sys.path`, allowing modules such as `session_manager` to be imported
even if the current working directory differs.
