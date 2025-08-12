# Credit Repair Cloud (Demo)

This repository contains a simplified demo of a credit repair automation flow used for testing.

## Documentation

- [System Overview](docs/SYSTEM_OVERVIEW.md)
- [Module Guide](docs/MODULE_GUIDE.md)
- [Data Models](docs/DATA_MODELS.md)
- [Contributing](docs/CONTRIBUTING.md)

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

### Upload step inputs

Only two fields are accepted when starting the process:

- **email** (optional)
- **file** – the credit report PDF

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

## Rulebook Overview

The system enforces guardrails defined in YAML files under the `rules/` directory:

- `rules/dispute_rules.yaml` – core systemic rules such as `RULE_NO_ADMISSION`,
  `RULE_PII_LIMIT`, and `RULE_NEUTRAL_LANGUAGE`.
- `rules/neutral_phrases.yaml` – predefined neutral statements referenced in summaries.
- `rules/state_rules.yaml` – state-specific clauses, disclosures, and prohibitions.

Each rule has a **severity** (`critical` or `warning`). Critical violations are
automatically fixed when a `fix_template` is provided; otherwise they are surfaced
to callers for remediation.

To add or modify a rule, edit the appropriate YAML file and include the new
`id`, `description`, `severity`, `block_patterns`, and optional `fix_template`.

## Letter Generation Flow

1. **Upload** – users upload a credit report PDF.
2. **Extraction** – accounts and explanations are parsed.
3. **Explanation Normalization** – user text is sanitized and paraphrased into a
   structured summary.
4. **Letter Generation** – templates are filled using the structured summary
   only; raw user text is never inserted.
5. **Guardrail Enforcement** – the draft letter passes through the rule checker
   which masks PII, replaces admissions, enforces neutral tone, and injects
   applicable state clauses or disclosures.
6. **Outcome Recording** – when results arrive from bureaus, outcomes are stored
   and can be exported weekly.

An overview diagram is available in `architecture.svg`.

## State Compliance

State rules are automatically applied based on the provided state code. Clauses
are appended for certain debt types (e.g., New York medical debt), disclosures
are added where required, and service is blocked in prohibited states such as
Georgia. Update `rules/state_rules.yaml` to introduce new clauses or
disclosures.

## Development Setup

### Backend

```bash
pip install -r requirements.txt
python app.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Environment Variables

Create a `.env` file in the project root. The most important variable is
`OPENAI_API_KEY`. A sample:

```env
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://api.openai.com/v1
```

## Testing

Run backend tests with a dummy API key to avoid live requests:

```bash
OPENAI_API_KEY=dummy pytest --maxfail=1 --disable-warnings -q
```

Frontend Jest tests:

```bash
cd frontend
npm test
```

## Extending the System

- **New dispute types** – add neutral phrasing in `neutral_phrases.yaml` and
  handle any bespoke rules in `dispute_rules.yaml`.
- **New state rules** – extend `state_rules.yaml` with clauses, disclosures, or
  prohibitions.
- **Additional compliance checks** – introduce new systemic rules in
  `dispute_rules.yaml` and corresponding tests.

## Contributing

See `CONTRIBUTING.md` for coding standards, testing requirements, and the pull
request process.
