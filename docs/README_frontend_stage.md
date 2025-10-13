# Frontend Stage README

The frontend stage builds lightweight account packs that power customer-facing experiences. These packs mirror the case artifacts produced earlier in the run so they can be regenerated without coordinating with merge or validation.

## Pack Data Sources
- **`holder_name`** – pulled from `cases/accounts/<N>/meta.json` (`heading_guess`). If the heading is missing, fall back to parsing `cases/accounts/<N>/raw_lines.json`. Never read `bureaus.json` for this field.
- **`primary_issue`** – the first issue entry in `cases/accounts/<N>/tags.json` (`{"kind": "issue", "type": ...}`). Additional issues may also be emitted in an `issues` array for UI enrichment.
- **Other account details** – flow through directly from existing case artifacts in `cases/accounts/<N>/` so that the packs remain consumer-safe and omit tolerance/debug-only state.

## Field Guarantees
- Every account with a case has at least one issue tag, so `primary_issue` is always populated.
- `holder_name` may be `null` when neither `meta.json` nor the raw heading lines produce a usable value.
- All legacy fields in the pack schema are preserved; newly added fields only extend the payload.

## Idempotency & Runflow Signals
- The generator only reads previously materialised case artifacts, so rerunning the stage reuses the same inputs and produces identical pack files and index counts.
- Runflow records the count of `runs/<SID>/frontend/accounts/*/pack.json` files and any creation errors. No entries are emitted when the stage simply revalidates existing packs.
- Because the stage is decoupled from merge/validation, operators can rebuild the frontend packs independently after correcting case artifacts.
