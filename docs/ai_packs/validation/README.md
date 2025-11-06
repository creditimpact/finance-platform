# Validation AI Pack Schema

The validation builder emits newline-delimited JSON (`.jsonl`) packs. Each line
represents a single weak field that requires adjudication and mirrors the
structure consumed by the merge pack flow.

## Input payload (pack line)

Every line produced by `ValidationPackWriter` serialises a mapping with the
following top-level keys:

| Key | Description |
| --- | --- |
| `id` | Stable identifier for the `(account, field)` pair (`acc_<ACCID>__<FIELDKEY>`). |
| `sid` / `account_id` / `account_key` | Run + account metadata to link the pack back to the run. |
| `field` / `field_key` | Human-readable and normalised field identifiers. |
| `category` | High-level grouping from the validation requirement (may be `null`). |
| `documents` | Normalised list of supporting documents requested for the field. |
| `min_days` | Minimum age in **calendar days** used by legacy consumers. |
| `min_days_business` | Minimum age in business days (source of truth for SLAs). |
| `duration_unit` | Unit associated with `min_days_business` (`business_days` today). |
| `strength` | Requirement strength normalised to `weak` or `soft`; strong items are filtered out. |
| `bureaus` | Per-bureau `raw` and `normalized` values for the field. |
| `context` | Supplemental consistency signals (consensus summary, disagreeing or missing bureaus, history snapshots, requirement notes, etc.). |
| `prompt` | The message payload we hand to the adjudication model. Contains `system`, `user`, and `guidance` keys so the line is self-contained. |
| `expected_output` | JSON schema specifying the response contract (see below). |

The `prompt.user` block echoes the metadata above (SID, account identifiers,
field identifiers, bureau values, and context) so downstream tooling can send
lines directly to the model without additional lookups.

`min_days` remains a calendar-day number so existing consumers do not need to
change units. The canonical SLA is tracked via `min_days_business` and
`duration_unit` so that downstream builders can migrate to business-day logic
without breaking older integrations.

## Output payload (result line)

Models must answer with a JSON object that satisfies the `expected_output`
schema embedded in the pack line:

* `decision`: either `strong` (consumer has a usable validation argument) or
  `no_case` (insufficient basis).
* `justification`: free-form explanation that justifies the decision.
* `labels`: non-empty array of strings tagging the drivers behind the call.
* `citations`: array of strings referencing the bureau facts relied upon.
* `confidence`: float between `0` and `1` indicating the model's self-assessed
  certainty. Low-confidence (`< AI_MIN_CONFIDENCE`) responses are downgraded to
  `no_case` by the sender guardrails.

Additional metadata (e.g., `model`, `request_lines`, timestamps) is attached by
our ingestion helpers when writing the `.result.jsonl` and `.result.json` files
to `runs/<SID>/ai_packs/validation/results/`.

## Decision labels quick reference

| Label | Meaning |
| --- | --- |
| `strong` | The bureau data supports moving forward with validation actions. |
| `no_case` | The evidence is insufficient or contradictory; pause automation and escalate. |

## Rationale and confidence expectations

Responses must include justification text, at least one supporting label, and a
confidence score. The sender rejects malformed outputs (missing required keys)
and downgrades `strong` decisions when the reported confidence falls below the
configured guardrail threshold.
