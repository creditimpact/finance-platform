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
| `min_days` | Optional minimum-age requirement for the documents. |
| `strength` | Requirement strength normalised to `weak` or `soft`; strong items are filtered out. |
| `bureaus` | Per-bureau `raw` and `normalized` values for the field. |
| `context` | Supplemental consistency signals (consensus summary, disagreeing or missing bureaus, history snapshots, requirement notes, etc.). |
| `prompt` | The message payload we hand to the adjudication model. Contains `system`, `user`, and `guidance` keys so the line is self-contained. |
| `expected_output` | JSON schema specifying the response contract (see below). |

The `prompt.user` block echoes the metadata above (SID, account identifiers,
field identifiers, bureau values, and context) so downstream tooling can send
lines directly to the model without additional lookups.

## Output payload (result line)

Models must answer with a JSON object that satisfies the `expected_output`
schema embedded in the pack line:

* `decision`: either `strong` (consumer has a usable validation argument) or
  `no_case` (insufficient basis).
* `rationale`: free-form explanation that justifies the decision.
* `citations`: array of strings referencing the bureau facts relied upon.
* `confidence` *(optional)*: float between `0` and `1` indicating the model's
  self-assessed certainty. The builder and result ingesters gracefully handle its
  absence.

Additional metadata (e.g., `model`, `request_lines`, timestamps) is attached by
our ingestion helpers when writing the `.result.json` files to
`runs/<SID>/ai_packs/validation/results/`.

## Decision labels quick reference

| Label | Meaning |
| --- | --- |
| `strong` | The bureau data supports moving forward with validation actions. |
| `no_case` | The evidence is insufficient or contradictory; pause automation and escalate. |

## Rationale and confidence expectations

Responses should always provide a concise rationale describing the deciding
facts. When available, include the optional `confidence` value so analysts can
triage borderline calls quickly. Lack of confidence simply means the model was
unable or unwilling to provide an estimate.
