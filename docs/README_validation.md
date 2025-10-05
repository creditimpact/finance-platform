# Validation Workflow Overview

The validation pipeline evaluates bureau discrepancies in two passes: deterministic
screening to detect "missing" reports versus "mismatched" values, followed by
optional AI adjudication for semantic disagreements. All logic shares the same
normalisation rules so that bureau data is compared on an even footing before
any escalation decisions are made.

## Findings without AI

* **Missing** – Only the 18 deterministic fields participate. When at least one
  bureau reports a value and another bureau omits it entirely we create a
  *Missing* finding. Packs are never created for these cases and the AI path is
  skipped entirely.
* **Deterministic mismatches** – Amounts, dates, enumerations and history blocks
  are reconciled with pure code. Normalisation includes tolerance windows (see
  below) so that equivalent reporting formats do not escalate.

## AI escalation

Only three semantic fields may route to AI: `account_type`, `creditor_type` and
`account_rating`. After normalisation, a mismatch on any of these fields sets
`send_to_ai=true` and generates a validation pack when the validation packs
feature flag is enabled. The associated reason codes are limited to the C3/C4/C5
series and packs are never generated for missing data.

## Environment configuration

The behaviour of the validation pipeline can be tuned through the following
environment variables. Each reader falls back to the documented default when no
value is supplied.

| Variable | Default | Description |
| --- | --- | --- |
| `AMOUNT_TOL_ABS` | `50` | Absolute USD tolerance for amount mismatches. |
| `AMOUNT_TOL_RATIO` | `0.01` | Relative ratio tolerance for amount mismatches. |
| `LAST_PAYMENT_DAY_TOL` | `5` | Day window applied when comparing payment dates. |
| `VALIDATION_PACKS_ENABLED` | `1` | Toggle to build validation packs. |
| `VALIDATION_REASON_ENABLED` | `1` | Enables reason capture and observability logging. |
| `VALIDATION_INCLUDE_CREDITOR_REMARKS` | `0` | Optional toggle to include `creditor_remarks` validation (disabled by default). |

Boolean toggles accept the standard set of truthy values (`1`, `true`, `yes`,
`on`, `y`) and falsy values (`0`, `false`, `no`, `off`, `n`). Unrecognised inputs
fall back to their defaults so a misconfigured deployment will not disable
critical observability.

These flags are consumed by both the deterministic merge layer and the AI pack
builders. For example, tolerance defaults flow into the merge configuration used
by `account_merge.get_merge_cfg`, while `_reasons_enabled()` and
`_packs_enabled()` in the AI path read their respective toggles when deciding
whether to attach packs, send them automatically, and emit reason codes.

## History normalisation

Payment history blocks are compared after removing empty dictionaries so that
missing data is not incorrectly treated as a mismatch. This keeps the focus on
substantive discrepancies and aligns with our policy to exclude `creditor_remarks`
from the validation scope entirely.

For deeper implementation details explore the modules under
`backend/core/logic/report_analysis/` and `backend/ai/` which contain the
configuration readers and pack builders.
