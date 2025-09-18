# Analyzer Inputs and Triad Adapter

This document explains what the problem detector expects as input and how we
adapt triad-shaped Stage‑A account data to the flat fields used by the rules
engine.

## What the detector consumes

- Entry point: `evaluate_account_problem(fields: dict) -> dict | None`
- Input shape: a flat mapping of normalized fields describing an account.
- In `detect_problem_accounts(sid)`, we use `account["fields"]` if present.
  Otherwise, we derive a flat mapping from the triad data with
  `build_rule_fields_from_triad(account)`.

## Fields produced by the triad adapter

The adapter returns the following keys:

- `past_due_amount` (float | null)
- `balance_owed` (float | null)
- `credit_limit` (float | null)
- `payment_status` (str | null)
- `account_status` (str | null)
- `days_late_7y` (int)
- `has_derog_2y` (bool)
- `account_type` (str | null)
- `creditor_remarks` (str | null)

These are passed unchanged to `evaluate_account_problem`.

## How values are derived

- Bureau precedence: If `account.triad.order` is present, use that
  order; otherwise fallback to `transunion → experian → equifax`.
- Status picking: For each textual field (e.g., `payment_status`,
  `account_status`, `creditor_remarks`, `account_type`), pick the first
  non-empty, non-"--" value following the bureau precedence.
- Currency parsing: Numbers like `"$12,091"` are normalized by removing
  any character except digits, decimal point, and minus sign, then parsed
  as float. Blank or invalid values produce `null`.
- 2‑year derogatory flag (`has_derog_2y`): True if any token in any
  bureau’s `two_year_payment_history` is not equal to `OK` (case-insensitive).
- 7‑year late days (`days_late_7y`): For each bureau, compute
  `late30 + late60 + late90` from `seven_year_history` and take the
  maximum across bureaus (conservative aggregation).

## Example

Given this triad snippet:

```json
{
  "triad": {"order": ["experian", "equifax", "transunion"]},
  "triad_fields": {
    "transunion": {"payment_status": "Current", "past_due_amount": "$0", "credit_limit": "$2,500"},
    "experian":   {"payment_status": "Late",    "past_due_amount": "$12,091", "credit_limit": "$2,600"},
    "equifax":    {"payment_status": "OK",      "past_due_amount": "$0", "credit_limit": "$2,700"}
  },
  "two_year_payment_history": {"experian": ["OK", "30", "OK"]},
  "seven_year_history": {
    "transunion": {"late30": 1, "late60": 0, "late90": 0},
    "experian":   {"late30": 0, "late60": 2, "late90": 0},
    "equifax":    {"late30": 0, "late60": 0, "late90": 3}
  }
}
```

The adapter yields:

```json
{
  "past_due_amount": 12091.0,
  "credit_limit": 2600.0,
  "payment_status": "Late",
  "days_late_7y": 3,
  "has_derog_2y": true
}
```

## Implementation notes

- Loader: accounts are read from the run manifest via
  `traces.accounts_table.accounts_json` (no PDF or legacy `traces/blocks` paths).
- Adapter and detector live in
  `backend/core/logic/report_analysis/problem_extractor.py`.
- The rules in `problem_detection` are unchanged; only inputs are adapted when
  `fields` are missing in the Stage‑A account object.

## Decision rules & thresholds

The analyzer applies simple, semantic rules to the flat `fields` mapping in
`backend/core/logic/report_analysis/problem_extractor.py`:

- Numeric thresholds:
  - `past_due_amount > 0` → add reason `past_due_amount:<value>` and consider delinquent.
  - `days_late_7y >= 1` → add reason `late_history: days_late_7y=<n>`.

- Status tokens (case-insensitive, substring match):
  - BAD_PAYMENT = {`late`, `delinquent`, `past due`, `charge-off`, `collection`, `derog`, `120`, `150`, `co`}
    - If `payment_status` contains any, add `bad_payment_status:<value>`.
  - BAD_ACCOUNT = {`collections`, `charge-off`, `charged off`, `repossession`, `foreclosure`}
    - If `account_status` contains any, add `bad_account_status:<value>`.

- Optional consistency check:
  - If `balance_owed > 0` and `account_status == "Closed"`, add `positive_balance_on_closed`.

- Primary issue precedence (first match wins):
  1) `charge_off` / `collection` (from status tokens)
  2) `delinquency` (from `past_due_amount > 0`)
  3) `late_history` (from `days_late_7y >= 1`)
  4) `status` (other BAD status tokens)

- Emission gate:
  - A candidate is emitted only if `problem_reasons` is non-empty.

### Provenance

The adapter returns `(fields, prov)` where `prov` maps each derived field to the
source bureau, e.g., `{"payment_status":"experian", "past_due_amount":"transunion"}`.
The detector augments `reason.debug.signals` with bureau-tagged strings like:

- `past_due_amount:12091.00 (bureau=experian)`
- `payment_status:120 (bureau=experian)`
- `account_status:collections (bureau=transunion)`

This helps explain which bureau supplied each triggering value.

## Problematic account merge stage

After problematic accounts are detected, we run a deterministic merge stage
to collapse near-duplicate accounts (e.g., an original charge-off and its
collection tradeline) before building case folders. The scorer compares every
pair of candidates from the same run/SID and assigns a weighted similarity
score in the range 0–1.

### Scoring features and weights

The scorer computes five partial scores and combines them with configurable
weights. Missing or malformed values simply contribute `0.0` for that part, so
only fields present on both sides influence the final score.

| Feature  | Default weight | Compared fields | Notes |
|----------|----------------|-----------------|-------|
| `acct` | 0.25 | `account_number`, `acct_num`, `number`, `account_number_display` | Exact match scores 1.0. Matching only the last four digits scores 0.7; otherwise 0.0.【F:backend/core/logic/report_analysis/account_merge.py†L334-L365】 |
| `dates`  | 0.20 | `date_opened`, `date_of_last_activity`, `closed_date` | Dates are parsed (`dd.mm.yyyy` tolerant of `/` or `-`). Each aligned pair contributes up to 1.0, scaled down linearly when the difference grows toward one year.【F:backend/core/logic/report_analysis/account_merge.py†L368-L400】 |
| `balowed` | 0.25 | `past_due_amount`, `balance_owed` | Currency strings are normalized, then each pair contributes based on relative difference. Missing values yield no contribution.【F:backend/core/logic/report_analysis/account_merge.py†L403-L452】 |
| `status` | 0.20 | `payment_status`, `account_status` | Strings are normalized and bucketed (collection, delinquent, paid, current, closed, bankruptcy). Any shared bucket yields 1.0; otherwise 0.0.【F:backend/core/logic/report_analysis/account_merge.py†L455-L486】 |
| `strings` | 0.10 | `creditor`, `remarks` | Lowercased creditor and remark text are concatenated and compared with `SequenceMatcher` for a fuzzy 0–1 ratio.【F:backend/core/logic/report_analysis/account_merge.py†L489-L505】 |

The total score is the weighted average of the five parts, and `score_accounts`
also returns the per-part contributions so they can be logged alongside the
overall value.【F:backend/core/logic/report_analysis/account_merge.py†L508-L529】

### Thresholds and AI gray band

Decisions are derived from the final score using the following default policy:

- `score ≥ 0.78` → `decision="auto"` (auto-merge pair).
- `0.35 ≤ score < 0.78` → `decision="ai"` (gray band; we only mark it for a
  later human/AI review).
- `score < 0.30` → `decision="different"` (never escalated to AI).

The thresholds and weights can be overridden via environment variables such as
`MERGE_AUTO_MIN`, `MERGE_AI_MIN`, or `MERGE_W_ACCT`, but the defaults above are
applied when no overrides are present.【F:backend/core/logic/report_analysis/account_merge.py†L31-L223】【F:backend/core/logic/report_analysis/account_merge.py†L532-L549】

### Example: account 11 vs 16

Our reference pair (account 11 vs 16: original card → later collection
tradeline) shares the exact account number, near-identical open/last-activity
dates, matching collection statuses, similar balances, and overlapping creditor
text. That produces per-part scores roughly `acct=1.0`, `dates≈0.9`,
`balowed≈0.8`, `status=1.0`, `strings≈0.6`. Applying the default weights yields a
final score around `0.84–0.88`, so both sides get a `decision="auto"` and fall
into the same merge group.

### Observability and logs

- Pairwise scoring emits `MERGE_SCORE sid=<...> i=<i> j=<j> parts=<...> score=<...>`
  followed by `MERGE_DECISION sid=<...> i=<i> j=<j> decision=<...> score=<...>` for every comparison. Use ripgrep to
  inspect them, e.g. `rg "MERGE_DECISION" runs/<sid>/ -g"*.log"`.
- At the end of a run we log `MERGE_SUMMARY sid=<...> clusters=<...>
  auto_pairs=<...> ai_pairs=<...> skipped_pairs=<...>` summarizing the merge
  graph.

Both log lines come from `cluster_problematic_accounts`, so they are emitted in
the same deterministic order as the pairwise loop.【F:backend/core/logic/report_analysis/account_merge.py†L794-L902】

### Where `merge_tag` is stored

For each problematic account, `cluster_problematic_accounts` attaches a
`merge_tag` with the group id, final decision, sorted `score_to` list, best
match, and per-part scores.【F:backend/core/logic/report_analysis/account_merge.py†L881-L892】 The case builder then persists that tag into
`runs/<sid>/cases/accounts/<account_id>/summary.json` alongside the rest of the
per-account summary payload.【F:backend/core/logic/report_analysis/problem_case_builder.py†L224-L290】 This keeps merge context available for downstream review and auditing.
