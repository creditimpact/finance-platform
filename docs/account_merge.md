# Account Merge Scoring & Overrides

The problematic-account merge scorer compares each pair of candidate accounts by
building *per-account case folders* in Stage A, computing weighted similarity
scores across those folders, and then choosing an action (`auto`, `ai`, or
`different`). Only problematic accounts participate in this flow.

This document summarizes the configuration knobs that control the AI override
paths, including the new account-number trigger.

## Environment variables

| Env var | Default | Description |
| --- | --- | --- |
| `MERGE_ACCTNUM_TRIGGER_AI` | `any` | Minimum account-number match level that can lift a low overall score into the AI band. Accepted values: `exact`, `last6_bin`, `last6`, `any`. |
| `MERGE_ACCTNUM_MIN_SCORE` | `0.31` | Floor used when forcing an AI decision for account-number matches. The lifted score is the max of the part score, this floor, and `MERGE_AI_HARD_MIN`. |
| `MERGE_ACCTNUM_REQUIRE_MASKED` | `0` | When set to `1`, the override only fires if at least one side used a masked account number (e.g., `XXXX1234`). |

> **Note:** The balance-owed override continues to use its existing
> configuration (unchanged in this release). Account-number overrides run in
> addition to balance-driven overrides and both write their reasons into the
> case metadata.

## `acctnum_level`

During scoring we normalize account numbers (strip formatting characters,
capture the raw string, collapse repeated masks, and extract the trailing
digits). Each pair receives an `acctnum_level`:

- `exact` – The normalized digits are identical (after trimming leading zeros).
- `last6_bin` – Both the first six digits (BIN) and the last six digits match.
- `last6` – The numbers share the same last six digits but diverge elsewhere.
- `masked_match` – Both sides include masking characters with the same
  canonical mask signature and at least one side reveals digits.
- `none` – No usable match.

We also track `acctnum_masked_any`, a boolean indicating whether *either* side
included masking characters (`X`, `*`, `•`, etc.).

The account-number part score itself remains unchanged. Overrides act *after*
the weighted score is computed.

## How the override works

1. Compute the weighted score and baseline decision.
2. If the baseline decision is below `MERGE_AI_MIN`, check whether the balance
   override or the account-number override can lift it into the AI band.
3. For account numbers we evaluate the configured trigger (`MERGE_ACCTNUM_TRIGGER_AI`):
   - `off` disables the override.
   - `exact` requires an `acctnum_level` of `exact`.
   - `last6_bin` requires `acctnum_level == "last6_bin"`.
   - `last6` accepts `acctnum_level` values of `last6_bin` or `last6`.
   - `any` accepts `last6_bin`, `last6`, or `exact`.
4. When `MERGE_ACCTNUM_REQUIRE_MASKED=1`, the override only activates if
   `acctnum_masked_any` is `True`.
5. Eligible matches lift the score to `max(current_score, MERGE_ACCTNUM_MIN_SCORE, MERGE_AI_HARD_MIN)`
   and force an `ai` decision. This never violates `MERGE_AI_HARD_MIN` (currently `0.30`).
6. Override reasons are attached to the merge summary (`acctnum_only_triggers_ai`,
   `acctnum_match_level`, `acctnum_masked_any`, plus any balance-owed reasons).
7. When we enter the AI band, we emit a log entry and build `ai_pack.json` for
   downstream review. No external AI is invoked; the pack is a local artifact.

### Interaction with the balance-owed override

- Both overrides can activate on the same pair. Each reason is preserved under
  `override_reasons` so downstream tools can see *why* the score was lifted.
- Neither override bypasses the hard minimum. If `MERGE_AI_HARD_MIN` changes,
  keep both overrides’ `*_MIN_SCORE` defaults above it.

## Examples

These examples assume `MERGE_AI_MIN=0.35`, `MERGE_AI_HARD_MIN=0.30`, and
`MERGE_ACCTNUM_MIN_SCORE=0.31`.

1. **Exact match forces AI**

   - Config: `MERGE_ACCTNUM_TRIGGER_AI=any`, `MERGE_ACCTNUM_REQUIRE_MASKED=0`.
   - Scenario: Weighted score = `0.12`, `acctnum_level="exact"`, neither side masked.
   - Result: Score lifts to `0.31`, decision = `ai`, reasons include
     `acctnum_only_triggers_ai=True`.

2. **Masked last-six match triggers AI**

   - Config: `MERGE_ACCTNUM_TRIGGER_AI=last6`, `MERGE_ACCTNUM_REQUIRE_MASKED=1`.
   - Scenario: Weighted score = `0.18`, account A number `XXXX-567890`, account B `***-567890`.
   - Result: `acctnum_level="last6"`, `acctnum_masked_any=True`; score lifts to `0.31`
     (above `MERGE_AI_HARD_MIN`) and forces `ai`.

3. **Mask required but missing**

   - Config: `MERGE_ACCTNUM_TRIGGER_AI=last6`, `MERGE_ACCTNUM_REQUIRE_MASKED=1`.
   - Scenario: Weighted score = `0.22`, both sides expose the same trailing six digits with no masking.
   - Result: `acctnum_masked_any=False`; override does **not** trigger, so the pair
     stays below the AI band (unless the balance-owed override lifts it separately).

4. **Balance + account-number overrides**

   - Config: `MERGE_ACCTNUM_TRIGGER_AI=exact`, balance override enabled.
   - Scenario: Weighted score = `0.28`, `acctnum_level="exact"`, balance override also
     fires. Both reasons appear in `override_reasons`; decision remains `ai`.

These behaviors ensure deterministic handling of strong account-number signals
without altering the underlying part score.

## Telemetry & verification

- Candidate gating emits `CANDIDATE_LOOP_START`, `CANDIDATE_CONSIDERED`,
  `CANDIDATE_SKIPPED`, and `CANDIDATE_LOOP_END` lines. To review them for a
  specific session, run:

  ```bash
  rg "CANDIDATE_(CONSIDERED|SKIPPED)" runs/<sid>/ai_packs/logs.txt
  ```

- Account-number normalization logs the winning bureau pair via
  `MERGE_V2_ACCT_BEST`. Inspect those entries with:

  ```bash
  rg "MERGE_V2_ACCT_BEST" runs/<sid>/ai_packs/logs.txt
  ```

These commands make it easy to confirm which account pairs were considered,
why candidates were rejected, and which bureaus supplied the matched
account-number digits.
