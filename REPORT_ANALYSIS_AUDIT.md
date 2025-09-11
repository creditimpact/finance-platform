# Report Analysis Audit

## Problematic Accounts Source

Stage-A writes account summaries to `accounts_from_full.json` during report
analysis. After `TRACE_CLEANUP` the file remains under
`traces/blocks/<sid>/accounts_table/`.

### Flow
1. **Stage-A** – extracts accounts and saves `accounts_from_full.json`.
2. **Cleanup** – preserves the JSON alongside `_debug_full.tsv` and
   `general_info_from_full.json`.
3. **Problem Cases** – `build_problem_cases` loads
   `accounts_from_full.json` and writes case artifacts under
   `cases/<sid>/...`.

### Outputs
- `cases/<sid>/index.json` – summary of problematic accounts.
- `cases/<sid>/accounts/<account_id>.json` – individual case files.

### Optional Case Store
If a legacy Case Store session exists, the system reads problem accounts from
that path instead. The Case Store remains supported, but `accounts_from_full.json`
is the source of truth when it is absent.
