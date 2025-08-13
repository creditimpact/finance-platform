# Stage 2.5 Deployment Guide

Stage 2.5 introduces normalization and rule evaluation of client statements so that strategy output and audit logs capture a "legal_safe_summary" along with rule metadata.

## Pre-deploy Checks
1. **Run unit tests** to ensure the normalizer and logging behave as expected:
   ```bash
   pytest tests/strategy/test_stage_2_5_pipeline.py tests/strategy/test_rule_logging.py
   ```
2. **Execute the minimal workflow integration test** to confirm Stage 2.5 data is persisted during the full pipeline:
   ```bash
   pytest tests/test_local_workflow.py::test_skip_goodwill_when_identity_theft
   ```

## Verifying in a Sandbox
1. **Process a sample SmartCredit report** through the CLI:
   ```bash
   python main.py path/to/report.pdf user@example.com
   ```
2. After the run, inspect the generated client folder under `Clients/<YYYY-MM>/<Client>_cli/`.
   - The `strategy.json` file should contain Stage 2.5 fields for each account:
     - `legal_safe_summary`
     - `rule_hits`
     - `needs_evidence`
     - `red_flags`
   - Accounts without statements will show `"legal_safe_summary": "No statement provided"` and `"rule_hits": []`.
3. Review audit logs (or analytics counters) for `rule_evaluated` events to ensure Stage 2.5 evaluations are logged.

Once these checks pass, Stage 2.5 can be promoted to production.
