# Tools

Utility scripts for development and maintenance.

## process_accounts.py

Process a SmartCredit analysis JSON report into bureau-specific payloads.

```bash
python tools/process_accounts.py path/to/analyzed_report.json output_dir
```

## replay_outcomes.py

Recompute outcome events from raw bureau reports for debugging.

```bash
python tools/replay_outcomes.py report1.json [report2.json ...]
```
