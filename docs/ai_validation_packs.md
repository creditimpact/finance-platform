# AI Validation Pack Layout

The AI validation pack flow mirrors the merge pack structure while consolidating
all account data for a run into a single folder tree. The target layout under a
run directory is:

```
runs/<SID>/ai_packs/validation/
  packs/
    val_acc_<ACCID>.jsonl
    # Optional (per-field variant):
    # val_acc_<ACCID>__field_<FIELDKEY>.jsonl
  results/
    val_acc_<ACCID>.result.json
    # Optional (per-field variant):
    # val_acc_<ACCID>__field_<FIELDKEY>.result.json
  index.json
  logs.txt
```

## Naming rules

* `<SID>` is the run identifier (same as other AI pack flows).
* `<ACCID>` identifies the account within the run. Use a single helper such as
  `fmt_accid(14) -> "014"` so all file names zero-pad to three digits where
  practical, keeping listings aligned. Padding is optional if a consumer requires
  the raw numeric string, but the helper should provide the consistent default.
* `<FIELDKEY>` is the snake_cased version of the field name (for example,
  `creditor_remarks`). It only appears when the optional per-field mode is
  enabled.
* Pack files use the `.jsonl` extension to hold one JSON line per weak field
  payload.
* AI results use the `.result.json` suffix to pair with their pack name.

## Pack granularity

* Default behavior writes **one pack per account**, bundling all weak
  `ai_needed` fields for that account. This keeps the pack directory concise and
  mirrors merge pack hygiene.
* A per-field mode remains supported behind a writer flag, emitting one pack per
  `(account, field)` pair using the same naming rules above.

## Index and manifest

* `index.json` catalogs every pack/result pair so downstream tooling can reason
  about validation coverage without scanning the filesystem.
* `logs.txt` captures builder activity, mirroring the merge flow.
* The run-level `manifest.json` should reference the `ai_packs.validation`
  locations so orchestration tools can discover validation artifacts the same
  way they do for merge packs.
