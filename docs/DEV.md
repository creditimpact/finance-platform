# Developer workflows

## Validation manifest v2 (Validation AI)

The validation sender now trusts a single manifest as the source of truth for
pack and result locations. When a run finishes, it writes
`runs/<SID>/ai_packs/validation/index.json`; both the pack builder and sender
use that manifest exclusively. Paths in schema v2 are **always POSIX-style and
relative to the manifest** so the index remains stable if the run directory is
moved (for example, when syncing `runs/` between machines or CI workers).

### Schema overview

Schema v2 documents look like the example below. Every relative path is resolved
from the directory that contains `index.json`.

```json
{
  "schema_version": 2,
  "sid": "41495514-931e-4100-90ca-4928464dcda8",
  "root": ".",
  "packs_dir": "packs",
  "results_dir": "results",
  "packs": [
    {
      "account_id": 1,
      "pack": "packs/val_acc_001.jsonl",
      "result_jsonl": "results/acc_001.result.jsonl",
      "result_json": "results/acc_001.result.json",
      "lines": 12,
      "status": "built",
      "built_at": "2024-06-18T02:14:03Z",
      "weak_fields": ["account_name"],
      "source_hash": "3a6408f2"
    }
  ]
}
```

Field reference:

| Field | Notes |
| --- | --- |
| `schema_version` | Must be `2` for relative-path manifests. |
| `sid` | Run identifier. |
| `root` | Common base directory shared by `packs_dir` and `results_dir`. Usually `"."`. |
| `packs_dir` / `results_dir` | Relative directories under `root` for pack inputs and model outputs. |
| `packs` | Array of per-account records. Each record stores relative paths for the pack payload and both result files, plus metadata such as `lines`, `status`, `built_at`, optional `weak_fields`, and any extra keys persisted by builders. |

### PowerShell quickstart (Windows)

Commands below assume the repository lives at `C:\dev\credit-analyzer` and the
virtual environment has already been created as `.venv`.

```powershell
cd C:\dev\credit-analyzer
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD"
$SID = "41495514-931e-4100-90ca-4928464dcda8"
python -m backend.validation.manifest --sid $SID --check
python -m backend.validation.send --sid $SID
```

`backend.validation.manifest` validates that every pack referenced in the
manifest exists. `backend.validation.send` reads only the manifest, prepares the
result directories if necessary, and writes the model responses next to the
referenced result paths.

### Expected command output

Successful manifest check:

```
Validation packs for SID 41495514-931e-4100-90ca-4928464dcda8:
ACCOUNT  PACK                      STATUS  LINES  RESULT_JSONL                 RESULT_JSON
------   ------------------------  ------  -----  ---------------------------  --------------------------
001      packs/val_acc_001.jsonl   OK      12     results/acc_001.result.jsonl results/acc_001.result.json
002      packs/val_acc_002.jsonl   OK      10     results/acc_002.result.jsonl results/acc_002.result.json

Manifest: index.json
Packs dir: packs
Results dir: results
All 2 packs present for SID 41495514-931e-4100-90ca-4928464dcda8.
```

Missing pack example (after deleting a pack):

```
Validation packs for SID 41495514-931e-4100-90ca-4928464dcda8:
ACCOUNT  PACK                      STATUS  LINES  RESULT_JSONL                 RESULT_JSON
------   ------------------------  ------  -----  ---------------------------  --------------------------
001      packs/val_acc_001.jsonl   OK      12     results/acc_001.result.jsonl results/acc_001.result.json
002      packs/val_acc_002.jsonl   MISSING 0      results/acc_002.result.jsonl results/acc_002.result.json

Manifest: index.json
Packs dir: packs
Results dir: results
Missing packs detected: 1 of 2.
```

Sender preflight summary when everything is present:

```
MANIFEST: runs/41495514-931e-4100-90ca-4928464dcda8/ai_packs/validation/index.json
PACKS: 2, missing: 0
RESULTS DIR: ok
[acc=001] pack=packs/val_acc_001.jsonl -> results/acc_001.result.jsonl, results/acc_001.result.json  (lines=12)
[acc=002] pack=packs/val_acc_002.jsonl -> results/acc_002.result.jsonl, results/acc_002.result.json  (lines=10)
```

If a pack is missing the sender reports it relative to the manifest:

```
MANIFEST: runs/41495514-931e-4100-90ca-4928464dcda8/ai_packs/validation/index.json
PACKS: 2, missing: 1
RESULTS DIR: ok
[acc=001] pack=packs/val_acc_001.jsonl -> results/acc_001.result.jsonl, results/acc_001.result.json  (lines=12)
[acc=002] pack=packs/val_acc_002.jsonl -> results/acc_002.result.jsonl, results/acc_002.result.json  (lines=0)  [MISSING: packs/val_acc_002.jsonl]
```

### Quick success checklist

1. Build or reuse a run so `runs/<SID>/ai_packs/validation/index.json` exists with `"schema_version": 2` and manifest paths that start with `packs/` and `results/`.
2. `python -m backend.validation.manifest --sid <SID> --check` should print the tabular summary and `All <N> packs present` when every file exists.
3. `python -m backend.validation.send --sid <SID>` must write new result files under the manifest-defined `results/` directory.
4. Remove or rename a pack and re-run `--check`; it should clearly report `MISSING` and exit with a non-zero status.
5. Run your usual pipeline command with `$env:AUTO_VALIDATION_SEND = "1"` (or set the variable in your orchestrator). The builder generates packs, the manifest stays relative, and the sender runs automatically using only the manifest for resolution.

