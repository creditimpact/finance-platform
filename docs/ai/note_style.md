# note_style AI pipeline

## Overview

The note_style stage builds pack files at `ai_packs/note_style/packs/` and sends
each pack to the OpenAI Chat Completions API. Successful responses are
normalized and written to `ai_packs/note_style/results/*.result.jsonl`. When a
response cannot be parsed, the pipeline records a failure entry alongside a raw
response snapshot in `ai_packs/note_style/results_raw/` for later debugging.

## Response modes

The request mode is controlled by the `NOTE_STYLE_RESPONSE_MODE` environment
variable. Supported values are `"tool"` and `"json"`. Legacy `"auto"` and
`"json_object"` values are still accepted and are normalized to `"json"`. The
default mode is `"json"`.

* `tool` &mdash; Always attaches the `submit_note_style_analysis` tool definition
  and instructs the model to answer via a tool call. Tool mode is additionally
  gated by the `NOTE_STYLE_ALLOW_TOOL_CALLS` feature flag (disabled by
  default).
* `json` &mdash; Sends a `response_format` payload (`{"type": "json_object"}` by
  default) and expects the model to respond directly in `message.content`.

The sender only includes the `response_format` parameter in the OpenAI request
when the effective mode is `json`. Tool mode exclusively relies on the
function-call contract and never sets `response_format`.

When `NOTE_STYLE_ALLOW_TOOL_CALLS` is `false` the pipeline forces JSON mode even
if `NOTE_STYLE_RESPONSE_MODE` is set to `"tool"`. Operators can temporarily set
`NOTE_STYLE_ALLOW_TOOL_CALLS=true` alongside `NOTE_STYLE_RESPONSE_MODE=tool` to
opt back into the legacy tool workflow.

For existing runs that still have `*.result` artifacts, use
`scripts/migrate_note_style_results_to_jsonl.py` to migrate them to
`*.result.jsonl` before enabling strict JSON mode.

These modes keep the legacy behavior working out of the box, while allowing
operators to enforce a particular interaction style when needed.

## Logging and debugging

Every model attempt emits a structured `NOTE_STYLE_MODEL_METRICS` log entry with
fields: `sid`, `account_id`, `model`, `mode`, `request_id`, `response_format`,
`tool_choice`, `prompt_tokens`, `response_tokens`, `parse_ok`, and
`retry_count`. These logs can be aggregated to monitor request health and token
usage.

When the sender writes a failure result it includes two debugging helpers:

* `raw_openai_mode` &mdash; Whether the API reply contained JSON content or a tool
  call.
* `raw_openai_payload_excerpt` &mdash; A truncated, redacted snapshot of the raw API
  payload.

The full raw response body is stored in
`ai_packs/note_style/results_raw/acc_<account>.raw.txt` and is never truncated.
Both the failure artifact and raw text snapshot make it easier to diagnose
parsing issues without rerunning the pack.
