# Testing and QA

Run the full test suite with:

```
pytest
```

The suite includes regression tests that cover:

- Sanitization of raw client explanations containing sensitive or emotional language
- Handling of missing or malformed structured summaries
- Fallback behavior for unrecognized dispute types

During letter generation, warnings and log entries are emitted for:

- `[PolicyViolation]` when raw notes are sanitized
- `[Sanitization]` for missing summaries
- `[Fallback]` when generic dispute content is used

Example warning output:

```
[Fallback] Unrecognized dispute type 'strange' for 'Bank A', using generic.
[PolicyViolation] Raw client notes provided for Experian; sanitized.
```

These logs provide an audit trail for compliance reviews.
