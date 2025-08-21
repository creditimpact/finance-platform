# Field population runbook

This runbook explains how missing field data is escalated when automatic fillers
cannot populate a value.

## Error emission

Filler failures emit an audit event:

```
fields.populate_errors{tag, field, reason}
```

The event payload identifies the action tag, the missing field, and the reason
for failure. The field is recorded on the account context under
`missing_fields` for downstream consumers.

## Escalation

* **Critical fields** – `name`, `address`, `date_of_birth`, `ssn_masked`,
  `creditor_name`, `account_number_masked`, `inquiry_creditor_name`, and
  `inquiry_date`. When any of these are missing the planner defers the
  action tag and the user is prompted to supply the information.
* **Optional fields** – `days_since_cra_result`, `amount`, and `medical_status`.
  These fall back to a safe default template when missing and do not block
  processing.

The planner or letter router can reference `critical_missing_fields` on the
context to determine which pathway was taken.
