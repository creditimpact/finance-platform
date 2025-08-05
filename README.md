# Credit Repair Cloud (Demo)

This repository contains a simplified demo of a credit repair automation flow used for testing.

## Action Tags

Accounts may contain an `action_tag` field used to control which letters are generated. The allowed values are:

- `dispute` – generate dispute letters for the bureaus
- `goodwill` – create goodwill request letters to creditors
- `custom_letter` – produce a one-off custom letter
- `ignore` – no letters are generated

`action_tag` is preferred over the older `recommended_action` field. When both are present, the tag takes priority.

### Automatic Tagging

Accounts with derogatory statuses are automatically tagged for dispute. Both `process_analyzed_report()` and `merge_strategy_data()` assign `action_tag: "dispute"` whenever the status text contains keywords such as "collection", "chargeoff"/"charge off", "repossession", "repos", "delinquent", or "late payments", or when a `dispute_type` is present. This prevents obvious dispute items from being skipped even if the strategist omits a tag.

Goodwill letters are only generated for late-payment accounts that are not in collections, chargeoff, repossession, or other clearly derogatory statuses.
