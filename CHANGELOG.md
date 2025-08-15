# Changelog

## Unreleased
### Added
- Analysis prompt now instructs models to respond with valid JSON only, using `null` or empty arrays when data is unknown and avoiding invented fields. Bump `ANALYSIS_PROMPT_VERSION` to 2.
- Strategy snapshot now records rulebook version, rule hits, evidence needs, and red flags and emits structured audit events.
- Document deterministic Stage 2.5 rollout with explicit rule evaluation order, precedence/exclusion rules, metrics, and rulebook update workflow.
- Optional AI admission detection for Stage 2.5 when regex patterns find no admission.
- Document Stage 3 hardening flow, versions, metrics, and troubleshooting guidance.
### Removed
- Remove deprecated shims and aliases in audit, letter rendering, goodwill letters, instructions, and report analysis modules.
- Remove unused `logic.copy_documents` module.
