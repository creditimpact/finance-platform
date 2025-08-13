# Changelog

## Unreleased
### Added
- Analysis prompt now instructs models to respond with valid JSON only, using `null` or empty arrays when data is unknown and avoiding invented fields. Bump `ANALYSIS_PROMPT_VERSION` to 2.
### Removed
- Remove deprecated shims and aliases in audit, letter rendering, goodwill letters, instructions, and report analysis modules.
- Remove unused `logic.copy_documents` module.
