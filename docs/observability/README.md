# Observability

This project emits structured logs and metrics for traceability.

## Costs
High-volume logs and extended metrics retention incur storage expenses. To control cost:
- Per-family tri-merge logs are sampled at 10% (`if random() < 0.1`).
- `*_debug` metrics are retained for 7 days; production metrics remain for 30 days.
