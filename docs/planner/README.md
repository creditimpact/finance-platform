# Planner metrics

The planner emits the following metrics for monitoring:

- `planner.cycle_progress{cycle,step}` – counts step transitions per cycle. Cardinality: cycle and step each <10.
- `planner.time_to_next_step_ms` – gauge of milliseconds until the next eligible planner action. Cardinality: 1.
- `planner.resolution_rate` – fraction of accounts resolved in a run. Cardinality: 1.
- `planner.avg_cycles_per_resolution` – average cycles spent per resolved account. Cardinality: 1.
- `planner.sla_violations_total` – total number of SLA breaches when sends occur after the deadline. Cardinality: 1.
- `planner.error_count` – total planner exceptions. Cardinality: 1.

These metrics power dashboards used to track planner throughput and reliability.
