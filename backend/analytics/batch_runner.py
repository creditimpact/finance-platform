"""Batch processing utilities for generating analytics reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

FLAGS = [
    "LETTERS_ROUTER_PHASED",
    "ENFORCE_TEMPLATE_VALIDATION",
    "SAFE_CLIENT_SENTENCE_ENABLED",
]


def _percentile(values: List[float], pct: float) -> float:
    """Return the ``pct`` percentile of ``values``."""

    if not values:
        return 0.0
    vals = sorted(values)
    k = max(0, int(math.ceil(pct / 100 * len(vals))) - 1)
    return vals[k]


def _flatten_heatmap(
    data: Mapping[str, Mapping[str, int]]
) -> Mapping[str, Mapping[str, int]]:
    """Return a nested dict with integer counts."""

    result: Dict[str, Dict[str, int]] = {}
    for tag, fields in data.items():
        bucket = result.setdefault(tag, {})
        for field, count in fields.items():
            bucket[field] = bucket.get(field, 0) + int(count)
    return result


def process_samples(samples: Iterable[Mapping[str, object]]) -> Dict[str, object]:
    """Process ``samples`` and return an aggregated metrics report."""

    candidate_counts: Dict[str, int] = {}
    finalized_counts: Dict[str, int] = {}
    missing_stage1: Dict[str, Dict[str, int]] = {}
    missing_stage2: Dict[str, Dict[str, int]] = {}
    validation_breakdown: Dict[str, Dict[str, int]] = {}
    sanitizer_applied: Dict[str, int] = {}
    policy_override: Dict[str, Dict[str, int]] = {}
    total_letters: Dict[str, int] = {}
    render_times: Dict[str, List[float]] = {}
    ai_costs: Dict[str, List[float]] = {}

    for sample in samples:
        tag = str(sample.get("action_tag", ""))
        template = str(sample.get("template", ""))

        candidate_counts[tag] = candidate_counts.get(tag, 0) + 1

        for field in sample.get("candidate_missing_fields", []) or []:
            bucket = missing_stage1.setdefault(tag, {})
            bucket[field] = bucket.get(field, 0) + 1

        if sample.get("validation_failed_fields"):
            for field in sample.get("final_missing_fields", []) or []:
                bucket = missing_stage2.setdefault(tag, {})
                bucket[field] = bucket.get(field, 0) + 1
            vf = validation_breakdown.setdefault(template, {})
            for field in sample.get("validation_failed_fields", []) or []:
                vf[field] = vf.get(field, 0) + 1
        else:
            finalized_counts[tag] = finalized_counts.get(tag, 0) + 1
            for field in sample.get("final_missing_fields", []) or []:
                bucket = missing_stage2.setdefault(tag, {})
                bucket[field] = bucket.get(field, 0) + 1

        total_letters[template] = total_letters.get(template, 0) + 1

        if sample.get("sanitizer_overrides"):
            sanitizer_applied[template] = sanitizer_applied.get(template, 0) + 1
            po_bucket = policy_override.setdefault(template, {})
            for reason in sample.get("sanitizer_overrides", []) or []:
                sanitized = str(reason).replace(" ", "_")
                po_bucket[sanitized] = po_bucket.get(sanitized, 0) + 1

        render_times.setdefault(template, []).append(float(sample.get("render_ms", 0)))
        ai_costs.setdefault(template, []).append(float(sample.get("ai_cost", 0)))

    pass_rate: Dict[str, float] = {}
    for tag, count in candidate_counts.items():
        finalized = finalized_counts.get(tag, 0)
        pass_rate[tag] = finalized / count if count else 0.0

    sanitize_rate: Dict[str, float] = {}
    for template, total in total_letters.items():
        applied = sanitizer_applied.get(template, 0)
        sanitize_rate[template] = applied / total if total else 0.0

    render_avg = {tpl: sum(vals) / len(vals) for tpl, vals in render_times.items()}
    render_p95 = {tpl: _percentile(vals, 95) for tpl, vals in render_times.items()}
    ai_avg = {tpl: sum(vals) / len(vals) for tpl, vals in ai_costs.items()}
    ai_p95 = {tpl: _percentile(vals, 95) for tpl, vals in ai_costs.items()}

    report: Dict[str, object] = {
        "finalization_pass_rate": pass_rate,
        "missing_fields": {
            "after_normalization": _flatten_heatmap(missing_stage1),
            "after_strategy": _flatten_heatmap(missing_stage2),
        },
        "validation_failed": _flatten_heatmap(validation_breakdown),
        "sanitizer": {
            "applied_rate": sanitize_rate,
            "policy_override_reason": _flatten_heatmap(policy_override),
        },
        "render_latency_ms": {"avg": render_avg, "p95": render_p95},
        "ai_cost": {"avg": ai_avg, "p95": ai_p95},
    }
    return report


def _write_csv(report: Mapping[str, object], path: Path) -> None:
    rows: List[List[object]] = []
    for tag, rate in report["finalization_pass_rate"].items():
        rows.append(["finalization_pass_rate", tag, "", "", rate])
    for stage_key, stage_data in report["missing_fields"].items():
        stage_label = f"missing_fields_{stage_key}"
        for tag, fields in stage_data.items():
            for field, count in fields.items():
                rows.append([stage_label, tag, field, "", count])
    for template, fields in report["validation_failed"].items():
        for field, count in fields.items():
            rows.append(["validation_failed", template, field, "", count])
    for template, rate in report["sanitizer"]["applied_rate"].items():
        rows.append(["sanitizer_applied_rate", template, "", "", rate])
    for template, reasons in report["sanitizer"]["policy_override_reason"].items():
        for reason, count in reasons.items():
            rows.append(["policy_override_reason", template, reason, "", count])
    for template, val in report["render_latency_ms"]["avg"].items():
        rows.append(["render_latency_avg_ms", template, "", "", val])
    for template, val in report["render_latency_ms"]["p95"].items():
        rows.append(["render_latency_p95_ms", template, "", "", val])
    for template, val in report["ai_cost"]["avg"].items():
        rows.append(["ai_cost_avg", template, "", "", val])
    for template, val in report["ai_cost"]["p95"].items():
        rows.append(["ai_cost_p95", template, "", "", val])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "category", "item", "extra", "value"])
        writer.writerows(rows)


def run_staging_batch(
    samples_path: str | Path,
    limit: int | None = None,
    output_dir: str | Path | None = None,
) -> Dict[str, object]:
    """Process sample inputs and write a batch report."""

    for flag in FLAGS:
        os.environ[flag] = "true"

    with open(samples_path, "r", encoding="utf-8") as f:
        samples: List[Mapping[str, object]] = json.load(f)

    if limit is not None:
        samples = samples[:limit]

    report = process_samples(samples)

    out_dir = Path(output_dir or Path("backend/analytics/batch_reports"))
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"{timestamp}.json"
    csv_path = out_dir / f"{timestamp}.csv"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(report, jf, indent=2)
    _write_csv(report, csv_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staging batch and emit report")
    parser.add_argument("samples", help="Path to sample JSON file")
    parser.add_argument("-n", "--num", type=int, help="Number of samples to process")
    args = parser.parse_args()
    run_staging_batch(args.samples, limit=args.num)


if __name__ == "__main__":  # pragma: no cover - CLI only
    main()


__all__ = ["run_staging_batch", "process_samples"]
