import json

from backend.core.logic.report_analysis.candidate_logger import StageATraceLogger


def test_stageA_trace_lines(tmp_path):
    logger = StageATraceLogger("sess1", base_folder=tmp_path)
    logger.append(
        {
            "normalized_name": "foo",
            "account_id": "1234",
            "decision_source": "ai",
            "primary_issue": "collection",
            "confidence": 0.9,
            "tier": 1,
            "reasons": ["r1"],
            "ai_latency_ms": 10,
            "ai_tokens_in": 5,
            "ai_tokens_out": 2,
            "ai_error": None,
        }
    )
    logger.append(
        {
            "normalized_name": "bar",
            "account_id": "abcd",
            "decision_source": "rules",
            "primary_issue": "unknown",
            "confidence": 0.0,
            "tier": 0,
            "reasons": [],
            "ai_latency_ms": 0,
            "ai_tokens_in": 0,
            "ai_tokens_out": 0,
            "ai_error": None,
        }
    )
    path = tmp_path / "sess1" / "stageA_trace.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert "ts" in row
        assert "decision_source" in row
        assert "ai_latency_ms" in row
        assert "ai_tokens_in" in row
        assert "ai_tokens_out" in row
        assert "ai_error" in row
