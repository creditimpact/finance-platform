import json
import logging

from backend.core.logic.report_analysis import config as merge_config


def test_invalid_env_values_fall_back_to_defaults(monkeypatch, caplog):
    monkeypatch.setenv("AI_PACK_MAX_LINES_PER_SIDE", "3")
    monkeypatch.setenv("AI_MODEL", " ")
    monkeypatch.setenv("AI_REQUEST_TIMEOUT", "zero")
    monkeypatch.setenv("MERGE_V2_ONLY", "maybe")

    with caplog.at_level(
        logging.WARNING, logger="backend.core.logic.report_analysis.config"
    ):
        assert merge_config.get_ai_pack_max_lines_per_side() == 20
        assert merge_config.get_ai_model() == "gpt-4o-mini"
        assert merge_config.get_ai_request_timeout() == 30
        assert merge_config.get_merge_v2_only() is True

        # Repeat to confirm warnings are only emitted once per key
        assert merge_config.get_ai_pack_max_lines_per_side() == 20
        assert merge_config.get_ai_model() == "gpt-4o-mini"
        assert merge_config.get_ai_request_timeout() == 30
        assert merge_config.get_merge_v2_only() is True

    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("MERGE_V2_CONFIG_DEFAULT ")
    ]

    assert len(warning_messages) == 4

    payloads = [json.loads(message.split(" ", 1)[1]) for message in warning_messages]
    keys = {payload["key"] for payload in payloads}
    assert keys == {
        "AI_PACK_MAX_LINES_PER_SIDE",
        "AI_MODEL",
        "AI_REQUEST_TIMEOUT",
        "MERGE_V2_ONLY",
    }
