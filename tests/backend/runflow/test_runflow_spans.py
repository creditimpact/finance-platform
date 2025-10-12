from __future__ import annotations

import importlib


def _reload_modules():
    import backend.core.runflow as runflow
    import backend.core.runflow_spans as runflow_spans

    importlib.reload(runflow)
    importlib.reload(runflow_spans)
    return runflow, runflow_spans


def test_runflow_spans_start_and_end_emit_parent_relationship(monkeypatch):
    calls: list[tuple[tuple, dict]] = []

    def _fake_runflow_step(*args, **kwargs):
        calls.append((args, kwargs))

    runflow, runflow_spans = _reload_modules()
    monkeypatch.setattr(runflow_spans, "runflow_step", _fake_runflow_step)
    monkeypatch.setattr(runflow_spans, "_ACTIVE_SPANS", {})

    span_id = runflow_spans.start_span("SID", "merge", "outer", ctx={"foo": 1})
    assert span_id

    assert len(calls) == 1
    _, start_kwargs = calls[0]
    assert start_kwargs["status"] == "start"
    assert start_kwargs["metrics"] == {"foo": 1}
    assert start_kwargs["span_id"] == span_id
    assert start_kwargs["parent_span_id"] is None

    runflow_spans.end_span(span_id, status="error", reason="boom")

    assert len(calls) == 2
    _, end_kwargs = calls[1]
    assert end_kwargs["status"] == "error"
    assert end_kwargs["reason"] == "boom"
    assert end_kwargs["span_id"] == span_id
    assert end_kwargs["parent_span_id"] is None
    assert runflow_spans._ACTIVE_SPANS == {}


def test_runflow_spans_nested_parent_ids_propagate(monkeypatch):
    calls: list[tuple[tuple, dict]] = []

    def _fake_runflow_step(*args, **kwargs):
        calls.append((args, kwargs))

    runflow, runflow_spans = _reload_modules()
    monkeypatch.setattr(runflow_spans, "runflow_step", _fake_runflow_step)
    monkeypatch.setattr(runflow_spans, "_ACTIVE_SPANS", {})

    parent_id = runflow_spans.start_span("SID", "merge", "parent")
    child_id = runflow_spans.start_span(
        "SID",
        "merge",
        "child",
        parent_span_id=parent_id,
    )

    runflow_spans.end_span(child_id, status="success", metrics={"count": 1})
    runflow_spans.end_span(parent_id, status="success")

    assert runflow_spans._ACTIVE_SPANS == {}

    child_end = [kwargs for _, kwargs in calls if kwargs.get("span_id") == child_id][-1]
    assert child_end["parent_span_id"] == parent_id
    assert child_end["metrics"] == {"count": 1}

    parent_entries = [kwargs for _, kwargs in calls if kwargs.get("span_id") == parent_id]
    assert len(parent_entries) == 2
    assert parent_entries[0]["status"] == "start"
    assert parent_entries[1]["status"] == "success"


def test_runflow_span_step_uses_parent_without_span_id(monkeypatch):
    calls: list[tuple[tuple, dict]] = []

    def _fake_runflow_step(*args, **kwargs):
        calls.append((args, kwargs))

    runflow, runflow_spans = _reload_modules()
    monkeypatch.setattr(runflow_spans, "runflow_step", _fake_runflow_step)

    runflow_spans.span_step(
        "SID",
        "merge",
        "summary",
        parent_span_id="parent123",
        status="success",
        metrics={"pairs": 3},
        out={"path": "merge/summary.json"},
    )

    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["parent_span_id"] == "parent123"
    assert kwargs["span_id"] is None
    assert kwargs["metrics"] == {"pairs": 3}
    assert kwargs["out"] == {"path": "merge/summary.json"}

