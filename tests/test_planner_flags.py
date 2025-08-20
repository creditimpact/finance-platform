import importlib

import backend.api.config as config
import tactical


def reload_orchestrators():
    import backend.core.orchestrators as orch

    importlib.reload(config)
    importlib.reload(orch)
    return orch


def test_planner_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_PLANNER", "0")
    monkeypatch.setenv("PLANNER_CANARY_PERCENT", "100")
    orch = reload_orchestrators()
    calls = []

    def fake_plan(session, tags):
        calls.append("planner")
        return tags

    def fake_generate(session, tags):
        calls.append(("tactical", tags))

    monkeypatch.setattr(orch, "plan_next_step", fake_plan)
    monkeypatch.setattr(tactical, "generate_letters", fake_generate)

    orch.plan_and_generate_letters({"strategy": {}}, ["dispute"])

    assert calls == [("tactical", ["dispute"])]

    monkeypatch.delenv("ENABLE_PLANNER", raising=False)
    monkeypatch.delenv("PLANNER_CANARY_PERCENT", raising=False)
    reload_orchestrators()


def test_planner_canary(monkeypatch):
    monkeypatch.setenv("ENABLE_PLANNER", "1")
    monkeypatch.setenv("PLANNER_CANARY_PERCENT", "50")
    orch = reload_orchestrators()
    calls = []

    def fake_plan(session, tags):
        calls.append("planner")
        return ["allowed"]

    def fake_generate(session, tags):
        calls.append(("tactical", tags))

    monkeypatch.setattr(orch, "plan_next_step", fake_plan)
    monkeypatch.setattr(tactical, "generate_letters", fake_generate)

    monkeypatch.setattr(orch.random, "random", lambda: 0.2)
    orch.plan_and_generate_letters({"strategy": {}}, ["dispute"])
    assert calls == ["planner", ("tactical", ["allowed"])]

    calls.clear()
    monkeypatch.setattr(orch.random, "random", lambda: 0.8)
    orch.plan_and_generate_letters({"strategy": {}}, ["dispute"])
    assert calls == [("tactical", ["dispute"])]

    monkeypatch.delenv("ENABLE_PLANNER", raising=False)
    monkeypatch.delenv("PLANNER_CANARY_PERCENT", raising=False)
    reload_orchestrators()
