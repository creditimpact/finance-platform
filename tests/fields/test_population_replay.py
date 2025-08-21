from backend.core.letters.field_population import (
    apply_field_fillers,
    clear_filler_cache,
)


def test_population_replay(monkeypatch):
    calls = []

    def fake_populate_name(ctx, profile, corrections):
        calls.append(1)
        if ctx.get("name"):
            return
        if profile and profile.get("name"):
            ctx["name"] = profile["name"]

    monkeypatch.setattr(
        "backend.core.letters.field_population.populate_name", fake_populate_name
    )
    clear_filler_cache()

    profile = {"name": "Alice"}
    outcome = {"timestamp": "2024-01-01T00:00:00+00:00"}

    ctx1 = {
        "account_id": "acct-1",
        "cra_outcome": outcome,
        "now": "2024-01-10T00:00:00+00:00",
    }
    apply_field_fillers(ctx1, profile=profile)
    assert ctx1["name"] == "Alice"
    assert ctx1["days_since_cra_result"] == 9
    assert len(calls) == 1

    ctx2 = {
        "account_id": "acct-1",
        "cra_outcome": outcome,
        "now": "2024-01-10T00:00:00+00:00",
    }
    apply_field_fillers(ctx2, profile=profile)
    assert ctx2["name"] == "Alice"
    assert ctx2["days_since_cra_result"] == 9
    assert len(calls) == 1

