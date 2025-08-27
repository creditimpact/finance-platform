import pytest

from backend.core.ai import service
from backend.core.case_store import telemetry


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})]


class RecordingClient:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def chat_completion(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.fixture(autouse=True)
def clear_telemetry():
    telemetry.set_emitter(None)
    yield
    telemetry.set_emitter(None)


def test_run_llm_prompt_happy_path(monkeypatch):
    events: list[tuple[str, dict]] = []
    telemetry.set_emitter(lambda e, f: events.append((e, f)))

    client = RecordingClient(FakeResponse("{\"ok\":true}"))
    monkeypatch.setattr(service, "get_ai_client", lambda: client)

    out = service.run_llm_prompt("sys", "user", temperature=0, timeout_s=2)
    assert isinstance(out, str) and out

    assert ("ai_llm_call",) == tuple(e for e, _ in events)


def test_run_llm_prompt_timeout(monkeypatch):
    events: list[tuple[str, dict]] = []
    telemetry.set_emitter(lambda e, f: events.append((e, f)))

    client = RecordingClient(TimeoutError())
    monkeypatch.setattr(service, "get_ai_client", lambda: client)

    with pytest.raises(TimeoutError):
        service.run_llm_prompt("sys", "user", temperature=0, timeout_s=2)

    assert events and events[0][0] == "ai_llm_call_error"
    assert events[0][1]["error"] == "TimeoutError"


def test_run_llm_prompt_http_error(monkeypatch):
    class HTTPError(Exception):
        pass

    events: list[tuple[str, dict]] = []
    telemetry.set_emitter(lambda e, f: events.append((e, f)))

    client = RecordingClient(HTTPError())
    monkeypatch.setattr(service, "get_ai_client", lambda: client)

    with pytest.raises(HTTPError):
        service.run_llm_prompt("sys", "user", temperature=0, timeout_s=2)

    assert events and events[0][0] == "ai_llm_call_error"
    assert events[0][1]["error"] == "HTTPError"


def test_run_llm_prompt_respects_kwargs(monkeypatch):
    client = RecordingClient(FakeResponse("{}"))
    monkeypatch.setattr(service, "get_ai_client", lambda: client)

    service.run_llm_prompt(
        "sys",
        "user",
        temperature=0.5,
        timeout_s=5,
        model="gpt-test",
        max_tokens=123,
    )

    assert client.kwargs["model"] == "gpt-test"
    assert client.kwargs["temperature"] == 0.5
    assert client.kwargs["timeout"] == 5
    assert client.kwargs["max_tokens"] == 123

