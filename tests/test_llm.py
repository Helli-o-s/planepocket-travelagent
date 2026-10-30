"""call_llm resilience: quota fallback, bounded retries. No network."""
import pytest
from langchain_core.messages import AIMessage

from travel import graph


class Fake:
    def __init__(self, errors: list[Exception]):
        self.errors, self.calls = errors, 0

    def invoke(self, _):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return AIMessage(content="ok")


@pytest.fixture
def chain(monkeypatch):
    fakes = {"a": Fake([Exception("429 RESOURCE_EXHAUSTED")]), "b": Fake([])}
    monkeypatch.setenv("GEMINI_MODELS", "a,b")
    monkeypatch.setattr(graph, "_active", 0)
    monkeypatch.setattr(graph, "get_llm", lambda name: fakes[name])
    monkeypatch.setattr(graph.time, "sleep", lambda s: None)
    events = []
    monkeypatch.setattr(graph, "SINKS", [lambda k, d: events.append(k)])
    return fakes, events


def test_quota_error_falls_back_silently_and_sticks(chain):
    fakes, events = chain
    assert graph.call_llm([]).content == "ok"
    assert graph.call_llm([]).content == "ok"
    assert (fakes["a"].calls, fakes["b"].calls) == (1, 2)    # exhausted model not retried
    assert events == ["fallback"]                            # logged to trace, no retry wait


def test_overloaded_model_also_falls_back(chain):
    fakes, events = chain
    fakes["a"].errors = [Exception("503 UNAVAILABLE. This model is currently experiencing high demand")]
    assert graph.call_llm([]).content == "ok" and events == ["fallback"]


def test_unavailable_model_is_skipped(chain):
    fakes, events = chain
    fakes["a"].errors = [Exception("404 NOT_FOUND. This model is no longer available to new users")]
    assert graph.call_llm([]).content == "ok" and events == ["fallback"]


def test_network_error_retries_same_model(chain):
    # Seen live: "RemoteProtocolError: Server disconnected" aborted a run mid-replan.
    fakes, events = chain
    fakes["a"].errors = [Exception("RemoteProtocolError: Server disconnected without sending a response.")]
    assert graph.call_llm([]).content == "ok"
    assert fakes["b"].calls == 0 and events == ["retry"]


def test_persistent_network_error_fails_cleanly(chain):
    fakes, events = chain
    fakes["a"].errors = [Exception("ConnectError: getaddrinfo failed")] * 5
    with pytest.raises(graph.LLMUnavailable):
        graph.call_llm([])
    assert fakes["a"].calls == graph.LLM_RETRIES and fakes["b"].calls == 0


def test_last_model_exhausted_retries_then_fails_cleanly(chain):
    fakes, events = chain
    fakes["b"].errors = [Exception("429")] * 5
    with pytest.raises(graph.LLMUnavailable):
        graph.call_llm([])
    assert fakes["b"].calls == graph.LLM_RETRIES
    assert events.count("retry") == graph.LLM_RETRIES - 1
