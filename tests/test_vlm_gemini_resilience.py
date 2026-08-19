from types import SimpleNamespace

import pytest

from src.config import CFG
from src.vlm.client import GeminiVLMClient


class ApiFailure(Exception):
    def __init__(self, code: int):
        super().__init__(f"HTTP {code}")
        self.code = code


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate_content(self, **_kwargs):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)


def make_client(monkeypatch, outcomes, sleeps):
    models = FakeModels(outcomes)
    client = GeminiVLMClient(
        model="test-gemini",
        client=SimpleNamespace(models=models),
        sleep=sleeps.append,
        jitter=lambda low, high: (low + high) / 2,
    )
    monkeypatch.setattr(client, "_to_part", lambda _image: "image-part")
    return client, models


def valid_response():
    return '{"num_people": 1, "shot": "full_half", "action": "standing", "view": "front", "relationship": "solo", "approx_boxes": []}'


def test_configures_google_sdk_request_timeout(monkeypatch):
    from google import genai

    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(models=FakeModels([]))

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(CFG, "gemini_request_timeout_ms", 12_345)
    monkeypatch.setattr(genai, "Client", fake_client)

    GeminiVLMClient()

    assert captured["api_key"] == "test-key"
    assert captured["http_options"].timeout == 12_345


def test_retries_503_then_succeeds(monkeypatch):
    monkeypatch.setattr(CFG, "gemini_max_attempts", 3)
    monkeypatch.setattr(CFG, "gemini_retry_base_seconds", 0.5)
    monkeypatch.setattr(CFG, "gemini_retry_max_seconds", 2.0)
    sleeps = []
    client, models = make_client(
        monkeypatch,
        [ApiFailure(503), ApiFailure(503), valid_response()],
        sleeps,
    )

    result = client.analyze("ignored", 100, 100)

    assert result.num_people == 1
    assert models.calls == 3
    assert sleeps == [0.5, 1.0]


def test_stops_after_bounded_429_attempts(monkeypatch):
    monkeypatch.setattr(CFG, "gemini_max_attempts", 3)
    sleeps = []
    client, models = make_client(
        monkeypatch,
        [ApiFailure(429), ApiFailure(429), ApiFailure(429)],
        sleeps,
    )

    with pytest.raises(ApiFailure) as raised:
        client.analyze("ignored", 100, 100)

    assert raised.value.code == 429
    assert models.calls == 3
    assert len(sleeps) == 2


@pytest.mark.parametrize("failure", [ApiFailure(400), TimeoutError("timed out")])
def test_does_not_retry_non_transient_or_timeout(monkeypatch, failure):
    monkeypatch.setattr(CFG, "gemini_max_attempts", 3)
    sleeps = []
    client, models = make_client(monkeypatch, [failure], sleeps)

    with pytest.raises(type(failure)):
        client.analyze("ignored", 100, 100)

    assert models.calls == 1
    assert sleeps == []
