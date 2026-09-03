import time
from types import SimpleNamespace

import pytest

from src.config import CFG
from src.vlm.client import GeminiVLMClient, VLMUnavailable


class ApiFailure(Exception):
    def __init__(self, code: int):
        super().__init__(f"HTTP {code}")
        self.code = code


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.timeouts = []

    on_call = None
    #: 시도마다 넘어온 요청 단위 timeout(초). 남은 예산에 맞춰 줄어드는지 본다.
    timeouts: list

    def generate_content(self, **kwargs):
        config = kwargs.get("config")
        http_options = getattr(config, "http_options", None)
        if http_options is not None and http_options.timeout is not None:
            self.timeouts.append(http_options.timeout / 1000)
        if self.on_call is not None:
            self.on_call()
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)


class FakeClock:
    """느린 503을 결정적으로 재현한다. 실제로 503은 4~35초씩 걸려서 돌아온다."""

    def __init__(self, per_call_seconds=0.0):
        self.now = 1000.0
        self.per_call_seconds = per_call_seconds

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds

    def on_request(self):
        self.advance(self.per_call_seconds)


def make_client(monkeypatch, outcomes, sleeps, clock=None):
    models = FakeModels(outcomes)
    if clock is not None:
        models.on_call = clock.on_request
    client = GeminiVLMClient(
        model="test-gemini",
        client=SimpleNamespace(models=models),
        sleep=(lambda seconds: (sleeps.append(seconds),
                                clock.advance(seconds) if clock else None))
              if clock else sleeps.append,
        jitter=lambda low, high: (low + high) / 2,
        clock=clock or time.monotonic,
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

    # 소진된 상류 혼잡은 VLMUnavailable로 올라간다 — api가 503으로 번역할 수 있어야 하고,
    # 파이프라인 버그와 같은 통로(500 + UNHANDLED_ERROR 알림)로 새면 안 된다.
    with pytest.raises(VLMUnavailable) as raised:
        client.analyze("ignored", 100, 100)

    assert raised.value.status == 429
    assert raised.value.attempts == 3
    assert raised.value.__cause__.code == 429
    assert models.calls == 3
    assert len(sleeps) == 2


def test_wraps_exhausted_5xx_as_unavailable(monkeypatch):
    """503만이 아니라 5xx 전체를 상류 혼잡으로 본다.

    Gemini는 500 INTERNAL도 돌려준다. 전에는 503만 재시도 대상이라 500은 첫 실패에서
    바로 처리되지 않은 예외가 됐다.
    """
    monkeypatch.setattr(CFG, "gemini_max_attempts", 2)
    sleeps = []
    client, models = make_client(monkeypatch, [ApiFailure(500), ApiFailure(500)], sleeps)

    with pytest.raises(VLMUnavailable) as raised:
        client.analyze("ignored", 100, 100)

    assert raised.value.status == 500
    assert models.calls == 2


@pytest.mark.parametrize("failure", [ApiFailure(400), ValueError("bad json")])
def test_our_own_failures_are_not_disguised_as_upstream(monkeypatch, failure):
    """4xx와 파싱 오류는 우리 잘못이다. 그대로 올려 보내 500 + 알림을 받는다.

    상태 코드가 없다는 것만으로 "상류 혼잡"이라고 분류하면 이런 버그가 재시도 안내
    뒤에 숨는다.
    """
    monkeypatch.setattr(CFG, "gemini_max_attempts", 3)
    sleeps = []
    client, models = make_client(monkeypatch, [failure], sleeps)

    with pytest.raises(type(failure)):
        client.analyze("ignored", 100, 100)

    assert models.calls == 1
    assert sleeps == []


def test_timeout_is_not_retried_but_is_reported_as_upstream(monkeypatch):
    """timeout은 재시도하지 않는다(끊긴 호출도 Gemini 쪽에선 계속 생성 중일 수 있다).
    다만 원인은 상류 지연이므로 503으로 번역될 수 있게 감싼다."""
    monkeypatch.setattr(CFG, "gemini_max_attempts", 3)
    sleeps = []
    client, models = make_client(monkeypatch, [TimeoutError("timed out")], sleeps)

    with pytest.raises(VLMUnavailable) as raised:
        client.analyze("ignored", 100, 100)

    assert raised.value.status is None
    assert models.calls == 1
    assert sleeps == []


def test_retry_fits_in_the_remaining_budget_instead_of_being_skipped(monkeypatch):
    """느린 503 뒤에도 **남은 예산 안에서** 한 번 더 시도한다.

    2026-08-21 프로덕션: 503이 4~35초 걸려 돌아오는데 "남은 예산 ≥ 전체 timeout(45초)"
    일 때만 재시도하는 규칙이라, 예산 75초에서 두 번째 시도조차 못 하고 실패했다
    (실패 4건 중 2건이 budget_exhausted, 그중 1건은 재시도 0회). 시도의 timeout을
    남은 예산으로 줄여 잡으면 짧게 한 번 더 해 볼 수 있다.
    """
    monkeypatch.setattr(CFG, "gemini_max_attempts", 3)
    monkeypatch.setattr(CFG, "gemini_request_timeout_ms", 45_000)
    monkeypatch.setattr(CFG, "gemini_total_budget_seconds", 75)
    monkeypatch.setattr(CFG, "gemini_min_attempt_seconds", 10)
    clock = FakeClock(per_call_seconds=35)      # 503이 35초 만에 돌아오는 상황
    sleeps = []
    client, models = make_client(
        monkeypatch, [ApiFailure(503), valid_response()], sleeps, clock=clock
    )

    result = client.analyze("ignored", 100, 100)

    assert result.num_people == 1
    assert models.calls == 2, "예전 규칙이면 여기서 재시도 없이 실패했다"
    # 1차는 전체 timeout, 2차는 남은 예산(75-35-대기)으로 줄어든다.
    assert models.timeouts[0] == 45
    assert 30 < models.timeouts[1] < 40


def test_stops_when_remaining_budget_is_too_small_to_matter(monkeypatch):
    """예산이 의미 있는 시도 하나도 못 담으면 멈춘다.

    끝까지 시도하면 BFF의 분석 상한(120초)을 먹어 사용자는 원인을 알 수 없는
    ANALYSIS_TIMEOUT을 받는다.
    """
    monkeypatch.setattr(CFG, "gemini_max_attempts", 3)
    monkeypatch.setattr(CFG, "gemini_request_timeout_ms", 45_000)
    monkeypatch.setattr(CFG, "gemini_total_budget_seconds", 50)
    monkeypatch.setattr(CFG, "gemini_min_attempt_seconds", 10)
    clock = FakeClock(per_call_seconds=45)      # 첫 시도가 예산을 거의 다 쓴다
    sleeps = []
    client, models = make_client(
        monkeypatch, [ApiFailure(503), ApiFailure(503), ApiFailure(503)], sleeps, clock=clock
    )

    with pytest.raises(VLMUnavailable) as raised:
        client.analyze("ignored", 100, 100)

    assert models.calls == 1, "남은 5초로는 시도할 가치가 없다"
    assert raised.value.status == 503
    assert sleeps == []


def test_retries_within_budget(monkeypatch):
    """예산이 넉넉하면 기존대로 429/503을 재시도한다(예산 가드가 정상 경로를 막지 않는다)."""
    monkeypatch.setattr(CFG, "gemini_max_attempts", 3)
    monkeypatch.setattr(CFG, "gemini_request_timeout_ms", 45_000)
    monkeypatch.setattr(CFG, "gemini_total_budget_seconds", 120)
    sleeps = []
    client, models = make_client(
        monkeypatch, [ApiFailure(503), valid_response()], sleeps
    )

    result = client.analyze("ignored", 100, 100)

    assert result.num_people == 1
    assert models.calls == 2
    assert len(sleeps) == 1
