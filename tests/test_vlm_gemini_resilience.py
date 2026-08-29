import logging
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
        self.models = []
        self.thinking = []

    on_call = None
    #: 시도마다 넘어온 요청 단위 timeout(초). 남은 예산에 맞춰 줄어드는지 본다.
    timeouts: list
    #: 시도마다 태운 모델 이름. 폴백 체인이 실제로 다른 풀을 두드리는지 본다.
    models: list
    #: 시도마다 나간 thinking_config(없으면 None).
    thinking: list

    def generate_content(self, **kwargs):
        config = kwargs.get("config")
        self.models.append(kwargs.get("model"))
        self.thinking.append(getattr(config, "thinking_config", None))
        http_options = getattr(config, "http_options", None)
        if http_options is not None and http_options.timeout is not None:
            self.timeouts.append(http_options.timeout / 1000)
        if self.on_call is not None:
            self.on_call()
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, str):
            return SimpleNamespace(text=outcome)
        return outcome            # 사용량 메타데이터까지 실은 완성 응답


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


def make_client(monkeypatch, outcomes, sleeps, clock=None, fallbacks=""):
    # 기본은 폴백 없음. 기존 테스트가 "모델 하나일 때의 재시도"를 계속 문서화하도록
    # 두고, 체인 동작은 fallbacks를 명시한 테스트에서만 본다.
    monkeypatch.setattr(CFG, "gemini_fallback_models", fallbacks)
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


def test_logs_token_usage_for_cost_tracking(monkeypatch, caplog):
    """유료 티어에서 비용이 어디서 났는지 사후 추적하려면 호출당 사용량이 남아야 한다."""
    response = SimpleNamespace(
        text=valid_response(),
        usage_metadata=SimpleNamespace(
            prompt_token_count=812,
            candidates_token_count=57,
            thoughts_token_count=120,
            total_token_count=989,
        ),
    )
    client, _ = make_client(monkeypatch, [response], [])

    with caplog.at_level(logging.INFO, logger="standin"):
        client.analyze("ignored", 100, 100)

    record = next(r for r in caplog.records
                  if getattr(r, "event_type", None) == "gemini_request")
    assert record.fields["promptTokens"] == 812
    assert record.fields["outputTokens"] == 57
    assert record.fields["thoughtTokens"] == 120
    assert record.fields["totalTokens"] == 989
    # 모델·SDK가 안 주는 값은 0이 아니라 필드 자체를 만들지 않는다(0과 미보고 구분).
    assert "cachedTokens" not in record.fields


def test_analyze_succeeds_when_sdk_reports_no_usage(monkeypatch):
    """사용량 로깅이 분석 자체를 실패시키면 안 된다(구버전 SDK·응답 형태 변화)."""
    client, _ = make_client(monkeypatch, [SimpleNamespace(text=valid_response())], [])

    assert client.analyze("ignored", 100, 100).num_people == 1


# ── 폴백 모델 체인 ──────────────────────────────────────────────────────────
# 503은 **모델별 용량 풀**의 문제다. 같은 모델 재시도는 실패가 상관돼 있어 3시도가
# 통째로 같은 스파이크에 걸린다(2026-08-28 프로덕션: 3시도 전부 503, 18~37초 만에 소진).


def test_falls_back_to_another_model_when_one_pool_is_overloaded(monkeypatch):
    """1차 모델이 503으로 소진되면 남은 예산으로 **다른 모델**을 태운다."""
    monkeypatch.setattr(CFG, "gemini_max_attempts", 2)
    sleeps = []
    client, models = make_client(
        monkeypatch,
        [ApiFailure(503), ApiFailure(503), valid_response()],
        sleeps,
        fallbacks="backup-gemini",
    )

    result = client.analyze("ignored", 100, 100)

    assert result.num_people == 1
    assert models.models == ["test-gemini", "test-gemini", "backup-gemini"], (
        "1차 소진 뒤에는 같은 풀을 또 두드리지 말고 다른 모델로 넘어가야 한다"
    )


def test_fallback_reports_every_model_tried(monkeypatch):
    """체인이 전부 붐볐으면 알림이 '어디까지 태웠는지'를 말해야 한다.

    1차만 붐볐다면 모델을 바꾸면 되고, 둘 다 붐볐다면 구글 전역 혼잡이라 기다리는 수밖에
    없다 — 운영이 이 둘을 구분하려면 모델 목록과 **합산** 시도 횟수가 필요하다.
    """
    monkeypatch.setattr(CFG, "gemini_max_attempts", 2)
    sleeps = []
    client, models = make_client(
        monkeypatch,
        [ApiFailure(503)] * 4,
        sleeps,
        fallbacks="backup-gemini",
    )

    with pytest.raises(VLMUnavailable) as raised:
        client.analyze("ignored", 100, 100)

    assert raised.value.models_tried == ("test-gemini", "backup-gemini")
    assert raised.value.attempts == 4, "시도 횟수는 체인 전체의 합이다"
    assert models.calls == 4


def test_duplicate_fallback_is_not_tried_twice(monkeypatch):
    """폴백에 1차와 같은 모델을 적어도 같은 풀을 두 번 두드리지 않는다."""
    monkeypatch.setattr(CFG, "gemini_max_attempts", 1)
    sleeps = []
    client, models = make_client(
        monkeypatch, [ApiFailure(503)] * 2, sleeps,
        fallbacks="test-gemini, backup-gemini",
    )

    with pytest.raises(VLMUnavailable):
        client.analyze("ignored", 100, 100)

    assert models.models == ["test-gemini", "backup-gemini"]


def test_fallback_is_skipped_when_the_budget_is_gone(monkeypatch):
    """예산이 없으면 폴백도 안 태운다.

    끝까지 태우면 BFF의 분석 상한(120초)을 넘겨 사용자는 원인을 알 수 없는
    ANALYSIS_TIMEOUT을 받는다. 폴백은 '1차가 안 쓰고 남긴 예산'으로만 도는 것이다.
    """
    monkeypatch.setattr(CFG, "gemini_max_attempts", 1)
    monkeypatch.setattr(CFG, "gemini_total_budget_seconds", 50)
    monkeypatch.setattr(CFG, "gemini_min_attempt_seconds", 10)
    clock = FakeClock(per_call_seconds=45)      # 첫 시도가 예산을 거의 다 쓴다
    sleeps = []
    client, models = make_client(
        monkeypatch, [ApiFailure(503), valid_response()], sleeps,
        clock=clock, fallbacks="backup-gemini",
    )

    with pytest.raises(VLMUnavailable) as raised:
        client.analyze("ignored", 100, 100)

    assert models.calls == 1, "남은 5초로는 폴백을 태울 가치가 없다"
    assert raised.value.models_tried == ("test-gemini",)


def test_our_own_failure_does_not_trigger_fallback(monkeypatch):
    """4xx·파싱 오류에 폴백을 태우면 같은 이유로 또 실패하고 비용만 두 배가 된다."""
    monkeypatch.setattr(CFG, "gemini_max_attempts", 2)
    sleeps = []
    client, models = make_client(
        monkeypatch, [ApiFailure(400), valid_response()], sleeps,
        fallbacks="backup-gemini",
    )

    with pytest.raises(ApiFailure):
        client.analyze("ignored", 100, 100)

    assert models.calls == 1
    assert models.models == ["test-gemini"]


def test_timeout_does_not_trigger_fallback(monkeypatch):
    """timeout에는 폴백도 걸지 않는다.

    끊긴 호출도 상류에서는 계속 생성 중일 수 있다. 재시도를 막은 이유(비용 중복)가
    폴백에도 그대로 적용된다 — 게다가 timeout은 이미 예산을 크게 먹은 상태다.
    """
    monkeypatch.setattr(CFG, "gemini_max_attempts", 2)
    sleeps = []
    client, models = make_client(
        monkeypatch, [TimeoutError("deadline"), valid_response()], sleeps,
        fallbacks="backup-gemini",
    )

    with pytest.raises(VLMUnavailable) as raised:
        client.analyze("ignored", 100, 100)

    assert models.calls == 1
    assert raised.value.status is None
    assert raised.value.models_tried == ("test-gemini",)


# ── 사고 토큰 ───────────────────────────────────────────────────────────────


def test_thinking_is_disabled_by_default(monkeypatch):
    """이 단계는 열거형 태깅이라 추론 여력이 필요 없다. 호출 시간과 과금만 늘어난다."""
    sleeps = []
    client, models = make_client(monkeypatch, [valid_response()], sleeps)

    client.analyze("ignored", 100, 100)

    assert models.thinking[0] is not None
    assert models.thinking[0].thinking_budget == 0


def test_thinking_config_is_omitted_when_turned_off(monkeypatch):
    """thinking_config 자체를 거부하는 구형 모델을 위해 필드를 안 보낼 수 있어야 한다.

    보내면 400이 오고, 400은 이 파일의 규칙상 '우리 잘못'으로 분류돼 알림이 된다.
    """
    monkeypatch.setattr(CFG, "gemini_thinking_budget", "none")
    sleeps = []
    client, models = make_client(monkeypatch, [valid_response()], sleeps)

    client.analyze("ignored", 100, 100)

    assert models.thinking[0] is None


def test_unreadable_thinking_budget_falls_back_to_model_default(monkeypatch):
    """설정 오타가 분석 전체를 죽이면 안 된다 — 모델 기본값으로 조용히 넘어간다."""
    monkeypatch.setattr(CFG, "gemini_thinking_budget", "빠르게")
    sleeps = []
    client, models = make_client(monkeypatch, [valid_response()], sleeps)

    result = client.analyze("ignored", 100, 100)

    assert result.num_people == 1
    assert models.thinking[0] is None


def test_thinking_config_is_not_sent_to_fallback_models(monkeypatch):
    """폴백 모델에는 사고 토큰 설정을 보내지 않는다.

    2026-08-29 프로덕션 키 실측: lite 계열(gemini-flash-lite-latest,
    gemini-3.5-flash-lite)은 thinking_config를 400 INVALID_ARGUMENT로 거부한다.
    400은 이 파일의 규칙상 "우리 잘못"이라 폴백도 못 타고 500 + 알림이 된다 —
    상류 혼잡을 구제하려던 경로가 오히려 알림을 만드는 셈이다.
    """
    monkeypatch.setattr(CFG, "gemini_max_attempts", 1)
    sleeps = []
    client, models = make_client(
        monkeypatch, [ApiFailure(503), valid_response()], sleeps,
        fallbacks="backup-gemini",
    )

    client.analyze("ignored", 100, 100)

    assert models.thinking[0].thinking_budget == 0, "1차에는 설정이 나가야 한다"
    assert models.thinking[1] is None, "폴백에는 필드 자체를 안 보낸다"
