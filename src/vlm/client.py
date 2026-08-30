"""
VLM 클라이언트 추상화.

한 provider에 묶이지 않도록 인터페이스로 감싼다(설계 원칙: 기성 모델, 교체 가능).
- MockVLMClient  : 오프라인 기본. 규칙 기반으로 그럴듯한 결과 반환 → API 키 없이 파이프라인 전체 실행.
- GeminiVLMClient : gemini-2.5-flash. 저비용·그라운딩·구조화 출력에 강함(2주 스프린트 권장 기본값).
- OpenAIVLMClient : gpt-5-mini. 구조화 출력이 매우 안정적인 대안.

실제 어댑터는 import를 지연시켜 의존성 없이도 mock이 돌아가게 한다.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Optional

from ..schema import VLMAnalysis, BBox, Shot, Action, View, Relationship
from ..config import CFG
from ..logging_setup import log_info, log_warn
from . import prompts


def _coerce(analysis: dict, img_w: int, img_h: int) -> VLMAnalysis:
    """VLM이 준 dict(JSON)를 타입 안전한 VLMAnalysis로 변환. 잘못된 값은 안전 폴백."""
    def pick(enum_cls, val, default):
        try:
            return enum_cls(val)
        except Exception:
            return default

    boxes = []
    for b in analysis.get("approx_boxes", []) or []:
        try:
            # 0~1 정규화 좌표를 픽셀로 환산(대략)
            boxes.append(BBox(
                x1=float(b["x1"]) * img_w, y1=float(b["y1"]) * img_h,
                x2=float(b["x2"]) * img_w, y2=float(b["y2"]) * img_h,
                source="vlm", score=0.5,
            ))
        except Exception:
            continue

    num = int(analysis.get("num_people", len(boxes)) or 0)
    raw_lower = analysis.get("lower_body_visible")
    if isinstance(raw_lower, list) and len(raw_lower) == num:
        lower_body_visible = [value is True for value in raw_lower]
        lower_body_visibility_known = [
            isinstance(value, bool) for value in raw_lower
        ]
    else:
        # refine은 fail-closed로 동결하되, 검색은 누락을 '반신'으로 해석하지 않는다.
        lower_body_visible = [False] * max(num, 0)
        lower_body_visibility_known = [False] * max(num, 0)
    return VLMAnalysis(
        num_people=num,
        shot=pick(Shot, analysis.get("shot"), Shot.FULL_HALF),
        action=pick(Action, analysis.get("action"), Action.OTHER),
        view=pick(View, analysis.get("view"), View.FRONT),
        relationship=pick(Relationship, analysis.get("relationship"),
                          Relationship.SOLO if num <= 1 else Relationship.TALKING),
        approx_boxes=boxes,
        dialogue=analysis.get("dialogue"),
        raw=analysis,
        lower_body_visible=lower_body_visible,
        lower_body_visibility_known=lower_body_visibility_known,
    )


def _extract_json(text: str) -> dict:
    """모델 출력에서 첫 JSON 블록만 안전하게 파싱."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"VLM 응답에서 JSON을 찾지 못함: {text[:200]}")
    return json.loads(m.group(0))


class VLMUnavailable(RuntimeError):
    """상류 VLM이 **일시적으로** 요청을 처리하지 못했다(429·5xx 소진, 데드라인).

    파이프라인 버그와 구분하기 위한 타입이다. 이 구분이 없으면 Gemini 과부하가
    `/analyze`에서 처리되지 않은 예외로 올라가 500 + P2 UNHANDLED_ERROR 알림이 되고,
    사용자는 "다른 이미지로 다시 시도해 주세요"라는 잘못된 안내를 받는다. 다른 이미지를
    넣어도 상류가 붐비는 동안에는 똑같이 실패한다.

    호출자(api/app.py)가 503 + Retry-After로 번역한다.
    """

    def __init__(self, message: str, *, status: int | None = None,
                 attempts: int = 0, elapsed_seconds: float = 0.0):
        super().__init__(message)
        self.status = status
        self.attempts = attempts
        self.elapsed_seconds = elapsed_seconds


class BaseVLMClient:
    def analyze(self, image, img_w: int, img_h: int) -> VLMAnalysis:  # noqa
        raise NotImplementedError

    def rerank(self, image, candidates: list, query_tags: dict) -> list:
        """
        Top-N 후보를 의미 기준으로 재정렬(선택). 기본은 no-op(순서 유지).
        실제 구현은 후보 썸네일 + 쿼리 컷을 함께 넣어 순위를 반환하게 만든다.
        """
        return list(range(len(candidates)))


class MockVLMClient(BaseVLMClient):
    """
    이미지 파일명/메타의 힌트를 규칙으로 읽어 결과를 만든다.
    파이프라인·검색·스키마를 API 없이 end-to-end로 검증하기 위한 스텁.
    실제 그림 이해는 하지 않는다(그건 Gemini/OpenAI 어댑터의 몫).
    """
    def analyze(self, image, img_w: int, img_h: int) -> VLMAnalysis:
        hint = str(getattr(image, "hint", "") or image if isinstance(image, str) else "")
        hint = hint.lower()

        shot = Shot.FULL_HALF
        if "bust" in hint or "흉상" in hint:
            shot = Shot.BUST
        elif "face" in hint or "얼굴" in hint:
            shot = Shot.FACE

        action = Action.STANDING
        for a in Action:
            if a.value in hint:
                action = a
        view = View.FRONT
        for v in View:
            if v.value in hint:
                view = v

        num = 2 if ("2p" in hint or "two" in hint or "2인" in hint) else 1
        rel = Relationship.SOLO
        if num >= 2:
            rel = Relationship.HUGGING if "hug" in hint else Relationship.TALKING

        boxes = []
        if num == 1:
            boxes = [BBox(0.3*img_w, 0.1*img_h, 0.7*img_w, 0.95*img_h, "vlm", 0.5)]
        else:
            boxes = [
                BBox(0.05*img_w, 0.1*img_h, 0.5*img_w, 0.95*img_h, "vlm", 0.5),
                BBox(0.5*img_w, 0.1*img_h, 0.95*img_w, 0.95*img_h, "vlm", 0.5),
            ]
        lower_hidden = any(token in hint for token in (
            "half_body", "half-body", "lower_hidden", "반신", "하체 비관측",
        ))
        return VLMAnalysis(
            num, shot, action, view, rel, boxes,
            dialogue=None,
            raw={
                "mock": True,
                "hint": hint,
                "lower_body_visible": [not lower_hidden] * num,
            },
            lower_body_visible=[not lower_hidden] * num,
            lower_body_visibility_known=[True] * num,
        )


class GeminiVLMClient(BaseVLMClient):
    """Gemini Flash 어댑터 (현행 google-genai SDK). `pip install google-genai pillow`, env GEMINI_API_KEY.
    모델은 GEMINI_MODEL env로 지정(기본 gemini-2.5-flash). 최신 Flash로 올리려면 그 값을 바꾼다."""
    def __init__(self, model: Optional[str] = None, *, client=None, sleep=time.sleep,
                 jitter=random.uniform, clock=time.monotonic):
        if client is None:
            from google import genai            # 신 SDK: from google import genai
            from google.genai import types
            import os
            client = genai.Client(
                api_key=os.environ["GEMINI_API_KEY"],
                http_options=types.HttpOptions(timeout=CFG.gemini_request_timeout_ms),
            )
        self._client = client
        self._model = model or CFG.gemini_model
        self._sleep = sleep
        self._jitter = jitter
        # 예산 계산의 시계. 테스트가 느린 503을 시뮬레이션하려면 주입할 수 있어야 한다.
        self._clock = clock

    @staticmethod
    def _to_part(image):
        """PIL.Image 또는 파일경로 → genai Part(png bytes)."""
        from google.genai import types
        from PIL import Image
        import io
        img = Image.open(image) if isinstance(image, str) else image
        buf = io.BytesIO(); img.convert("RGB").save(buf, format="PNG")
        return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")

    def analyze(self, image, img_w: int, img_h: int) -> VLMAnalysis:
        from google.genai import types
        part = self._to_part(image)
        attempts = max(1, CFG.gemini_max_attempts)
        # 재시도가 BFF의 분석 상한(기본 120초)을 먹어 치우지 않게 VLM 단계 전체에 예산을 둔다.
        # 예산이 없으면 timeout을 올릴 때마다 최악 지연이 attempts배로 늘어난다.
        full_timeout = CFG.gemini_request_timeout_ms / 1000
        min_attempt = max(1.0, CFG.gemini_min_attempt_seconds)
        started_all = self._clock()
        deadline = started_all + CFG.gemini_total_budget_seconds
        for attempt in range(1, attempts + 1):
            # ⚠ 이번 시도의 상한은 **남은 예산**이다. 전체 timeout을 그대로 쓰면 예산을
            #   넘겨 BFF가 먼저 끊는다. 반대로 "남은 예산이 전체 timeout보다 작으면
            #   아예 시도하지 않는" 규칙이면 재시도가 사실상 사라진다 — 실제로 503은
            #   4~35초씩 걸려 돌아와서, 예산 75초에 45초 timeout이면 두 번째 시도조차
            #   못 했다(2026-08-21 프로덕션: 실패 4건 중 2건이 budget_exhausted).
            remaining = deadline - self._clock()
            # 0 이하로 내려가지만 않으면 된다. 하한을 min_attempt로 두면 운영자가
            # 설정한 timeout보다 **긴** 시도가 나가 버린다.
            timeout_seconds = max(0.05, min(full_timeout, remaining))
            started = self._clock()
            try:
                resp = self._client.models.generate_content(
                    model=self._model,
                    contents=[prompts.USER_TEMPLATE, part],
                    config=types.GenerateContentConfig(
                        system_instruction=prompts.SYSTEM,
                        response_mime_type="application/json",
                        temperature=0,
                        # 요청 단위 상한. 클라이언트 생성 시의 값(전체 timeout)을 덮어쓴다.
                        http_options=types.HttpOptions(
                            timeout=int(timeout_seconds * 1000)),
                    ),
                )
                log_info(
                    "gemini_request",
                    model=self._model,
                    attempt=attempt,
                    status="ok",
                    elapsedMs=round((self._clock() - started) * 1000),
                )
                return _coerce(_extract_json(resp.text), img_w, img_h)
            except Exception as error:
                status = _http_status(error)
                # 429와 5xx는 상류 혼잡이다. timeout도 같은 부류로 본다 — 원인은 달라도
                # 사용자가 할 수 있는 일("잠시 후 다시")이 같고, 둘 다 우리 버그가 아니다.
                # 반면 4xx와 파싱 오류는 우리 잘못이므로 그대로 올려 보내 알림을 받는다.
                timed_out = _is_timeout(error)
                transient = status == 429 or (status is not None and 500 <= status < 600)
                upstream = transient or timed_out
                left = deadline - self._clock()
                # 남은 예산으로 의미 있는 시도를 한 번 더 할 수 있을 때만 재시도한다.
                budget_left = left >= min_attempt
                # timeout은 재시도하지 않는다 — 끊긴 호출도 Gemini 쪽에서는 계속 생성
                # 중일 수 있어 재시도가 비용만 두 배로 쓴다. 대신 데드라인을 넉넉히 잡는다.
                retryable = transient and attempt < attempts and budget_left
                reason = (
                    "retrying" if retryable
                    else "timed_out" if timed_out
                    else "not_transient" if not transient
                    else "attempts_exhausted" if attempt >= attempts
                    else "budget_exhausted"
                )
                log_warn(
                    "gemini_request",
                    model=self._model,
                    attempt=attempt,
                    status=status or "error",
                    retry=retryable,
                    retryDecision=reason,
                    timeoutSeconds=round(timeout_seconds, 1),
                    budgetLeftSeconds=round(max(0.0, left), 1),
                    elapsedMs=round((self._clock() - started) * 1000),
                    errorCode=f"GEMINI_{status}" if status else "GEMINI_ERROR",
                )
                if not retryable:
                    if upstream:
                        # 상류 혼잡을 파이프라인 버그와 같은 통로로 올리지 않는다.
                        raise VLMUnavailable(
                            f"VLM 상류가 응답하지 못했다({reason})",
                            status=status,
                            attempts=attempt,
                            elapsed_seconds=self._clock() - started_all,
                        ) from error
                    raise
                delay = min(
                    CFG.gemini_retry_base_seconds * (2 ** (attempt - 1)),
                    CFG.gemini_retry_max_seconds,
                )
                # 대기도 예산을 먹는다. 다음 시도 몫을 남기지 못할 만큼은 쉬지 않는다.
                delay = max(0.0, min(delay, left - min_attempt))
                if delay:
                    self._sleep(self._jitter(delay * 0.8, delay * 1.2))
        raise RuntimeError("unreachable")


def _http_status(error: Exception) -> int | None:
    """google-genai APIError와 테스트 대역에서 HTTP 상태 코드를 안전하게 읽는다."""
    for name in ("code", "status_code"):
        value = getattr(error, name, None)
        if isinstance(value, int):
            return value
    return None


# httpx의 timeout 계열. 이름으로 보는 이유는 아래 주석 참고.
_TIMEOUT_CLASS_NAMES = frozenset({
    "TimeoutError", "TimeoutException",
    "ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout",
})


def _is_timeout(error: Exception) -> bool:
    """요청 데드라인에 잘린 실패인가.

    이 모듈은 google-genai·httpx 없이도 import돼야 하므로(mock 우선 저장소, 코어 의존성은
    numpy뿐) isinstance 대신 클래스 이름으로 본다. 상태 코드가 없다는 것만으로 timeout이라고
    보면 안 된다 — JSON 파싱 실패 같은 **우리 버그**도 상태 코드가 없기 때문이다.
    """
    if isinstance(error, TimeoutError):
        return True
    return any(cls.__name__ in _TIMEOUT_CLASS_NAMES for cls in type(error).__mro__)


class OpenAIVLMClient(BaseVLMClient):
    """gpt-5-mini 등 OpenAI 비전 어댑터. `pip install openai`, env OPENAI_API_KEY 필요."""
    def __init__(self, model: Optional[str] = None):
        from openai import OpenAI  # 지연 import
        self._client = OpenAI()
        self._model = model or CFG.openai_model

    def analyze(self, image_data_url: str, img_w: int, img_h: int) -> VLMAnalysis:
        resp = self._client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompts.SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": prompts.USER_TEMPLATE},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ]},
            ],
        )
        return _coerce(_extract_json(resp.choices[0].message.content), img_w, img_h)


def build_vlm_client() -> BaseVLMClient:
    """CFG.vlm_provider에 따라 어댑터 생성. 실패 시 mock 폴백."""
    p = CFG.vlm_provider.lower()
    try:
        if p == "gemini":
            return GeminiVLMClient()
        if p == "openai":
            return OpenAIVLMClient()
    except Exception as e:  # 키/패키지 없음 등
        # 조용한 폴백은 프로덕션에서 runtime_guard가 기동을 막는다. 여기서는
        # "왜" 폴백했는지를 남긴다 — 가드가 잡은 뒤 원인을 찾는 유일한 단서다.
        log_warn("backend_fallback", "VLM 초기화 실패 → mock 폴백",
                 errorCode="VLM_BACKEND_INIT_FAILED", backend=p,
                 errorName=type(e).__name__, detail=str(e)[:300])
    return MockVLMClient()
