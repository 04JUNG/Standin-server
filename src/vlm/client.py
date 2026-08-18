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
    )


def _extract_json(text: str) -> dict:
    """모델 출력에서 첫 JSON 블록만 안전하게 파싱."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"VLM 응답에서 JSON을 찾지 못함: {text[:200]}")
    return json.loads(m.group(0))


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
        return VLMAnalysis(num, shot, action, view, rel, boxes,
                           dialogue=None, raw={"mock": True, "hint": hint})


class GeminiVLMClient(BaseVLMClient):
    """Gemini Flash 어댑터 (현행 google-genai SDK). `pip install google-genai pillow`, env GEMINI_API_KEY.
    모델은 GEMINI_MODEL env로 지정(기본 gemini-2.5-flash). 최신 Flash로 올리려면 그 값을 바꾼다."""
    def __init__(self, model: Optional[str] = None, *, client=None, sleep=time.sleep,
                 jitter=random.uniform):
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
        for attempt in range(1, attempts + 1):
            started = time.monotonic()
            try:
                resp = self._client.models.generate_content(
                    model=self._model,
                    contents=[prompts.USER_TEMPLATE, part],
                    config=types.GenerateContentConfig(
                        system_instruction=prompts.SYSTEM,
                        response_mime_type="application/json",
                        temperature=0,
                    ),
                )
                log_info(
                    "gemini_request",
                    model=self._model,
                    attempt=attempt,
                    status="ok",
                    elapsedMs=round((time.monotonic() - started) * 1000),
                )
                return _coerce(_extract_json(resp.text), img_w, img_h)
            except Exception as error:
                status = _http_status(error)
                retryable = status in (429, 503) and attempt < attempts
                log_warn(
                    "gemini_request",
                    model=self._model,
                    attempt=attempt,
                    status=status or "error",
                    retry=retryable,
                    elapsedMs=round((time.monotonic() - started) * 1000),
                    errorCode=f"GEMINI_{status}" if status else "GEMINI_ERROR",
                )
                if not retryable:
                    raise
                delay = min(
                    CFG.gemini_retry_base_seconds * (2 ** (attempt - 1)),
                    CFG.gemini_retry_max_seconds,
                )
                self._sleep(self._jitter(delay * 0.8, delay * 1.2))
        raise RuntimeError("unreachable")


def _http_status(error: Exception) -> int | None:
    """google-genai APIError와 테스트 대역에서 HTTP 상태 코드를 안전하게 읽는다."""
    for name in ("code", "status_code"):
        value = getattr(error, name, None)
        if isinstance(value, int):
            return value
    return None


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
