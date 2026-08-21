"""`/analyze`가 상류 VLM 혼잡을 어떻게 번역하는지 고정한다.

2026-08-21 프로덕션: Gemini가 503 UNAVAILABLE("high demand")을 돌려주자 그 예외가
`/analyze`를 그대로 빠져나가 500 + P2 UNHANDLED_ERROR 알림이 됐다. 운영에서는 "우리
버그"로 분류됐고, BFF는 500을 INFERENCE_FAILED로 접어 사용자에게 "다른 이미지로 다시
시도해 주세요"라고 안내했다 — 상류가 붐비는 동안에는 어떤 이미지도 실패한다.
"""

from __future__ import annotations

from fastapi import HTTPException

from api.app import VLM_RETRY_AFTER_SECONDS, _vlm_unavailable_response
from src.vlm.client import VLMUnavailable


def test_upstream_congestion_becomes_503_with_retry_after():
    exc = _vlm_unavailable_response(
        VLMUnavailable("상류 혼잡", status=503, attempts=3, elapsed_seconds=47.7)
    )

    assert isinstance(exc, HTTPException)
    # 500이 아니라 503이어야 BFF·클라이언트가 "일시적, 다시 시도"로 다룰 수 있다.
    assert exc.status_code == 503
    assert exc.detail["code"] == "VLM_UNAVAILABLE"
    assert exc.headers["Retry-After"] == str(VLM_RETRY_AFTER_SECONDS)
    # 사용자에게 보이는 문구는 원인과 다음 행동을 담는다(이미지 탓으로 돌리지 않는다).
    assert "잠시 후" in exc.detail["message"]


def test_timeout_without_status_is_still_translated():
    """timeout에는 HTTP 상태가 없다. 그래도 원인은 상류 지연이라 같은 안내를 준다."""
    exc = _vlm_unavailable_response(
        VLMUnavailable("데드라인", status=None, attempts=1, elapsed_seconds=45.0)
    )

    assert exc.status_code == 503
    assert exc.detail["code"] == "VLM_UNAVAILABLE"
