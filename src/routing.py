"""
컷 3갈래 라우팅(설계문서 v2 §6).
  전신·반신 → core(코어 파이프라인)
  흉상      → bust(파이프라인에서 관측 관절 기준 검색, 결과 route=core)
  얼굴      → skip(작가 직접)
채택 모델 Body는 흉상이 core로 오분류돼도 폭발하지 않음 → 판별이 완벽하지 않아도 안전.
"""
from __future__ import annotations

from .schema import VLMAnalysis, Shot


def route(vlm: VLMAnalysis) -> str:
    if vlm.shot == Shot.FACE:
        return "skip"
    if vlm.shot == Shot.BUST:
        return "bust"
    return "core"
