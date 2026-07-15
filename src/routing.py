"""
컷 3갈래 라우팅(설계문서 v2 §6).
  전신·반신 → core(코어 파이프라인)
  흉상      → bust(상반신 방향·앵글 — MVP 후순위, 여기선 라우팅만)
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
