"""
Descriptor 결합(설계문서 v2 §5, VLM활용 파트 B).
VLM 의미 태그 + 스켈레톤 피처를 하나의 검색 키로 묶는다. LLM 불필요 — JSON 구조화로 충분.
"""
from __future__ import annotations

from typing import List

from .schema import VLMAnalysis, Skeleton, PersonDescriptor, BBox
from .features import normalize_skeleton


def build_descriptors(vlm: VLMAnalysis,
                      skeletons: List[Skeleton],
                      boxes: List[BBox]) -> List[PersonDescriptor]:
    """
    인물별 Descriptor 생성. 태그(shot/action/view/relationship)는 컷 단위 값을 공유하되,
    스켈레톤/피처/박스는 인물별로 갖는다.
    """
    out = []
    for i, sk in enumerate(skeletons):
        feat = normalize_skeleton(sk.keypoints, sk.scores) if sk is not None else None
        box = boxes[i] if i < len(boxes) else None
        out.append(PersonDescriptor(
            shot=vlm.shot, action=vlm.action, view=vlm.view,
            relationship=vlm.relationship,
            skeleton=sk, feature=feat, box=box,
        ))
    return out
