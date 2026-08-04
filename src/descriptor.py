"""
Descriptor 결합(설계문서 v2 §5, VLM활용 파트 B).
VLM 의미 태그 + 스켈레톤 피처를 하나의 검색 키로 묶는다. LLM 불필요 — JSON 구조화로 충분.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .schema import VLMAnalysis, Skeleton, PersonDescriptor, BBox
from .features import normalize_skeleton


def order_left_to_right(skeletons: List[Optional[Skeleton]],
                        boxes: List[Optional[BBox]]) -> Tuple[list, list]:
    """인물을 화면 왼쪽부터 정렬한다(좌→우 인물 태깅).

    person_index가 '검출/추출 순서'가 아니라 '컷에서 보이는 좌→우 위치'를
    따르도록, (skeleton, box) 쌍을 박스 좌측 x좌표 기준으로 정렬해 돌려준다.
    박스가 없으면 스켈레톤 x중앙값으로 대체하고, 둘 다 없으면 맨 뒤로 보낸다.
    정렬 키가 같으면 원래 순서를 유지한다(안정 정렬).
    """
    n = max(len(skeletons), len(boxes))

    def left_x(i: int) -> float:
        box = boxes[i] if i < len(boxes) else None
        if box is not None:
            return float(box.x1)
        skel = skeletons[i] if i < len(skeletons) else None
        if skel is not None:
            xs = np.asarray(skel.keypoints, dtype=np.float32).reshape(-1, 2)[:, 0]
            if xs.size:
                return float(xs.mean())
        return float("inf")

    order = sorted(range(n), key=left_x)
    sk_out = [skeletons[i] if i < len(skeletons) else None for i in order]
    bx_out = [boxes[i] if i < len(boxes) else None for i in order]
    return sk_out, bx_out


def build_descriptors(vlm: VLMAnalysis,
                      skeletons: List[Optional[Skeleton]],
                      boxes: List[Optional[BBox]]) -> List[PersonDescriptor]:
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


def build_slot_descriptors(vlm: VLMAnalysis, slots) -> List[PersonDescriptor]:
    """복구가 끝난 PersonSlot을 검색 descriptor로 변환한다.

    invalid/missing 슬롯도 descriptor 자리를 유지해 뒤 인물의 person_index가 당겨지지
    않게 한다. 명시적 mask는 순위 거리에 전달하고 raw score는 평가를 위해 보존한다.
    """
    out: List[PersonDescriptor] = []
    for slot in slots:
        skeleton = slot.skeleton
        evidence = slot.evidence
        searchable = bool(skeleton is not None and evidence is not None
                          and evidence.searchable and slot.state not in ("missing", "invalid"))
        valid_mask = evidence.valid_joint_mask.copy() if evidence is not None else None
        feature = None
        if searchable:
            feature = normalize_skeleton(
                skeleton.keypoints, skeleton.scores, valid_mask=valid_mask
            )
        refine_allowed = bool(
            searchable
            and slot.state == "valid"
            and evidence.coverage_class == "full"
            and slot.slot_origin != "rtm_provisional"
            and slot.skeleton_source != "crop_retry"
        )
        # hard invalid 좌표(NaN/Inf 포함)는 네트워크 경계로 내보내지 않는다. raw 값은
        # slot/evidence trace에만 남기고 descriptor 자리는 유지한다.
        output_skeleton = skeleton if slot.state not in ("missing", "invalid") else None
        if output_skeleton is not None:
            output_scores = evidence.effective_scores
            if not refine_allowed:
                # PR #7 전 API에는 refine_allowed가 없다. score를 0으로 보내면 기존
                # /refine의 low_skeleton_score 게이트가 안전하게 베이스를 반환한다.
                output_scores = np.zeros(17, dtype=np.float32)
            output_skeleton = Skeleton(
                np.asarray(output_skeleton.keypoints, dtype=np.float32).copy(),
                output_scores,
            )
        out.append(PersonDescriptor(
            shot=vlm.shot,
            action=vlm.action,
            view=vlm.view,
            relationship=vlm.relationship,
            skeleton=output_skeleton,
            feature=feature,
            box=slot.result_box,
            valid_joint_mask=valid_mask,
            skeleton_state=slot.state,
            coverage_class=(evidence.coverage_class if evidence is not None
                            else "insufficient"),
            slot_origin=slot.slot_origin,
            refine_allowed=refine_allowed,
            quality_reasons=list(dict.fromkeys(slot.reasons)),
        ))
    return out
