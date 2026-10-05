"""공백 분석에 넣을 관측 고르기.

라이브러리 공백만 보려면 추출 실패를 먼저 빼야 한다. 관절이 틀렸는데 거리가 멀게
나온 것은 라이브러리에 포즈를 더해도 해결되지 않는다.

- 정량 조건은 `pose_curation/coverage`(inference.py·evaluate.py)와 같다:
  점수 0.3 이상 기준으로 몸통 4점이 모두 보이고, 몸 관절 12개 중 10개 이상이 보이고,
  어깨 중점–엉덩이 중점 거리가 20px 이상.
- 운영 조건: VLM 슬롯, 전체 이미지 추출, state valid/partial, coverage full/reduced,
  전신 검색, 얽힘 아님, 관절이 틀렸다는 사용자 피드백 없음.
"""
from __future__ import annotations

import numpy as np

from .observations import Observation

TORSO = (5, 6, 11, 12)
SKELETON_FEEDBACK = frozenset({"skeleton_wrong", "person_missing"})
ENTANGLED = frozenset({"hugging", "fighting"})


def quantitative_eligible(keypoints, scores, kpt_thr: float = 0.3) -> bool:
    kp = np.asarray(keypoints, dtype=np.float32).reshape(17, 2)
    valid = np.asarray(scores, dtype=np.float32).reshape(17) >= kpt_thr
    torso_visible = bool(valid[list(TORSO)].all())
    body_visible = int(valid[5:].sum())
    torso_pixels = float(np.linalg.norm(kp[[5, 6]].mean(0) - kp[[11, 12]].mean(0)))
    return torso_visible and body_visible >= 10 and torso_pixels >= 20


def ineligibility(obs: Observation, kpt_thr: float = 0.3) -> str | None:
    """적격이면 None, 아니면 빠진 이유 코드."""
    if obs.slot_origin != "vlm":
        return "slot_origin"
    if obs.skeleton_source != "full_image":
        return "skeleton_source"
    if obs.skeleton_state not in ("valid", "partial"):
        return "skeleton_state"
    if obs.coverage_class not in ("full", "reduced"):
        return "coverage_class"
    if obs.search_scope != "full_body":
        return "search_scope"
    if str(obs.cut.get("tags", {}).get("relationship")) in ENTANGLED:
        return "entangled"
    if obs.behavior.get("job_feedback") in SKELETON_FEEDBACK:
        return "skeleton_feedback"
    if obs.raw_scores is None or obs.evidence_mask is None or obs.search_mask is None:
        return "missing_signals"
    if not obs.candidates or obs.rank_distance is None:
        return "no_candidates"
    if not quantitative_eligible(obs.keypoints, obs.raw_scores, kpt_thr):
        return "quantitative"
    return None
