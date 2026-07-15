"""
공통 데이터 스키마.

파이프라인의 각 단계가 주고받는 자료구조를 한곳에 모아 둔다.
VLM은 '의미·개수·종류'만, 정밀 좌표는 검출기/포즈 모델만 담당한다는
설계 원칙(설계문서 v2 §3-4)을 타입 수준에서 드러낸다.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import json

import numpy as np


# ---- 열거형(Controlled Vocabulary) ---------------------------------------
# ⚠ 라이브러리 태깅 전에 반드시 고정해야 하는 어휘(설계문서 v2 §12-3단계).
# 나중에 바꾸면 전량 재태깅이 필요하므로 여기서 단일 소스로 관리한다.

class Shot(str, Enum):
    FULL_HALF = "full_half"   # 전신·반신 → 코어 파이프라인
    BUST = "bust"             # 흉상 → 상반신 방향·앵글
    FACE = "face"             # 얼굴 → 배치 스킵(작가 직접)


class Action(str, Enum):
    STANDING = "standing"
    SITTING = "sitting"
    WALKING = "walking"
    RUNNING = "running"
    REACHING = "reaching"
    LYING = "lying"
    OTHER = "other"


class View(str, Enum):
    # Facing(←→↖↗)은 View에 통합. '필터' 아닌 '우선순위'로 쓴다.
    FRONT = "front"
    SIDE = "side"
    BACK = "back"
    THREE_QUARTER = "three_quarter"


class Relationship(str, Enum):
    SOLO = "solo"
    TALKING = "talking"
    HUGGING = "hugging"
    HOLDING_HANDS = "holding_hands"
    FIGHTING = "fighting"

    @property
    def is_entangled(self) -> bool:
        """물리적으로 얽히는 관계 → 2인 상호작용 세트를 한 덩어리로 검색(설계문서 v2 §10)."""
        return self in (Relationship.HUGGING, Relationship.FIGHTING)


# COCO-17 키포인트 순서(RTMPose Body 출력 규격).
COCO17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


# ---- 자료구조 -------------------------------------------------------------

@dataclass
class BBox:
    """이미지 좌표계 박스. VLM은 대략(coarse), 검출기는 정밀(precise)."""
    x1: float
    y1: float
    x2: float
    y2: float
    source: str = "detector"   # "detector" | "vlm"
    score: float = 1.0

    def as_list(self) -> list:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class Skeleton:
    """한 인물의 2D 스켈레톤. keypoints: (17,2), scores: (17,)."""
    keypoints: np.ndarray
    scores: np.ndarray

    def to_serializable(self) -> dict:
        return {
            "keypoints": np.asarray(self.keypoints).tolist(),
            "scores": np.asarray(self.scores).tolist(),
        }


@dataclass
class VLMAnalysis:
    """
    VLM이 컷 1개에서 뽑는 '의미' 결과.
    좌표는 approx_boxes(대략)만 담는다 — 관절 좌표는 절대 담지 않는다.
    """
    num_people: int
    shot: Shot
    action: Action
    view: View
    relationship: Relationship
    approx_boxes: list = field(default_factory=list)
    dialogue: Optional[str] = None
    raw: dict = field(default_factory=dict)


@dataclass
class PersonDescriptor:
    """검색 키 1인분. VLM 의미 태그 + 스켈레톤 피처 결합(LLM 불필요, JSON 구조화)."""
    shot: Shot
    action: Action
    view: View
    relationship: Relationship
    skeleton: Optional[Skeleton]
    feature: Optional[np.ndarray]
    box: Optional[BBox] = None

    def tag_dict(self) -> dict:
        return {
            "shot": self.shot.value,
            "action": self.action.value,
            "view": self.view.value,
            "relationship": self.relationship.value,
        }


@dataclass
class CutResult:
    """컷 1개 처리 결과(도원 담당 범위의 최종 출력)."""
    route: str
    count_confidence: str
    detector_count: int
    vlm_count: int
    descriptors: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    person_candidates: list = field(default_factory=list)  # 인물별 Top-K
    person_confidence: list = field(default_factory=list)  # 인물별 'high'|'low(폴백)'
    notes: list = field(default_factory=list)


@dataclass
class LibraryEntry:
    """
    포즈 라이브러리 1개 항목.
    3D 포즈(BVH)를 여러 가상 카메라로 2D 투영한 색인 중 하나에 해당.
    """
    pose_id: str
    view: View
    feature: np.ndarray
    tags: dict
    bvh_path: Optional[str] = None
    meta: dict = field(default_factory=dict)


@dataclass
class PoseCandidate:
    """검색 결과 후보 1개."""
    pose_id: str
    view: View
    distance: float
    tags: dict
    bvh_path: Optional[str] = None
    rerank_score: Optional[float] = None


def dumps(obj) -> str:
    """디버깅용 직렬화(ndarray/Enum 안전 처리)."""
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, "__dataclass_fields__"):
            return asdict(o)
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)
    return json.dumps(obj, default=default, ensure_ascii=False, indent=2)
