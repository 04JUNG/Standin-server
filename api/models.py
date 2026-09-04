"""
API 응답 모델(Pydantic). 이것이 앱 서버 팀과의 '계약'이다 — /docs(OpenAPI)로 자동 노출된다.
schema.py의 dataclass를 미러링하되, 네트워크 경계에 맞게 bvh_url 등 소비자 친화 필드를 더한다.
"""
from __future__ import annotations

import math
from typing import Annotated, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class CandidateOut(BaseModel):
    pose_id: str
    view: str
    distance: float
    tags: dict
    rerank_score: Optional[float] = None
    bvh_url: str = Field(..., description="동원 내보내기 팀이 받는 라이브러리 BVH 다운로드 경로")
    thumbnail_url: Optional[str] = Field(None, description="후보 시점 PNG 썸네일의 내부 다운로드 경로")


Point2D = Annotated[List[float], Field(min_length=2, max_length=2)]
Keypoints17 = Annotated[List[Point2D], Field(min_length=17, max_length=17)]
Scores17 = Annotated[List[float], Field(min_length=17, max_length=17)]


class SkeletonOut(BaseModel):
    schema_version: Literal["coco17-v1"] = "coco17-v1"
    keypoints: Keypoints17
    scores: Scores17


class ImageInfoOut(BaseModel):
    width: int
    height: int


class InferenceMetadataOut(BaseModel):
    deployment_version: str
    vlm_provider: str
    vlm_model: str
    pose_backend: str
    pose_model_version: str
    pose_library_version: str
    feature_version: int


class PersonOut(BaseModel):
    index: int
    box: Optional[List[float]] = Field(None, description="[x1,y1,x2,y2] 픽셀")
    tags: dict
    skeleton: Optional[SkeletonOut] = None
    candidates: List[CandidateOut]
    # /refine을 '순수 함수'로 만들기 위한 필드(docs/REFINE_DESIGN.md).
    # /analyze가 이미 추출한 값을 실어 보낼 뿐이라 연산 추가는 0이다.
    # 클라이언트는 이 두 값을 그대로 POST /refine에 되돌려주면 된다
    # → 러프 이미지 재전송·포즈 재추론 없음.
    keypoints: Optional[List[List[float]]] = Field(
        None, description="러프에서 추출한 2D 스켈레톤 (17×2, 이미지 픽셀 좌표)")
    scores: Optional[List[float]] = Field(
        None, description="관절별 신뢰도 (17,). 낮은 관절은 refine 손실에서 제외됨")
    raw_scores: Optional[List[float]] = Field(
        None, description="평가용 RTMPose 원본 score. scores는 구조 마스킹/안전정책 반영값")
    confidence: str = Field("low", description="high | low")
    skeleton_state: str = Field(
        "valid", description="valid | partial | suspect | missing | invalid")
    skeleton_source: str = Field(
        "full_image", description="none | full_image | crop_retry")
    coverage_class: str = Field(
        "full", description="full | reduced | sparse | insufficient")
    slot_origin: str = Field("vlm", description="vlm | rtm_provisional")
    search_stability: Optional[str] = Field(
        None, description="stable | ambiguous | unstable | not_required | not_available")
    distance_metric: Optional[str] = None
    rank_distance: Optional[float] = None
    confidence_threshold: Optional[float] = None
    valid_limbs: List[str] = Field(default_factory=list)
    refinable_limbs: List[str] = Field(default_factory=list)
    lower_body_observed: bool = Field(
        False,
        description=("VLM이 해당 인물의 양쪽 골반·무릎·발목을 실제 관측했다고 "
                     "판정했는지. false면 모든 하체 refine 차단"),
    )
    refine_allowed: bool = False
    quality_trace: dict = Field(default_factory=dict)
    quality_reasons: List[str] = Field(default_factory=list)


class CutResultOut(BaseModel):
    route: str = Field(..., description="core | bust | skip")
    count_confidence: str = Field(..., description="high(개수 일치) | low(불일치→폴백) | n/a")
    detector_count: int
    vlm_count: int
    people: List[PersonOut] = []
    notes: List[str] = []
    image: ImageInfoOut
    inference_metadata: InferenceMetadataOut


# ==== 포즈 미세조정 (docs/REFINE_DESIGN.md) ================================
# 작가가 Top-K 중 '고른 1개'만 러프에 맞춰 조정한다. 계산은 커밋된 포즈에만 든다.

class RefineRequest(BaseModel):
    pose_id: str = Field(..., description="작가가 고른 후보의 pose_id")
    view: Literal["front", "three_quarter", "side", "back"] = Field(
        ..., description="그 후보의 view(=매칭된 투영 각도)")
    keypoints: Keypoints17 = Field(
        ..., description="/analyze가 준 PersonOut.keypoints를 그대로 (17×2)")
    scores: Optional[Scores17] = Field(
        None, description="/analyze가 준 PersonOut.scores를 그대로 (17,)")
    search_distance: Optional[float] = Field(
        None, description=("그 후보의 distance. v1에서는 베이스 불일치 게이트, "
                           "v2.5에서는 실행을 막지 않는 검색 lineage"))
    refine_allowed: Optional[bool] = Field(
        None, description=("/analyze가 준 값을 그대로 전달. false면 차단하며 "
                           "v2.5에서는 policy lineage로 필수"))
    refinable_limbs: Optional[List[str]] = Field(
        None, description=("/analyze가 허용한 사지만 그대로 전달. 예: left_arm. "
                           "v2.5 policy lineage로 필수"))
    lower_body_observed: Optional[bool] = Field(
        None,
        description=("/analyze가 준 값을 그대로 전달. true가 아니면 leg/lower_pair/"
                     "leg-driving lap_contact/ankle 조정을 모두 차단"),
    )
    skeleton_state: Optional[str] = Field(
        None, description="/analyze의 skeleton_state; v2 진단 lineage")
    coverage_class: Optional[str] = Field(
        None, description="/analyze의 coverage_class; v2 진단 lineage")
    slot_origin: Optional[str] = Field(
        None, description="/analyze의 slot_origin; 인물 소유권 lineage")
    skeleton_source: Optional[str] = Field(
        None, description="/analyze의 skeleton_source; 인물 소유권 lineage")
    search_stability: Optional[str] = Field(
        None, description="partial 스켈레톤의 보수적 mask 검색 안정성")
    distance_metric: Optional[str] = Field(
        None, description="후보를 검색할 때 사용한 거리 metric")
    confidence_threshold: Optional[float] = Field(
        None, description="검색 당시 metric×coverage confidence threshold")
    gap_type: Literal["near_gap", "structural_gap", "unknown"] = Field(
        "unknown", description="평가 라벨. refine 실행 게이트로 사용하지 않음")
    refine_mode: Optional[Literal["conservative", "aggressive"]] = Field(
        None,
        description=("v2.5 조정 정책. 생략/null이면 서버 REFINE_DEFAULT_MODE를 사용. "
                     "aggressive도 final selector 실패 시 conservative/base로 복구"),
    )

    @field_validator("keypoints")
    @classmethod
    def _finite_keypoints(cls, value):
        if not all(math.isfinite(float(number)) for point in value for number in point):
            raise ValueError("keypoints must contain only finite values")
        return value

    @field_validator("scores")
    @classmethod
    def _finite_scores(cls, value):
        if value is not None and (not all(math.isfinite(float(number)) for number in value)
                                  or any(float(number) < 0.0 for number in value)):
            raise ValueError("scores must contain finite non-negative values")
        return value

    @field_validator("search_distance", "confidence_threshold")
    @classmethod
    def _finite_optional_number(cls, value):
        if value is not None and not math.isfinite(float(value)):
            raise ValueError("distance metadata must be finite")
        return value


class RefineThumbnailOut(BaseModel):
    """Stateless preview image returned inline with the final refine result.

    2026-09-04부터 후보 썸네일과 같은 렌더러(V3.2.5 converter가 변환한 FBX 남성
    모델의 anatomical 카메라 렌더)로 그린다. ``renderer_version``이
    ``fbx-anatomical-…``이면 그 경로, ``warm-mannequin-v1``이면 옛 2D 마네킹이다.
    ``media_type``은 기본 PNG이며 서버 설정(REFINE_THUMBNAIL_FORMAT)으로 JPEG가 될
    수 있으므로 표시할 때는 ``data:{media_type};base64,{data}``로 조립한다.
    """
    view: Literal["front", "three_quarter", "side", "back"]
    media_type: Literal["image/png", "image/jpeg"] = "image/png"
    encoding: Literal["base64"] = "base64"
    data: str = Field(..., description="base64-encoded 256×256 image bytes")
    width: Literal[256] = 256
    height: Literal[256] = 256
    renderer_version: str = Field(
        "fbx-anatomical-v1",
        description="그린 렌더러. fbx-anatomical-v1/<character_id> | "
                    "fbx-anatomical-v1/library | warm-mannequin-v1",
    )


class RefineResponse(BaseModel):
    """refined=False여도 오류가 아니다 — 안전 게이트가 조정을 버리고 베이스를 준 것."""
    pose_id: str
    view: str
    refined: bool = Field(..., description="조정본이 나왔는가. False면 bvh_url=베이스")
    reason: str = Field(..., description="ok | disabled | entangled_set | "
                                         "skeleton_policy | low_skeleton_score | base_mismatch | "
                                         "multiframe_base | insufficient_target_bones | "
                                         "low_observability | no_solvable_joints | "
                                         "already_matched | no_gain | ok_partial | "
                                         "movement_gate | global_no_gain | "
                                         "collision_gate | collision_unresolved | "
                                         "joint_limit | safety_gate | unchanged_geometry | "
                                         "timeout | final_collision_gate | "
                                         "final_extension_gate | diverged")
    bvh_url: str = Field(..., description="베이스 BVH 경로(/pose/{id}/bvh). refined 여부와 "
                                          "무관하게 항상 베이스다 — 조정본에는 URL이 없다.")
    bvh: Optional[str] = Field(
        None, description="조정본 BVH 본문(LF 개행). refined=true일 때만 채운다. "
                          "조정본을 얻는 **유일한** 경로이므로 소비자는 이 값을 받아 "
                          "자기 저장소에 보관해야 한다.")
    thumbnail: Optional[RefineThumbnailOut] = Field(
        None,
        description=("선택 후보의 매칭 view로 그린 최종 결과 256×256 이미지. "
                     "후보 썸네일과 같은 FBX 남성 모델 렌더러(converter)를 쓰며 "
                     "base64로 인라인 반환한다. 렌더에 실패하면 null이다 — 오류가 "
                     "아니라 '그림 없음'이며 나머지 필드는 그대로 유효하다."),
    )
    loss_base: Optional[float] = Field(None, description="조정 전 각도 손실")
    loss_final: Optional[float] = Field(None, description="조정 후 각도 손실")
    gain: Optional[float] = Field(None, description="손실 감소율(0.3=30% 개선)")
    backend: str = Field(..., description="scipy | numpy | scipy+numpy | none")
    refine_version: str = Field("v2.5.1", description="실행한 refine policy/code 버전")
    refine_outcome: Literal["improved", "unchanged", "reverted", "not_attempted"] = Field(
        "not_attempted", description="gap_type과 독립인 적용 결과")
    limbs: List[str] = Field(
        default_factory=list, description="P3까지 통과해 최종 조정된 사지")
    limb_decisions: dict = Field(
        default_factory=dict,
        description=("고려한 사지별 채택 여부·사유, 몸통 정규화 3D 이동량, "
                     "베이스 상대 손·전완-몸통 충돌 진단"))
    diagnostics: dict = Field(
        default_factory=dict,
        description="v2의 base/solved/adopted 손실·안전·버전 lineage")


# ==== 동원 Export 계약 (Tauri → 동원 내보내기) ============================
# /analyze(Top5 보여주기)와 별개. 작가가 '고른 하나'를 동원에게 넘기는 주문서.

class ExportSelection(BaseModel):
    """Tauri가 보내는 입력: 작가가 person별로 고른 후보."""
    person_index: int = Field(..., description="컷 안 인물 인덱스(0부터)")
    pose_id: str = Field(..., description="선택한 후보의 pose_id")
    view: str = Field(..., description="선택한 후보의 view(=매칭된 투영 각도)")


class ExportOrderRequest(BaseModel):
    cut_id: str = Field(..., description="컷 식별자(회차_컷번호 등)")
    source_image: Optional[str] = Field(None, description="원본 러프 파일명(추적용)")
    selections: List[ExportSelection]


class ExportItem(BaseModel):
    """동원이 실제로 소비하는 항목(DB에서 강화됨)."""
    person_index: int
    pose_id: str
    bvh_url: str = Field(..., description="GET으로 1인 BVH 원본을 읽는 경로(단일 소스)")
    view: str
    set_id: Optional[str] = Field(None, description="얽힘 그룹 id. 같은 값끼리 한 장면(상대 위치는 작가)")
    set_role: Optional[str] = Field(None, description="세트 내 역할(A/B 등). solo면 null")
    tags: dict


class ExportOrder(BaseModel):
    """동원한테 보내는 최종 JSON. item 1개 = 1인 BVH 1개(BVH는 다인 미지원).
    CSP 미러링·축 보정은 동원 단계 책임."""
    schema_version: str = "1.0"
    cut_id: str
    source_image: Optional[str] = None
    created_at: str
    items: List[ExportItem]
    notes: List[str] = []
