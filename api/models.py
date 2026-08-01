"""
API 응답 모델(Pydantic). 이것이 앱 서버 팀과의 '계약'이다 — /docs(OpenAPI)로 자동 노출된다.
schema.py의 dataclass를 미러링하되, 네트워크 경계에 맞게 bvh_url 등 소비자 친화 필드를 더한다.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class CandidateOut(BaseModel):
    pose_id: str
    view: str
    distance: float
    tags: dict
    rerank_score: Optional[float] = None
    bvh_url: str = Field(..., description="동원 내보내기 팀이 받는 라이브러리 BVH 다운로드 경로")
    thumbnail_url: Optional[str] = Field(None, description="후보 시점 PNG 썸네일의 내부 다운로드 경로")


class PersonOut(BaseModel):
    index: int
    box: Optional[List[float]] = Field(None, description="[x1,y1,x2,y2] 픽셀")
    tags: dict
    candidates: List[CandidateOut]


class CutResultOut(BaseModel):
    route: str = Field(..., description="core | bust | skip")
    count_confidence: str = Field(..., description="high(개수 일치) | low(불일치→폴백) | n/a")
    detector_count: int
    vlm_count: int
    people: List[PersonOut] = []
    notes: List[str] = []


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
