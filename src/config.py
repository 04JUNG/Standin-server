"""전역 설정. 실제 모델/키는 환경변수로 주입하고, 없으면 mock으로 폴백한다."""
from __future__ import annotations

import os
from dataclasses import dataclass

# .env 자동 로드. 아래 dataclass 필드 기본값이 클래스 정의 시점에 os.getenv로
# 평가되므로, 반드시 그 '전에' 환경변수를 채워야 한다.
# python-dotenv가 없으면 조용히 건너뛴다(코어는 env 없이도 mock으로 돈다).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class Config:
    # --- VLM ---  provider: "mock"(오프라인 기본) | "gemini" | "openai"
    vlm_provider: str = os.getenv("VLM_PROVIDER", "mock")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")

    # --- 검출/포즈 ---  "mock" | "rtmlib"
    pose_backend: str = os.getenv("POSE_BACKEND", "mock")

    # --- 검색 ---
    top_n_search: int = 20     # kNN 1차 후보 수(→ rerank 입력)
    top_k_final: int = 5       # 작가에게 보여줄 최종 후보 수
    use_rerank: bool = os.getenv("USE_RERANK", "1") == "1"

    # 같은 view면 거리에 이 값을 곱함(1 미만=우대). '필터'가 아니라 '우선순위'.
    view_priority_weight: float = 0.85

    index_path: str = os.getenv("INDEX_PATH", "data/index.pkl")

    # 검색 Top-1 거리가 이 값보다 크면 '확신 없음' → 저신뢰(얽힘·추출실패·라이브러리 공백) → 폴백.
    # 실데이터로 보정: 좋은 매칭 ~0.15, 앉기-서기 ~0.36, 추출실패 ~0.6+ 관측.
    fallback_distance: float = float(os.getenv("FALLBACK_DISTANCE", "0.45"))
    # 스켈레톤 평균 score가 이 값 미만이면 추출 실패로 간주.
    min_skeleton_score: float = float(os.getenv("MIN_SKELETON_SCORE", "0.2"))


CFG = Config()
