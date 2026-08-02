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
    # --- 실행 환경 ---  "development"(기본) | "production"
    # production에서는 합성 라이브러리 폴백과 mock 백엔드를 막는다(가짜 결과 서빙 방지).
    app_env: str = os.getenv("APP_ENV", "development")

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

    # --- 라이브러리 프로비저닝(배포) ---
    # 이미지에 데이터를 넣을 수 없어(라이선스·용량) 기동 시 번들을 받아 푼다.
    # 예: s3://standin-assets/pose-library/v1.tar.gz  또는 https://...
    # 비어 있으면 받지 않는다 → 로컬은 data/에 직접 둔 파일을 쓴다.
    pose_library_uri: str = os.getenv("POSE_LIBRARY_URI", "")
    data_dir: str = os.getenv("DATA_DIR", "data")
    deployment_version: str = os.getenv("DEPLOYMENT_VERSION", "development")
    pose_library_version: str = os.getenv("POSE_LIBRARY_VERSION", "v1")

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    # 검색 Top-1 거리가 이 값보다 크면 '확신 없음' → 저신뢰(얽힘·추출실패·라이브러리 공백) → 폴백.
    # 실데이터로 보정: 좋은 매칭 ~0.15, 앉기-서기 ~0.36, 추출실패 ~0.6+ 관측.
    # 검색 거리: "pos"(위치L2·기본) | "angle"(뼈 방향·비율 불변) | "hybrid"
    distance_metric: str = os.getenv("DISTANCE", "pos")
    hybrid_w: float = float(os.getenv("HYBRID_W", "0.7"))   # hybrid에서 각도 비중
    fallback_distance: float = float(os.getenv("FALLBACK_DISTANCE", "0.45"))
    # 스켈레톤 평균 score가 이 값 미만이면 추출 실패로 간주.
    min_skeleton_score: float = float(os.getenv("MIN_SKELETON_SCORE", "0.2"))

    # --- 포즈 미세조정(refine) ---  docs/REFINE_DESIGN.md
    # 설계 원칙: refine은 '좋아지거나, 그대로'다. 아래 값들은 전부 그 보장을 위한 것.
    refine_enabled: bool = os.getenv("REFINE_ENABLED", "1") == "1"   # 시연 중 비상 스위치
    # 손실이 base * 이 비율보다 못 줄면 조정을 버리고 베이스를 반환.
    refine_min_gain: float = float(os.getenv("REFINE_MIN_GAIN", "0.95"))
    # 베이스 손실이 이 값 이하면 '이미 맞음' → 조정하지 않는다.
    # 0.01 ≈ 뼈당 평균 8° 오차. 러프 추출 노이즈보다 작아 조정할 실익이 없다.
    refine_min_loss: float = float(os.getenv("REFINE_MIN_LOSS", "0.01"))
    refine_max_iter: int = int(os.getenv("REFINE_MAX_ITER", "100"))
    # 베이스 정규화 강도. 클수록 덜 움직인다(=더 안전, 덜 맞음).
    refine_lambda: float = float(os.getenv("REFINE_LAMBDA", "0.05"))
    # 채널당 베이스에서 벗어날 수 있는 최대 각도(하드 바운드).
    # ⚠ 이것만으로는 '미세조정'이 보장되지 않는다 — 각도 공간의 제약은 3D 공간의
    #   제약이 아니다. 어깨에서 45°면 손목은 몸통 길이의 0.7배까지 움직인다(실측).
    #   docs/REFINE_DESIGN.md §6-4 참조.
    refine_max_delta_deg: float = float(os.getenv("REFINE_MAX_DELTA_DEG", "45"))

    # refine 후보 사지: "arms"(기본) | "all"
    # 실제로 무엇을 조정할지는 아래 관측 감도 게이트가 컷마다 판단한다.
    # P2(3D 이동량 게이트) 전에는 다리를 풀지 않는 것이 MVP 안전 기본값이다.
    refine_limbs: str = os.getenv("REFINE_LIMBS", "arms")

    # --- 관측 감도 게이트 (docs/REFINE_DESIGN.md §6-5) ---
    # "다리를 켜냐 끄냐"는 잘못된 질문이다. 다리 관측 감도는 컷마다 8배 범위로 흩어진다
    # (전신 컷에서 또렷한 다리 = 팔과 비슷, 허벅지에서 잘린 컷 = 팔의 1/10).
    # 그래서 전역 on/off 대신 **컷마다 사지별로 재서** 안 보이는 사지만 동결한다.
    refine_observability_gate: bool = os.getenv("REFINE_OBS_GATE", "1") == "1"
    # 그 컷에서 가장 잘 보이는 사지 대비 이 비율 미만이면 동결(상대 기준이라
    # 포즈·스케일에 안 휘둘린다). 실측상 위험 구간이 '팔 중앙값의 1/3 미만'이었다.
    refine_min_observability: float = float(os.getenv("REFINE_MIN_OBS", "0.34"))
    # 모든 사지가 다 안 보이는 컷에서 상대 기준만으론 못 막으므로 절대 하한도 둔다.
    refine_min_observability_abs: float = float(os.getenv("REFINE_MIN_OBS_ABS", "0.10"))

    # --- P1a: 축별 관측 감도 정규화 (docs/REFINE_NEXT.md §3 P1a) ---
    # 한 관절의 회전축마다 2D에서 보이는 정도가 다르다. 잘 안 보이는 축은
    # 베이스 정규화를 강화해 전완/손·종아리/발이 공짜로 비틀리지 않게 한다.
    refine_axis_observability: bool = os.getenv("REFINE_AXIS_OBS", "1") == "1"
    # 가장 잘 보이는 축 대비 정규화 강화 배수의 상한. 기존 lambda보다 약하게
    # 만들지는 않고 [1, max] 범위에서 위험한 축만 강화한다.
    refine_axis_lambda_max_mult: float = float(
        os.getenv("REFINE_AXIS_LAMBDA_MAX_MULT", "100")
    )
    # --- P1b: 축 조합 null-space 정규화 (docs/REFINE_NEXT.md §3 P1b) ---
    # P1a는 X/Y/Z 축 하나가 안 보이는 경우를 잡는다. 실제 팔 관절의 일부는
    # 각 축은 따로 보이지만 여러 축을 섞은 방향이 안 보이므로, 그 조합 방향도
    # 야코비안 SVD로 찾아 베이스에 고정한다.
    refine_svd_observability: bool = os.getenv("REFINE_SVD_OBS", "1") == "1"
    refine_svd_lambda_max_mult: float = float(
        os.getenv("REFINE_SVD_LAMBDA_MAX_MULT", "2")
    )
    # --- P2: post-solve 사지별 3D 이동량 게이트 (docs/REFINE_P2.md) ---
    # 사람 눈 평가에서 유용한 큰 조정까지 폐기해 P3 검증 동안 하드 게이트는 보류한다.
    # 0이어도 이동량 진단은 limb_decisions에 계속 남는다.
    refine_move_gate: bool = os.getenv("REFINE_MOVE_GATE", "0") == "1"
    # 몸통 길이로 정규화한 사지 중간관절·말단 평균 / 말단 최대 이동량 상한.
    refine_max_move_mean: float = float(os.getenv("REFINE_MAX_MOVE_MEAN", "0.20"))
    refine_max_move_max: float = float(os.getenv("REFINE_MAX_MOVE_MAX", "0.35"))
    # --- P3a: 베이스 상대 팔-몸통 자기 충돌 게이트 (docs/REFINE_P3.md) ---
    # 실 러프 probe에서 171734의 깊은 손 관통과 124629 정상 Top-5가 분리돼 기본 활성화.
    # P2는 계속 로그 전용이며 P3만 하드 게이트로 사용한다.
    refine_collision_gate: bool = os.getenv("REFINE_COLLISION_GATE", "1") == "1"
    refine_collision_torso_shoulder_scale: float = float(
        os.getenv("REFINE_COLLISION_TORSO_SHOULDER_SCALE", "0.38")
    )
    refine_collision_torso_hip_scale: float = float(
        os.getenv("REFINE_COLLISION_TORSO_HIP_SCALE", "0.45")
    )
    refine_collision_arm_radius: float = float(
        os.getenv("REFINE_COLLISION_ARM_RADIUS", "0.035")
    )
    refine_collision_hand_radius: float = float(
        os.getenv("REFINE_COLLISION_HAND_RADIUS", "0.025")
    )
    refine_collision_samples: int = int(os.getenv("REFINE_COLLISION_SAMPLES", "9"))
    # 손끝 프록시 기준 171734 양성(0.055/0.090)과 124629 최대 음성(0.043)
    # 사이로 고정했다. 표본이 작으므로 실사용 로그를 계속 보정 근거로 남긴다.
    refine_collision_min_depth: float = float(
        os.getenv("REFINE_COLLISION_MIN_DEPTH", "0.05")
    )
    refine_collision_worsen_delta: float = float(
        os.getenv("REFINE_COLLISION_WORSEN_DELTA", "0.01")
    )
    # 팔꿈치·무릎이 이 각도보다 접히면 부자연 → 조정 폐기(베이스 반환).
    refine_min_bend_deg: float = float(os.getenv("REFINE_MIN_BEND_DEG", "20"))
    # 조정본 BVH 저장 위치(캐시). data_dir 기준 상대.
    refine_dir: str = os.getenv("REFINE_DIR", "refined")


CFG = Config()
