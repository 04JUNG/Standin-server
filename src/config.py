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
    # Gemini HTTP 호출 1회의 상한. google-genai HttpOptions.timeout 단위는 밀리초다.
    #
    # ⚠ 20초는 짧았다. 2026-08-19 프로덕션에서 관측된 Gemini 호출 3건이 전부 실패했고
    #   그중 2건이 20.0s·20.3s로 정확히 이 데드라인에 잘렸다(성공 표본 0건). 이 값을
    #   도입하기 전에는 상한이 없어 느린 호출도 결국 끝났다 — 즉 이 값이 "느리지만
    #   되던 것"을 "무조건 실패"로 바꿨다.
    #
    #   timeout에는 HTTP 상태가 없어 재시도 대상도 아니다(아래 attempts 주석 참고).
    #   그래서 데드라인에 한 번 걸리면 그 분석은 그대로 끝난다.
    gemini_request_timeout_ms: int = int(os.getenv("GEMINI_REQUEST_TIMEOUT_MS", "45000"))
    # attempts는 최초 호출을 포함한다. 429/503만 이 범위 안에서 재시도한다.
    # timeout은 재시도하지 않는다 — 끊긴 호출도 Gemini 쪽에서는 계속 생성 중일 수 있어
    # 재시도가 비용을 두 배로 쓰면서 같은 지연을 다시 겪게 만든다.
    gemini_max_attempts: int = int(os.getenv("GEMINI_MAX_ATTEMPTS", "3"))
    # VLM 단계 전체(재시도 포함)에 쓸 수 있는 시간. BFF의 분석 상한이 120초이고 그 안에서
    # 검출·포즈·검색도 끝나야 하므로, VLM이 예산을 다 먹으면 사용자는 원인을 알 수 없는
    # ANALYSIS_TIMEOUT을 받는다. 남은 예산으로 한 번 더 시도할 수 없으면 재시도를 멈춘다.
    #
    # 이 값이 없으면 timeout을 45초로 올린 순간 429/503 재시도 3회가 135초가 되어
    # BFF 상한을 넘는다.
    gemini_total_budget_seconds: float = float(os.getenv("GEMINI_TOTAL_BUDGET_SECONDS", "75"))
    gemini_retry_base_seconds: float = float(os.getenv("GEMINI_RETRY_BASE_SECONDS", "0.5"))
    gemini_retry_max_seconds: float = float(os.getenv("GEMINI_RETRY_MAX_SECONDS", "2.0"))
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

    # --- 관측성(로그·알림) --- 마스터독스 「관측성 — 로그·모니터링·디스코드 알림」
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    # 장애 알림 웹훅. ⚠ URL 자체가 비밀이다 — URL을 아는 누구나 그 채널에 글을 쓸 수 있다.
    # 배포에서는 Secrets Manager의 standin/discord에서 주입한다.
    # 비어 있으면 알림기는 조용히 no-op이다(로컬은 웹훅 없이 그대로 돈다).
    discord_webhook_alert: str = os.getenv("DISCORD_WEBHOOK_ALERT", "")   # P1
    discord_webhook_warn: str = os.getenv("DISCORD_WEBHOOK_WARN", "")     # P2
    discord_webhook_ops: str = os.getenv("DISCORD_WEBHOOK_OPS", "")       # P3
    # P1에 붙일 멘션(@here 등). 코드에 박지 않는다 — 야간 호출 정책은 팀이 정한다.
    discord_alert_mention: str = os.getenv("DISCORD_ALERT_MENTION", "")
    # 배치 창. 이 시간 안의 알림을 한 메시지로 묶어 웹훅 레이트리밋을 피한다.
    alert_flush_seconds: float = float(os.getenv("ALERT_FLUSH_SECONDS", "10"))
    # 같은 키를 다시 보내지 않는 시간. 그 사이의 재발은 세었다가 "×N"으로 보고한다.
    alert_suppress_seconds: float = float(os.getenv("ALERT_SUPPRESS_SECONDS", "300"))
    # 한 메시지에 담을 임베드 상한. 초과분은 "외 N종"으로 접는다.
    alert_max_per_flush: int = int(os.getenv("ALERT_MAX_PER_FLUSH", "5"))

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    # 검색 Top-1 거리가 이 값보다 크면 '확신 없음' → 저신뢰(얽힘·추출실패·라이브러리 공백) → 폴백.
    # 실데이터로 보정: 좋은 매칭 ~0.15, 앉기-서기 ~0.36, 추출실패 ~0.6+ 관측.
    # 검색 거리: "pos"(위치L2·기본) | "angle"(뼈 방향·비율 불변) | "hybrid"
    distance_metric: str = os.getenv("DISTANCE", "pos")
    hybrid_w: float = float(os.getenv("HYBRID_W", "0.7"))   # hybrid에서 각도 비중
    fallback_distance: float = float(os.getenv("FALLBACK_DISTANCE", "0.45"))
    # mask-aware 거리는 coverage마다 분모가 달라 raw distance를 서로 직접 비교할 수 없다.
    # 아래 값은 metric×coverage별 보정 지점이다. 아직 고정 평가셋으로 보정하지 않은
    # 환경에서는 기존 FALLBACK_DISTANCE를 호환 기본값으로 쓰되, sparse는 별도 정책으로
    # high confidence를 금지한다(docs/SKELETON_EXTRACTION_IMPROVEMENT.md §5-0).
    fallback_pos_full: float = float(os.getenv(
        "FALLBACK_POS_FULL", os.getenv("FALLBACK_DISTANCE", "0.45")))
    fallback_pos_reduced: float = float(os.getenv(
        "FALLBACK_POS_REDUCED", os.getenv("FALLBACK_DISTANCE", "0.45")))
    fallback_angle_full: float = float(os.getenv(
        "FALLBACK_ANGLE_FULL", os.getenv("FALLBACK_DISTANCE", "0.45")))
    fallback_angle_reduced: float = float(os.getenv(
        "FALLBACK_ANGLE_REDUCED", os.getenv("FALLBACK_DISTANCE", "0.45")))
    fallback_hybrid_full: float = float(os.getenv(
        "FALLBACK_HYBRID_FULL", os.getenv("FALLBACK_DISTANCE", "0.45")))
    fallback_hybrid_reduced: float = float(os.getenv(
        "FALLBACK_HYBRID_REDUCED", os.getenv("FALLBACK_DISTANCE", "0.45")))
    # 스켈레톤 평균 score가 이 값 미만이면 추출 실패로 간주.
    min_skeleton_score: float = float(os.getenv("MIN_SKELETON_SCORE", "0.2"))

    # --- 스켈레톤 추출 보완 --- docs/SKELETON_EXTRACTION_IMPROVEMENT.md
    skeleton_kpt_threshold: float = float(os.getenv("SKELETON_KPT_THRESHOLD", "0.3"))
    skeleton_torso_min_box_ratio: float = float(os.getenv(
        "SKELETON_TORSO_MIN_BOX_RATIO", "0.05"))
    # 구조 이상은 웹툰 과장을 허용하도록 넉넉한 상한을 쓰며, hard invalid가 아니라
    # crop/A-B 안정성 진단의 신호로 사용한다.
    skeleton_torso_width_ratio_max: float = float(os.getenv(
        "SKELETON_TORSO_WIDTH_RATIO_MAX", "2.50"))
    skeleton_arm_segment_ratio_max: float = float(os.getenv(
        "SKELETON_ARM_SEGMENT_RATIO_MAX", "1.80"))
    skeleton_leg_segment_ratio_max: float = float(os.getenv(
        "SKELETON_LEG_SEGMENT_RATIO_MAX", "2.50"))
    skeleton_adjacent_segment_ratio_max: float = float(os.getenv(
        "SKELETON_ADJACENT_SEGMENT_RATIO_MAX", "3.50"))
    slot_min_box_area_ratio: float = float(os.getenv("SLOT_MIN_BOX_AREA_RATIO", "0.001"))
    slot_assignment_max_cost: float = float(os.getenv("SLOT_ASSIGNMENT_MAX_COST", "0.85"))
    slot_assignment_ambiguity_margin: float = float(os.getenv(
        "SLOT_ASSIGNMENT_AMBIGUITY_MARGIN", "0.08"))
    slot_duplicate_iou: float = float(os.getenv("SLOT_DUPLICATE_IOU", "0.70"))
    # IoU만으로 중복을 판정하면 포옹처럼 실제 두 사람이 겹친 장면을 지운다.
    # 공통 body 관절의 평균 거리를 박스 대각선으로 정규화해 함께 확인한다.
    slot_duplicate_keypoint_distance: float = float(os.getenv(
        "SLOT_DUPLICATE_KEYPOINT_DISTANCE", "0.08"))
    slot_owner_padding: float = float(os.getenv("SLOT_OWNER_PADDING", "0.15"))
    slot_cross_owner_max_iou: float = float(os.getenv(
        "SLOT_CROSS_OWNER_MAX_IOU", "0.50"))
    slot_provisional_max_iou: float = float(os.getenv("SLOT_PROVISIONAL_MAX_IOU", "0.20"))
    slot_crop_max_per_cut: int = int(os.getenv("SLOT_CROP_MAX_PER_CUT", "2"))
    slot_crop_padding: float = float(os.getenv("SLOT_CROP_PADDING", "0.20"))
    # 음수면 family overlap만 판정하고 Top-1 angle은 진단값으로만 기록한다.
    # 고정 평가셋 보정 뒤 양수로 설정하면 두 조건을 함께 게이트한다.
    slot_stability_top1_angle_max: float = float(os.getenv(
        "SLOT_STABILITY_TOP1_ANGLE_MAX", "-1"))

    def fallback_threshold(self, metric: str, coverage_class: str) -> float | None:
        """metric×coverage confidence 임계값. sparse/insufficient는 high 대상이 아니다."""
        metric = metric.lower()
        coverage_class = coverage_class.lower()
        if coverage_class not in ("full", "reduced"):
            return None
        if metric not in ("pos", "angle", "hybrid"):
            metric = "pos"
        return float(getattr(self, f"fallback_{metric}_{coverage_class}"))

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
    # REFINE_DIR(조정본 로컬 캐시)는 제거됐다. 조정본은 /refine 응답으로만 나가고
    # 보관은 BFF가 한다(docs/REFINE_HANDOFF.md §3 4단계).


CFG = Config()
