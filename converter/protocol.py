"""Converter worker와 API runner가 공유하는 동결 실행 계약."""

from __future__ import annotations

JOB_SCHEMA_VERSION = 3
SOLVER_VERSION = "chain-transport-v3.2.5"
EXPECTED_BLENDER_VERSION = "5.2.0"
EXPECTED_BLENDER_BUILD_HASH = "fbe6228777e7"

RETARGET_SHA256 = "2fca3bd577445989c178aa4ca33e772c595ad2aeb6083f817a3824c93508a579"
ANKLE_POLICY_SHA256 = "79cb19adbc174d729cafbc7497e0862bea880e9370b00581ca6b567b1d80805f"
SOLVER_MANIFEST_SHA256 = "52eae0b0eb6a7023540fdf023ad4321f24c11d439f3ecf9f390b5b72358b24e3"

OUTPUT_MODE = "rigged_rest"
FRAME = 0
APPLY_ROOT_TRANSLATION = False
EMBED_TEXTURES = False

# ── 썸네일 렌더(선택 단계) ──
# anatomical 카메라는 유지하고, preview_style의 부드러운 라이브러리 조명을 쓴다.
# /refine 조정본에도 같은 스타일을 적용하며 렌더 버전으로 예전 결과와 구분한다.
# 렌더는 변환 report와 함께 검증되지만 solver 동결 범위(SHA256SUMS.v325) 밖이다.
THUMBNAIL_RENDERER_VERSION = "fbx-anatomical-soft-v2"
THUMBNAIL_VIEWS = ("front", "three_quarter", "side", "back")
THUMBNAIL_CAMERA_CONVENTION = "anatomical_{view}_from_shoulders_hips"
THUMBNAIL_RENDER_RESOLUTION = 256      # live preview 최종 크기(CPU Cycles 지연 상한)
THUMBNAIL_MIN_RENDER_RESOLUTION = 64
THUMBNAIL_MAX_RENDER_RESOLUTION = 1024
THUMBNAIL_RENDER_SAMPLES = 8           # headless CPU Cycles의 실시간 preview 예산
THUMBNAIL_MAX_RENDER_SAMPLES = 256
# 앞에서부터 시도한다. headless 컨테이너에서 EEVEE(GL)가 못 뜨면 Cycles(CPU)로 넘어간다.
THUMBNAIL_ENGINES = ("BLENDER_EEVEE", "CYCLES")
