"""Converter worker와 API runner가 공유하는 동결 실행 계약."""

from __future__ import annotations

JOB_SCHEMA_VERSION = 3
SOLVER_VERSION = "chain-transport-v3.2.5"
EXPECTED_BLENDER_VERSION = "5.2.0"
EXPECTED_BLENDER_BUILD_HASH = "fbe6228777e7"

RETARGET_SHA256 = "be57c8eaf7144994a9015783e244a418d70f57b18cef01218fe850b028334cbd"
ANKLE_POLICY_SHA256 = "79cb19adbc174d729cafbc7497e0862bea880e9370b00581ca6b567b1d80805f"
SOLVER_MANIFEST_SHA256 = "3693d91cc1607e787bdb7997201cd8a78b90e79740d60003c30d2a42536466ae"

OUTPUT_MODE = "rigged_rest"
FRAME = 0
APPLY_ROOT_TRANSLATION = False
EMBED_TEXTURES = False

# ── 썸네일 렌더(선택 단계) ──
# 라이브러리 썸네일(2026-09-03 번들)은 같은 캐릭터를 V3.2.5로 변환한 FBX를
# 어깨·골반에서 유도한 anatomical 카메라로 EEVEE 렌더한 뒤 256px JPEG로 줄인 것이다.
# /refine 조정본 preview도 같은 경로를 타야 후보 썸네일과 같은 그림이 된다.
# 렌더는 변환 report와 함께 검증되지만 solver 동결 범위(SHA256SUMS.v325) 밖이다.
THUMBNAIL_RENDERER_VERSION = "fbx-anatomical-v1"
THUMBNAIL_VIEWS = ("front", "three_quarter", "side", "back")
THUMBNAIL_CAMERA_CONVENTION = "anatomical_{view}_from_shoulders_hips"
THUMBNAIL_RENDER_RESOLUTION = 256      # live preview 최종 크기(CPU Cycles 지연 상한)
THUMBNAIL_MIN_RENDER_RESOLUTION = 64
THUMBNAIL_MAX_RENDER_RESOLUTION = 1024
THUMBNAIL_RENDER_SAMPLES = 8           # headless CPU Cycles의 실시간 preview 예산
THUMBNAIL_MAX_RENDER_SAMPLES = 256
# 앞에서부터 시도한다. headless 컨테이너에서 EEVEE(GL)가 못 뜨면 Cycles(CPU)로 넘어간다.
THUMBNAIL_ENGINES = ("BLENDER_EEVEE", "CYCLES")
