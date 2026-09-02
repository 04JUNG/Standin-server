"""런타임에 실제로 생성된 백엔드가 배포 정책을 만족하는지 검사한다."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .pose import MockPoseModel
from .vlm.client import MockVLMClient

if TYPE_CHECKING:
    from .pipeline import Pipeline


class MockBackendError(RuntimeError):
    """프로덕션 파이프라인이 mock 백엔드로 초기화됐다."""


def pose_runtime_identity(pipeline: "Pipeline") -> dict:
    """Return stable model identity without trusting an env/backend label."""
    pose = pipeline.pose
    if isinstance(pose, MockPoseModel):
        return {
            "model_id": "mock",
            "build_id": "mock",
            "backend": "mock",
            "adapter": type(pose).__name__,
            "initialized": True,
        }
    runtime_identity = getattr(pose, "runtime_identity", None)
    if callable(runtime_identity):
        identity = dict(runtime_identity())
        identity.setdefault("adapter", type(pose).__name__)
        identity.setdefault("initialized", True)
        return identity
    return {
        "model_id": "current-x",
        "build_id": "rtmlib-performance-runtime-default",
        "backend": "rtmlib",
        "adapter": type(pose).__name__,
        "initialized": True,
    }


def actual_backend_names(
    pipeline: "Pipeline",
    requested_vlm: str,
    requested_pose: str,
) -> tuple[str, str]:
    """설정 문자열이 아니라 생성된 객체를 반영한 백엔드 이름을 반환한다."""
    vlm = "mock" if isinstance(pipeline.vlm, MockVLMClient) else requested_vlm
    pose = "mock" if isinstance(pipeline.pose, MockPoseModel) else requested_pose
    return vlm, pose


def ensure_production_backends(
    pipeline: "Pipeline",
    *,
    is_production: bool,
    requested_vlm: str,
    requested_pose: str,
    requested_pose_variant: str = "current-x",
) -> None:
    """프로덕션에서 팩토리의 조용한 mock 폴백을 포함해 mock 사용을 차단한다."""
    if not is_production:
        return

    actual_vlm, actual_pose = actual_backend_names(
        pipeline,
        requested_vlm,
        requested_pose,
    )
    mocked = []
    if actual_vlm == "mock":
        mocked.append(
            f"VLM_PROVIDER={requested_vlm} "
            f"(actual={type(pipeline.vlm).__name__})"
        )
    if actual_pose == "mock":
        mocked.append(
            f"POSE_BACKEND={requested_pose} "
            f"(actual={type(pipeline.pose).__name__})"
        )

    if mocked:
        raise MockBackendError(
            "프로덕션에서 mock 백엔드로 초기화되었습니다: "
            f"{', '.join(mocked)}. API 키와 런타임 의존성을 확인하세요."
        )

    identity = pose_runtime_identity(pipeline)
    requested_variant = requested_pose_variant.strip().lower()
    if identity.get("model_id") != requested_variant:
        raise MockBackendError(
            "포즈 model identity가 요청과 다릅니다: "
            f"requested={requested_variant}, actual={identity.get('model_id')}, "
            f"adapter={identity.get('adapter')}"
        )
    if requested_variant == "humanart-m":
        if identity.get("license_review") != "approved":
            raise MockBackendError("Human-Art production requires approved license review")
        if identity.get("status") not in {"shadow", "canary", "promoted"}:
            raise MockBackendError(
                "Human-Art production manifest must be shadow/canary/promoted, "
                f"got {identity.get('status')}"
            )
    if requested_variant == "cascade":
        primary = identity.get("primary")
        fallback = identity.get("fallback")
        if not isinstance(primary, dict) or primary.get("model_id") != "current-x":
            raise MockBackendError(
                "cascade production primary must be current-x"
            )
        if not isinstance(fallback, dict) or fallback.get("model_id") != "humanart-m":
            raise MockBackendError(
                "cascade production fallback must be humanart-m"
            )
        if fallback.get("license_review") != "approved":
            raise MockBackendError(
                "cascade Human-Art fallback requires approved license review"
            )
        if fallback.get("status") not in {"shadow", "canary", "promoted"}:
            raise MockBackendError(
                "cascade Human-Art manifest must be shadow/canary/promoted, "
                f"got {fallback.get('status')}"
            )
        if identity.get("canary_stage") not in {
            "shadow", "canary-5", "canary-25", "canary-50", "canary-100",
        }:
            raise MockBackendError("cascade has no approved canary stage")
        if identity.get("fallback_contract_ready") is not True:
            raise MockBackendError("cascade fallback contract is not ready")
