"""런타임에 실제로 생성된 백엔드가 배포 정책을 만족하는지 검사한다."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .pose import MockPoseModel
from .vlm.client import MockVLMClient

if TYPE_CHECKING:
    from .pipeline import Pipeline


class MockBackendError(RuntimeError):
    """프로덕션 파이프라인이 mock 백엔드로 초기화됐다."""


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
