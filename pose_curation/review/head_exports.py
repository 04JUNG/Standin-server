"""Optional articulated bust parameters; generic pose exports keep their contract."""

from dataclasses import replace
from pydantic import Field, model_validator
from ..head.bust import relative_rotation
from ..orientation import Orientation
from .orientation_routes import ExportSpec


class ReferenceExportSpec(ExportSpec):
    body_yaw: float | None = Field(default=None, ge=-180, le=180)
    body_pitch: float | None = Field(default=None, ge=-90, le=90)
    body_roll: float | None = Field(default=None, ge=-180, le=180)

    @model_validator(mode="after")
    def complete_body(self):
        values = (self.body_yaw, self.body_pitch, self.body_roll)
        if any(v is not None for v in values) and (
            any(v is None for v in values) or self.scope != "bust"
        ):
            raise ValueError("몸통 방향은 흉상에서 세 각도를 함께 지정해야 합니다.")
        return self


def with_body(pose, face, body):
    if body is None:
        return pose
    relative_rotation(face, body)
    return replace(
        pose,
        metadata={
            **pose.metadata,
            "bust_body": {
                name: getattr(body, name) for name in ("yaw", "pitch", "roll")
            },
        },
    )


def configure_reference(pose, spec):
    body = (
        None
        if spec.body_yaw is None
        else Orientation(spec.body_yaw, spec.body_pitch, spec.body_roll)
    )
    return with_body(pose, spec.orientation(), body)
