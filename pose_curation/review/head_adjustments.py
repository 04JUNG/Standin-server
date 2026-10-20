"""Manual image observations refine only observable bust degrees of freedom."""

from urllib.parse import urlparse
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from ..head.bust import shoulder_roll
from ..orientation import Orientation
from .head_review_routes import ReviewAngles


class ShoulderInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    query: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    points: list[tuple[float, float]] = Field(min_length=2, max_length=2)
    body_angles: ReviewAngles


def head_adjustment_router(queries):
    router = APIRouter()

    @router.post("/api/head/shoulder-roll")
    def shoulders(spec: ShoulderInput, request: Request):
        origin = request.headers.get("origin")
        if request.headers.get("x-pose-review") != "1" or (
            origin and urlparse(origin).netloc != request.url.netloc
        ):
            raise HTTPException(403, "검수 화면에서 실행해 주세요.")
        try:
            query = queries.get(spec.query, spec.content_hash)
            result = shoulder_roll(
                spec.points, query["size"], Orientation(**spec.body_angles.model_dump())
            )
            return {
                "body_angles": {
                    n: getattr(result, n) for n in ("yaw", "pitch", "roll")
                },
                "source": "manual_shoulders_roll_only",
                "points": spec.points,
                "requires_visual_review": True,
            }
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    return router
