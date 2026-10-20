"""Review contracts for a chosen face and the exact preview inspected."""

import hashlib
import json
from typing import Literal
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..head.candidates import code_revision
from ..head.fitting import CANONICAL_SHA256, MANUAL_INDICES, canonical, fit_manual
from ..head.reviews import HeadReviewStore
from ..head.bust import validate_region, shoulder_roll
from ..orientation import Orientation
from .head_exports import with_body


class ReviewAngles(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    yaw: float = Field(ge=-180, le=180)
    pitch: float = Field(ge=-90, le=90)
    roll: float = Field(ge=-180, le=180)


class HeadAngleReview(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    reference_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    scope: Literal["head", "bust"]
    candidate: str | None = Field(default=None, pattern=r"^[a-f0-9]{24}$")
    points: list[tuple[float, float]] | None = Field(
        default=None, min_length=6, max_length=6
    )
    region: list[float] | None = Field(default=None, min_length=4, max_length=4)
    angles: ReviewAngles
    body_angles: ReviewAngles | None = None
    shoulder_points: list[tuple[float, float]] | None = Field(
        default=None, min_length=2, max_length=2
    )
    status: Literal["accepted", "hold", "rejected"]
    note: str = Field(default="", max_length=2000)
    expected_revision: int = Field(default=0, ge=0)
    preview_version: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    visual_confirmed: bool = False

    @model_validator(mode="after")
    def one_target(self):
        if sum(v is not None for v in (self.candidate, self.points, self.region)) != 1:
            raise ValueError("얼굴 후보, 6개 기준점, 얼굴 영역 중 하나가 필요합니다.")
        if self.body_angles is not None and self.scope != "bust":
            raise ValueError("몸통 방향은 흉상에서만 지정할 수 있습니다.")
        if self.shoulder_points is not None and self.body_angles is None:
            raise ValueError("어깨 관측은 흉상의 몸통 방향과 함께 저장해야 합니다.")
        return self


def head_review_router(curation, queries, candidates, previews, reference):
    router = APIRouter()
    store = HeadReviewStore(curation / "head-direction/angle-reviews.sqlite")

    def target(key, query, candidate, points, region):
        if region is not None:
            region = validate_region(region, query["size"])
            digest = hashlib.sha256(json.dumps(region).encode()).hexdigest()[:24]
            return "region:" + digest, {
                "region": region,
                "code_revision": code_revision(),
                "source": "manual_region",
                "original_orientation": None,
            }
        if candidate is not None:
            result = candidates.selected(key, query["content_hash"], candidate)
            return "candidate:" + candidate, {
                "candidate": candidate,
                "face_number": next(
                    i + 1
                    for i, face in enumerate(candidates.get(key)["candidates"])
                    if face["id"] == candidate
                ),
                "code_revision": code_revision(),
                "original_orientation": result["orientation"],
                "bbox": result["bbox"],
            }
        template = canonical(
            curation / "head-direction/models/canonical_face_model.obj", MANUAL_INDICES
        )
        result = fit_manual(points, query["size"], template)
        digest = hashlib.sha256(json.dumps(result["points"]).encode()).hexdigest()[:24]
        return "manual:" + digest, {
            "points": result["points"],
            "code_revision": code_revision(),
            "canonical_sha256": CANONICAL_SHA256,
            "original_orientation": result["orientation"],
        }

    @router.get("/api/head/queries/{key}/angle-reviews")
    def list_reviews(key: str):
        try:
            query = queries.get(key, allow_excluded=True)
            rows = store.list(query["content_hash"])
            for row in rows:
                try:
                    queries.get(key, query["content_hash"])
                    pose = reference("reference")
                    if row["reference_hash"] != pose.content_hash:
                        raise ValueError("기본 캐릭터가 변경되었습니다.")
                    if row["provenance"]["code_revision"] != code_revision():
                        raise ValueError("얼굴 추천 코드가 변경되었습니다.")
                    if row["provenance"].get("candidate"):
                        candidates.selected(
                            key, query["content_hash"], row["provenance"]["candidate"]
                        )
                    if row["status"] == "accepted":
                        angles = Orientation(**row["angles"])
                        body = row.get("body_angles")
                        pose = with_body(
                            pose, angles, Orientation(**body) if body else None
                        )
                        current = previews.status(
                            pose, row["scope"], orientation=angles
                        )
                        if (
                            current["status"] != "ready"
                            or current["version"] != row["preview_version"]
                        ):
                            raise ValueError(
                                "저장했던 미리보기를 다시 생성하고 검수해 주세요."
                            )
                    row["current"] = True
                except (ValueError, OSError, RuntimeError) as exc:
                    row["current"] = False
                    row["stale_reason"] = str(exc)
            return JSONResponse({"items": rows}, headers={"Cache-Control": "no-store"})
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/api/head/queries/{key}/angle-reviews")
    def save_review(key: str, spec: HeadAngleReview, request: Request):
        origin = request.headers.get("origin")
        if request.headers.get("x-pose-review") != "1" or (
            origin and urlparse(origin).netloc != request.url.netloc
        ):
            raise HTTPException(403, "검수 화면에서 실행해 주세요.")
        try:
            query = queries.get(key, spec.content_hash)
            pose = reference("reference")
            if pose.content_hash != spec.reference_hash:
                raise ValueError(
                    "기본 캐릭터가 변경되었습니다. 화면을 새로 불러와 주세요."
                )
            identity, provenance = target(
                key, query, spec.candidate, spec.points, spec.region
            )
            angles = Orientation(**spec.angles.model_dump())
            body = (
                Orientation(**spec.body_angles.model_dump())
                if spec.body_angles
                else None
            )
            pose = with_body(pose, angles, body)
            if spec.shoulder_points is not None:
                measured = shoulder_roll(spec.shoulder_points, query["size"], body)
                difference = abs((measured.roll - body.roll + 180) % 360 - 180)
                if difference > 0.05:
                    raise ValueError(
                        "저장할 몸통 기울기와 어깨 관측이 다릅니다. 어깨선을 다시 적용해 주세요."
                    )
            if spec.status == "accepted":
                current = previews.status(pose, spec.scope, orientation=angles)
                if (
                    not spec.visual_confirmed
                    or current["status"] != "ready"
                    or current["version"] != spec.preview_version
                ):
                    raise ValueError(
                        "현재 각도의 FBX 미리보기를 생성하고 확인한 뒤 채택해 주세요."
                    )
            record = {
                "image_sha256": query["content_hash"],
                "target": identity,
                "scope": spec.scope,
                "reference_hash": pose.content_hash,
                "angles": {
                    name: getattr(angles, name) for name in ("yaw", "pitch", "roll")
                },
                "status": spec.status,
                "shoulder_points": spec.shoulder_points,
                "body_angles": (
                    {name: getattr(body, name) for name in ("yaw", "pitch", "roll")}
                    if body
                    else None
                ),
                "note": spec.note,
                "provenance": provenance,
                "preview_version": (
                    spec.preview_version if spec.status == "accepted" else None
                ),
                "visual_confirmed": spec.status == "accepted",
            }
            return store.save(record, spec.expected_revision)
        except (ValueError, OSError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    return router
