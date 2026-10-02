"""User-observed facial anchors to neutral head FBX, in local review only."""

import threading
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ..head.fitting import canonical, fit_manual, MANUAL_INDICES, MANUAL_LABELS
from ..head.queries import HeadQueries
from ..head.candidates import HeadCandidates
from ..storage import sha256
from .framing import NativeReference
from .orientation_routes import orientation_router


class FaceInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    query: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    points: list[tuple[float, float]] = Field(min_length=6, max_length=6)


class QueryReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    excluded: bool
    reason: str = Field(default="두상·흉상 대상 없음", max_length=500)


class CandidateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate: str = Field(pattern=r"^[a-f0-9]{24}$")
    target_confirmed: bool = False


def head_router(curation, assets, previews):
    router, lock = APIRouter(), threading.Lock()
    queries = HeadQueries(curation)
    candidates = HeadCandidates(curation, queries)

    def reference(key):
        if key != "reference":
            raise HTTPException(404, "기본 얼굴 모델을 찾을 수 없습니다.")
        try:
            return NativeReference(previews.character, sha256(previews.character))
        except OSError as exc:
            raise HTTPException(503, "기본 캐릭터 파일이 필요합니다.") from exc

    router.include_router(
        orientation_router(
            previews, reference, base_path="/api/head", scopes={"head", "bust"}
        )
    )

    @router.get("/head")
    def page():
        return FileResponse(assets / "head.html")

    @router.get("/api/head")
    def config():
        pose = reference("reference")
        counts = candidates.counts()
        items = [
            {
                **item,
                "face_candidates": (
                    0 if item["excluded"] else counts.get(item["key"], 0)
                ),
            }
            for item in queries.list()
        ]
        return JSONResponse(
            {
                "reference": {"key": "reference", "content_hash": pose.content_hash},
                "labels": MANUAL_LABELS,
                "automatic_enabled": any(item["face_candidates"] for item in items),
                "target_selection_required": True,
                "items": items,
            },
            headers={"Cache-Control": "no-store"},
        )

    @router.get("/api/head/queries/{key}/candidates")
    def face_candidates(key: str):
        try:
            return JSONResponse(
                candidates.public(key), headers={"Cache-Control": "no-store"}
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/api/head/suggest")
    def suggest(spec: CandidateInput, request: Request):
        origin = request.headers.get("origin")
        if request.headers.get("x-pose-review") != "1" or (
            origin and urlparse(origin).netloc != request.url.netloc
        ):
            raise HTTPException(403, "검수 화면에서 실행해 주세요.")
        if spec.target_confirmed is not True:
            raise HTTPException(422, "먼저 그림 속 대상 얼굴을 선택해 주세요.")
        try:
            result = candidates.selected(spec.query, spec.content_hash, spec.candidate)
            return JSONResponse(result, headers={"Cache-Control": "no-store"})
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("/api/head/queries/{key}/image")
    def image(key: str):
        try:
            return FileResponse(
                queries.get(key, allow_excluded=True)["path"],
                headers={"Cache-Control": "no-store"},
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @router.post("/api/head/queries/{key}/review")
    def query_review(key: str, spec: QueryReview, request: Request):
        origin = request.headers.get("origin")
        if request.headers.get("x-pose-review") != "1" or (
            origin and urlparse(origin).netloc != request.url.netloc
        ):
            raise HTTPException(403, "검수 화면에서 실행해 주세요.")
        try:
            return queries.review(key, spec.content_hash, spec.excluded, spec.reason)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/api/head/fit")
    def fit(spec: FaceInput, request: Request):
        origin = request.headers.get("origin")
        if request.headers.get("x-pose-review") != "1" or (
            origin and urlparse(origin).netloc != request.url.netloc
        ):
            raise HTTPException(403, "검수 화면에서 실행해 주세요.")
        if not lock.acquire(blocking=False):
            raise HTTPException(
                429, "각도를 계산 중입니다. 잠시 후 다시 시도해 주세요."
            )
        try:
            query = queries.get(spec.query, spec.content_hash)
            template = canonical(
                curation / "head-direction/models/canonical_face_model.obj",
                MANUAL_INDICES,
            )
            result = fit_manual(spec.points, query["size"], template)
            queries.get(spec.query, spec.content_hash)
            return JSONResponse(
                {**result, "image_sha256": query["content_hash"], "query": spec.query},
                headers={"Cache-Control": "no-store"},
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc
        finally:
            lock.release()

    return router
