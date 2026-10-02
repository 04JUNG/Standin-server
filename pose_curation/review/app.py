"""Loopback-only review server. No inference models or production mutations."""

from __future__ import annotations

from collections import Counter
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.bvh import parse_bvh, fk
from .catalog import Catalog, Pose
from .store import ReviewStore
from .selection import decision, is_excluded
from .framing import FramedPreviews
from ..qa.checks import inspect_pose
from ..qa import policy

View = Literal["front", "three_quarter", "side", "back"]
Status = Literal["pending", "accepted", "hold", "rejected"]
PartialScope = Literal["half", "bust", "head"]


class ReviewInput(BaseModel):
    status: Status
    note: str = Field(default="", max_length=2000)
    content_hash: str = Field(min_length=64, max_length=64)
    visual_checks: list[str] = Field(default_factory=list, max_length=5)


@lru_cache(maxsize=256)
def skeleton(path: str, content_hash: str) -> dict:
    del content_hash  # cache is bound to the revision
    joints, frames = parse_bvh(path)
    positions = fk(joints, frames[0])
    return {
        "joints": [
            {
                "name": j[0],
                "parent": j[1],
                "end": j[4],
                "position": positions[i].tolist(),
            }
            for i, j in enumerate(joints)
        ],
        "frames": len(frames),
    }


def create_app(data_dir: Path, curation_dir: Path) -> FastAPI:
    catalog = Catalog(data_dir, curation_dir)
    store = ReviewStore(curation_dir / "reviews.sqlite")
    previews = FramedPreviews(curation_dir)

    @asynccontextmanager
    async def lifespan(app):
        yield
        previews.close()

    app = FastAPI(title="Standin Pose Review", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"]
    )
    assets = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=assets), name="static")
    from .coverage import coverage_router

    app.include_router(coverage_router(curation_dir, assets))
    from .scenarios import scenario_router

    app.include_router(scenario_router(catalog, store))

    def get_pose(key: str) -> Pose:
        pose = catalog.get(key)
        if pose is None:
            raise HTTPException(404, "포즈를 찾을 수 없습니다.")
        return pose

    from .orientation_routes import orientation_router
    app.include_router(orientation_router(previews, get_pose))
    from .head_routes import head_router
    app.include_router(head_router(curation_dir, assets, previews))
    from ..scoped.index import ScopedIndex
    from .scoped_routes import scoped_router
    scoped_index = ScopedIndex(curation_dir, catalog, store)
    app.include_router(scoped_router(curation_dir, assets, catalog, store, scoped_index))

    def public(pose: Pose, reviews: dict) -> dict:
        return {
            **pose.public(),
            "review": decision(pose, reviews),
            "excluded": is_excluded(pose, reviews),
        }

    @app.get("/")
    def home():
        return FileResponse(assets / "index.html")

    @app.get("/qa")
    def latest_qa_report():
        report = curation_dir / "qa/latest/report.html"
        if not report.is_file():
            raise HTTPException(
                404, "먼저 python -m pose_curation qa 명령을 실행해 주세요."
            )
        return FileResponse(
            report, media_type="text/html", headers={"Cache-Control": "no-store"}
        )

    @app.get("/api/summary")
    def summary():
        poses, reviews = catalog.all(), store.all()
        excluded = [p for p in poses if is_excluded(p, reviews)]
        poses = [p for p in poses if not is_excluded(p, reviews)]
        batch_counts = Counter(p.batch for p in poses)
        return {
            "total": len(poses),
            "excluded": len(excluded),
            "groups": dict(Counter(p.group for p in poses)),
            "categories": dict(
                Counter(
                    p.metadata.get("category", "")
                    for p in poses
                    if p.metadata.get("category")
                )
            ),
            "reviews": dict(
                Counter(public(p, reviews)["review"]["status"] for p in poses)
            ),
            "batches": [
                {
                    **batch,
                    "candidates": batch["poses"],
                    "poses": batch_counts[batch["id"]],
                }
                for batch in catalog.batches
            ],
            "errors": catalog.errors,
        }

    @app.get("/api/poses")
    def list_poses(
        group: Literal["all", "existing", "new", "excluded"] = "all",
        q: str = "",
        status: Literal["all", "pending", "accepted", "hold", "rejected"] = "all",
        batch: str = "",
        category: str = "",
        library_scope: Literal["all", "half", "bust"] = "all",
        offset: int = Query(0, ge=0),
        limit: int = Query(48, ge=1, le=96),
    ):
        poses, reviews = catalog.all(), store.all()
        query = q.casefold().strip()
        show_excluded = group == "excluded" or status == "rejected"
        representative_keys = None
        if library_scope != "all":
            try:
                manifest, _ = scoped_index.read()
                representative_keys = {manifest["sources"][i]["key"] for i in manifest["scopes"][library_scope]["indices"]}
            except (ValueError, OSError) as exc:
                raise HTTPException(409, str(exc)) from exc
        filtered = [
            p
            for p in poses
            if is_excluded(p, reviews) == show_excluded
            and (representative_keys is None or p.key in representative_keys)
            and (group in {"all", "excluded"} or p.group == group)
            and (not batch or p.batch == batch)
            and (not category or p.metadata.get("category") == category)
            and (
                not query
                or query
                in f"{p.pose_id} {p.metadata.get('style', '')} {p.metadata.get('movement', '')}".casefold()
            )
            and (status == "all" or public(p, reviews)["review"]["status"] == status)
        ]
        return {
            "total": len(filtered),
            "offset": offset,
            "items": [public(p, reviews) for p in filtered[offset : offset + limit]],
        }

    @app.get("/api/poses/{key}")
    def detail(key: str):
        return public(get_pose(key), store.all())

    @app.get("/api/qa/policy")
    def review_policy():
        return policy.manifest()

    @app.get("/api/poses/{key}/qa")
    def pose_qa(key: str):
        pose = get_pose(key)
        if pose.group == "existing":
            return {
                "status": "existing_snapshot",
                "findings": [],
                "visual_checks": policy.VISUAL_CHECKS,
            }
        result = inspect_pose(
            pose, store.all(), bases={p.pose_id: p for p in catalog.all()}
        )
        return {**result, "visual_checks": policy.VISUAL_CHECKS}

    @app.get("/api/poses/{key}/thumbnail")
    def thumbnail(key: str, view: View = "front"):
        path = get_pose(key).thumbnails.get(view)
        if path is None or not path.is_file():
            raise HTTPException(404, "썸네일이 없습니다.")
        return FileResponse(
            path,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.get("/api/poses/{key}/skeleton")
    def get_skeleton(key: str):
        pose = get_pose(key)
        return skeleton(str(pose.bvh), pose.content_hash)

    @app.api_route("/api/poses/{key}/framing", methods=["GET", "POST"])
    def framed_status(key: str, scope: PartialScope, request: Request):
        if request.method == "POST":
            origin = request.headers.get("origin")
            if request.headers.get("x-pose-review") != "1" or (
                origin and urlparse(origin).netloc != request.url.netloc
            ):
                raise HTTPException(403, "검수 화면에서 생성해 주세요.")
        try:
            return previews.status(get_pose(key), scope, start=request.method == "POST")
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/api/poses/{key}/framing/{scope}/{view}")
    def framed_image(key: str, scope: PartialScope, view: View, v: str):
        try:
            path = previews.image(get_pose(key), scope, view, v)
        except (OSError, ValueError) as exc:
            raise HTTPException(503, str(exc)) from exc
        if path is None:
            raise HTTPException(404, "현재 포즈의 부분 미리보기가 준비되지 않았습니다.")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/poses/{key}/bvh")
    def bvh(key: str):
        pose = get_pose(key)
        return FileResponse(
            pose.bvh, filename=pose.bvh.name, media_type="application/octet-stream"
        )

    @app.put("/api/poses/{key}/review")
    def save_review(key: str, value: ReviewInput, request: Request):
        # JSON + a custom header blocks cross-origin form writes. No CORS is enabled.
        origin = request.headers.get("origin")
        if request.headers.get("x-pose-review") != "1" or (
            origin and urlparse(origin).netloc != request.url.netloc
        ):
            raise HTTPException(403, "검수 화면에서 저장해 주세요.")
        pose = get_pose(key)
        if pose.content_hash != value.content_hash:
            raise HTTPException(
                409, "포즈가 변경되었습니다. 새로고침 후 다시 검수해 주세요."
            )
        evidence = None
        if value.status == "accepted" and pose.group == "new":
            result = inspect_pose(
                pose, store.all(), bases={p.pose_id: p for p in catalog.all()}
            )
            if not result["automatic_ready"]:
                raise HTTPException(
                    409,
                    "자동 검사 미완료: "
                    + "; ".join(
                        f["message"]
                        for f in result["findings"]
                        if f["severity"] == "block"
                    ),
                )
            if set(value.visual_checks) != set(policy.VISUAL_CHECKS):
                raise HTTPException(422, "5개 시각 검수 항목을 모두 확인해 주세요.")
            warnings = [
                f["code"] for f in result["findings"] if f["severity"] == "review"
            ]
            if warnings and not value.note.strip():
                raise HTTPException(
                    422,
                    "경고가 있는 자세를 채택하려면 확인한 근거를 검수 메모에 남겨 주세요.",
                )
            evidence = {
                "policy_fingerprint": policy.fingerprint(),
                "bvh_sha256": pose.content_hash,
                "thumbnail_sha256": pose.metadata["thumbnail_versions"],
                "visual_checks": value.visual_checks,
                "resolved_findings": warnings,
            }
        return store.save(
            key, pose.content_hash, value.status, value.note, evidence=evidence
        )

    @app.get("/api/reviews/export")
    def export_reviews():
        reviews = store.all()
        return {
            "schema_version": 1,
            "poses": [
                public(p, reviews)
                for p in catalog.all()
                if (p.key, p.content_hash) in reviews
            ],
        }

    return app
