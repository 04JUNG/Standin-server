"""Local final-artifact review. Human decisions never change deployment approval."""

from collections import Counter
from functools import lru_cache
import io
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .catalog import ArtifactCatalog
from .. import plausibility
from ..review.store import ReviewStore
from src.bvh import parse_bvh

AUTO_STATUS = {"reject": "rejected", "hold": "hold"}
AUTO_PREFIX = "[자동검수]"


class AutoApply(BaseModel):
    verdicts: list[Literal["reject", "hold"]] = Field(min_length=1)
    dry_run: bool = True


def auto_note(auto):
    labels = ", ".join(
        f"{f['label']} {f['value']}" for f in auto["findings"][:4]
    )
    return f"{AUTO_PREFIX} {labels}"[:2000]


def auto_matches(row, auto, code):
    if auto == "all":
        return True
    if row["auto"] is None:
        return auto == "unchecked"
    fix = row["auto"]["correction"]
    if auto == "flagged":
        ok = row["auto"]["verdict"] in AUTO_STATUS
    elif auto == "fixable":
        ok = bool(fix and fix["usable"] and fix["verdict"] == "pass")
    else:
        ok = row["auto"]["verdict"] == auto
    return ok and (not code or code in row["auto"]["codes"])


class Decision(BaseModel):
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["pending", "accepted", "hold", "rejected"]
    note: str = Field(default="", max_length=2000)


def create_app(base: Path, *, review_path: Path | None = None):
    catalog = ArtifactCatalog(base)
    store = ReviewStore(review_path or base / "human-review/reviews.sqlite")
    app = FastAPI(docs_url=None, redoc_url=None)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"]
    )
    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")
    app.mount("/vendor", StaticFiles(directory=base / "review-vendor"), name="vendor")

    def get(key):
        if key not in catalog.rows:
            raise HTTPException(404, "포즈를 찾을 수 없습니다.")
        return catalog.rows[key]

    def same_origin(request):
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != request.headers.get("host"):
            raise HTTPException(
                403, "다른 사이트에서는 검수 기록을 변경할 수 없습니다."
            )

    @lru_cache(maxsize=64)
    def comparison_png(key):
        from ..plausibility_sheet import comparison

        row = catalog.rows[key]
        joints, frames = parse_bvh(str(catalog.files[(key, "source", "bvh")]))
        before = frames[0]
        result = plausibility.evaluate(joints, before)
        after, after_findings = None, []
        fixed_path = catalog.files.get((key, "corrected", "bvh"))
        if fixed_path:
            after = parse_bvh(str(fixed_path))[1][0]
            after_findings = plausibility.evaluate(joints, after)["findings"]
        names = {"pass": "통과", "hold": "보류 권장", "reject": "제외 권장"}
        header = f"{row['pose_id']} · 자동 검사 {names.get(result['verdict'], result['verdict'])}"
        image = comparison(joints, before, after, result["findings"], after_findings, header)
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        return buffer.getvalue()

    def public(key, reviews):
        row = catalog.public(key)
        row["review"] = reviews.get(
            (key, row["content_hash"]), dict(status="pending", note="")
        )
        return row

    @app.get("/")
    def home():
        return FileResponse(
            static / "index.html", headers={"Cache-Control": "no-store"}
        )

    @app.get("/api/summary")
    def summary():
        reviews = store.all()
        rows = [public(key, reviews) for key in catalog.order]
        return dict(
            total=len(rows),
            scenes=sum(r["scene_reviewed"] for r in rows),
            repaired=sum(r["repaired"] for r in rows),
            statuses=dict(Counter(r["review"]["status"] for r in rows)),
            categories=sorted({r["category"] for r in rows}),
            groups=dict(Counter(r["group"] for r in rows)),
            auto=dict(Counter((r["auto"] or {}).get("verdict", "unchecked") for r in rows)),
            auto_fixable=sum(auto_matches(r, "fixable", "") for r in rows),
            auto_version=catalog.plausibility_version,
            auto_codes={
                code: plausibility.LIMITS[code][0]
                for code in sorted(
                    {c for r in rows if r["auto"] for c in r["auto"]["codes"]}
                    - {"collision"}
                )
            }
            | {"collision": "몸 관통"},
        )

    @app.get("/api/poses")
    def poses(
        q: str = "",
        group: str = "all",
        category: str = "",
        cohort: str = "all",
        status: str = "all",
        auto: str = "all",
        code: str = "",
        offset: int = Query(0, ge=0),
        limit: int = Query(12, ge=1, le=48),
    ):
        reviews = store.all()
        rows = [
            public(key, reviews)
            for key in catalog.order
            if (not q or q.casefold().strip() in catalog.rows[key]["_search"])
            and (group == "all" or catalog.rows[key]["group"] == group)
            and (not category or catalog.rows[key]["category"] == category)
            and (cohort != "scenes" or catalog.rows[key]["scene_reviewed"])
            and (cohort != "repaired" or catalog.rows[key]["repaired"])
            and auto_matches(catalog.rows[key], auto, code)
        ]
        rows = [r for r in rows if status == "all" or r["review"]["status"] == status]
        return dict(total=len(rows), offset=offset, items=rows[offset : offset + limit])

    @app.get("/api/poses/{key}")
    def detail(key: str):
        get(key)
        return public(key, store.all())

    @app.get("/api/assets/{key}/{scope}/{kind}")
    def asset(key: str, scope: str, kind: str):
        get(key)
        path = catalog.files.get((key, scope, kind))
        if not path:
            raise HTTPException(404, "이 포즈에는 해당 범위의 파일이 없습니다.")
        return FileResponse(
            path,
            media_type="model/gltf-binary"
            if kind == "glb"
            else "application/octet-stream",
            filename=path.name,
            content_disposition_type="inline" if kind == "glb" else "attachment",
        )

    @app.get("/api/poses/{key}/plausibility.png")
    def plausibility_sheet(key: str):
        get(key)
        if (key, "source", "bvh") not in catalog.files:
            raise HTTPException(404, "원본 BVH가 없어 자동 검사 그림을 만들 수 없습니다.")
        return Response(comparison_png(key), media_type="image/png")

    @app.post("/api/auto-review/apply")
    def auto_apply(value: AutoApply, request: Request):
        """Copy auto verdicts into untouched human reviews (pending, no note)."""
        same_origin(request)
        reviews = store.all()
        rows = []
        for key in catalog.order:
            row = catalog.rows[key]
            auto = row["auto"]
            if not auto or auto["verdict"] not in value.verdicts:
                continue
            current = reviews.get((key, row["content_hash"]))
            if current and (current["status"] != "pending" or current["note"]):
                continue
            rows.append((key, row["content_hash"], AUTO_STATUS[auto["verdict"]], auto_note(auto)))
        if not value.dry_run:
            store.save_many(rows)
        return dict(count=len(rows), statuses=dict(Counter(r[2] for r in rows)),
                    dry_run=value.dry_run)

    @app.post("/api/auto-review/revert")
    def auto_revert(request: Request, dry_run: bool = True):
        """Return auto-applied decisions (note still starts with the prefix) to pending."""
        same_origin(request)
        rows = [
            (r["pose_key"], r["content_hash"], "pending", "")
            for r in store.all().values()
            if r["note"].startswith(AUTO_PREFIX) and r["status"] in AUTO_STATUS.values()
        ]
        if not dry_run:
            store.save_many(rows)
        return dict(count=len(rows), dry_run=dry_run)

    @app.post("/api/poses/{key}/review")
    def save(key: str, value: Decision, request: Request):
        same_origin(request)
        row = get(key)
        if row["content_hash"] != value.content_hash or not catalog.verify_fbx(key):
            raise HTTPException(409, "검수 파일이 변경되었습니다. 새로고침해 주세요.")
        return store.save(key, value.content_hash, value.status, value.note)

    @app.get("/api/reviews/export")
    def export():
        return dict(artifact="r7-final-fbx", reviews=list(store.all().values()))

    return app
