"""Loopback review integration; private roughs never leave the local server."""
from collections import OrderedDict
import threading
from typing import Literal
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ..scoped.index import ScopedIndex
from ..scoped.queries import RoughQueries
from ..scoped.matching import match


class MatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+:\d+$")
    scope: Literal["half"] = "half"


def scoped_router(curation, assets, catalog, store, index: ScopedIndex):
    router, lock, cache = APIRouter(), threading.Lock(), OrderedDict()
    queries = RoughQueries(curation)

    @router.get("/scoped")
    def page():
        return FileResponse(assets / "scoped.html")

    @router.get("/api/scoped")
    def library():
        try:
            manifest, _ = index.read()
            return {k: v for k, v in manifest.items() if k not in {"payload", "payload_sha256", "omitted"}}
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("/api/scoped/queries")
    def query_list():
        return JSONResponse({"items": queries.list()}, headers={"Cache-Control": "no-store"})

    @router.get("/api/scoped/queries/{key}")
    def query_detail(key: str):
        try:
            return JSONResponse(queries.public(key), headers={"Cache-Control": "no-store"})
        except (ValueError, OSError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @router.get("/api/scoped/queries/{key}/image")
    def query_image(key: str):
        try:
            return FileResponse(queries.get(key)["path"], headers={"Cache-Control": "no-store"})
        except (ValueError, OSError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @router.post("/api/scoped/match")
    def fit(spec: MatchInput, request: Request):
        origin = request.headers.get("origin")
        if request.headers.get("x-pose-review") != "1" or (origin and urlparse(origin).netloc != request.url.netloc):
            raise HTTPException(403, "검수 화면에서 실행해 주세요.")
        if not lock.acquire(blocking=False):
            raise HTTPException(429, "다른 러프를 비교 중입니다. 잠시 후 다시 시도해 주세요.")
        try:
            manifest, arrays = index.read()
            query = queries.get(spec.query)
            identity = (manifest["revision"], query["key"], query["sha256"])
            if identity not in cache:
                value = match(manifest, arrays, query["keypoints"], query["scores"], image_size=query["size"])
                value["query"] = spec.query
                # Exclusion/change during a long search invalidates the result.
                index.read()
                cache[identity] = value
                if len(cache) > 32:
                    cache.popitem(last=False)
            return JSONResponse(cache[identity], headers={"Cache-Control": "no-store"})
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc
        finally:
            lock.release()

    return router
