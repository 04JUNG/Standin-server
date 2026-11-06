"""Opt-in private BFF routes, kept separate from legacy /refine contract."""
import asyncio
import hmac
import hashlib
import platform
from functools import lru_cache
from importlib.metadata import version, PackageNotFoundError
import os
import threading
from pathlib import Path

from fastapi import HTTPException, Request
from pydantic import ValidationError

from api.candidate_refine_models import PrepareRequest, PrepareResponse, ReuseRequest, CandidateResult, VERSION
from src.refine_candidates import CandidateError, CandidatePreparer, SourceSnapshot

MAX_BODY = 512 * 1024


@lru_cache(maxsize=1)
def source_runtime_identity():
    """Exact code/runtime evidence even in a local uncommitted checkout."""
    root = Path(__file__).resolve().parents[1]
    paths = sorted(set(root.glob("src/refine*.py")) | {
        root / name for name in ("api/app.py", "api/models.py", "api/candidate_refine.py",
            "api/candidate_refine_models.py", "api/candidate_presentation.py", "src/bvh.py",
            "src/library.py", "src/features.py", "src/config.py", "src/pose_quarantine.py")})
    code = hashlib.sha256()
    for path in paths:
        code.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    libraries = {"python": platform.python_version()}
    for package in ("numpy", "scipy"):
        try:
            libraries[package] = version(package)
        except PackageNotFoundError:
            libraries[package] = "unavailable"
    return {"source_code_sha256": code.hexdigest(), "runtime": libraries}


def install_candidate_routes(app):
    service = None
    service_lock = threading.Lock()

    def authorized(request):
        token = os.getenv("CANDIDATE_REFINE_TOKEN", "")
        if os.getenv("CANDIDATE_REFINE_ENABLED", "0") != "1" or len(token) < 32:
            raise HTTPException(503, {"code": "CANDIDATE_REFINE_DISABLED"})
        if not hmac.compare_digest(request.headers.get("authorization", "").encode(),
                                   ("Bearer " + token).encode()):
            raise HTTPException(401, {"code": "CANDIDATE_REFINE_UNAUTHORIZED"})

    def engine_identity():
        from api import app as runtime
        from src.refine_selector import SELECTOR_VERSION
        from src.refine_projection import VERSION as PROJECTION_VERSION
        return {"adapter_version": "candidate-source-adapter-v1",
                "selector_version": SELECTOR_VERSION, "projection_version": PROJECTION_VERSION,
                "policy_version": "structural-refine-v1", **runtime._refine_capability(),
                **source_runtime_identity()}

    def get_service():
        nonlocal service
        with service_lock:
            if service is None:
                from api import app as runtime

                def resolve(pose_id):
                    if runtime.quarantine_record(pose_id, runtime.CFG) is not None:
                        raise CandidateError("pose_quarantined")
                    meta = runtime.get_pose_meta(runtime.STATE["db_path"], pose_id)
                    if not meta or not meta.get("bvh_path"):
                        raise CandidateError("source_unavailable")
                    path = Path(meta["bvh_path"])
                    if path.stat().st_size > 2 * 1024 * 1024:
                        raise CandidateError("source_too_large")
                    return SourceSnapshot(path.read_bytes(), meta.get("set_id"))

                def solve(req, snapshot, path, deadline):
                    if (req.camera and req.camera.version == "candidate-camera-v2"
                            and not runtime.CFG.refine_v2_enabled):
                        raise CandidateError("search_camera_requires_v2")
                    return runtime._refine_impl(req, source_snapshot=({"set_id": snapshot.set_id}, path),
                                                execution_deadline=deadline, run_shadow=False)

                concurrency = int(os.getenv("CANDIDATE_REFINE_CONCURRENCY", "1"))
                service = CandidatePreparer(resolve, solve, engine_identity, concurrency=concurrency)
            return service

    async def payload(request, model):
        authorized(request)
        raw = bytearray()
        try:
            async with asyncio.timeout(10):
                async for chunk in request.stream():
                    if len(raw) + len(chunk) > MAX_BODY:
                        raise HTTPException(413, {"code": "CANDIDATE_REFINE_INPUT_TOO_LARGE"})
                    raw.extend(chunk)
        except TimeoutError:
            raise HTTPException(408, {"code": "CANDIDATE_REFINE_UPLOAD_TIMEOUT"}) from None
        try:
            return model.model_validate_json(raw)
        except ValidationError:
            # Do not echo private observations, tokens or invalid paths.
            raise HTTPException(422, {"code": "CANDIDATE_REFINE_INVALID_INPUT"}) from None

    @app.get("/internal/candidate-refine/v1/capabilities")
    def capabilities(request: Request):
        authorized(request)
        identity = engine_identity()
        return {"contract_version": VERSION, "available": bool(identity["enabled"]),
                "source_refine": bool(identity["enabled"]), "target_refine": False,
                "automatic_parts": False, "scene_joint_validation": False,
                "accepted_automatic_profiles": [], "quality_scope": "source_only",
                "max_candidates": 10, "max_candidates_per_person": 5,
                "concurrency": get_service().concurrency,
                "durable_jobs": False, "prepared_capacity": get_service().cache_capacity,
                "ttl_seconds": get_service().ttl, "engine": identity}

    @app.post("/internal/candidate-refine/v1/prepare", response_model=PrepareResponse)
    async def prepare(request: Request):
        batch = await payload(request, PrepareRequest)
        # ASGI cancellation does not release running source slots prematurely;
        # ThreadPoolExecutor owns admission through actual completion/deadline.
        return await asyncio.to_thread(get_service().prepare, batch)

    @app.post("/internal/candidate-refine/v1/reuse", response_model=CandidateResult)
    async def reuse(request: Request):
        binding = await payload(request, ReuseRequest)
        try:
            return await asyncio.to_thread(get_service().reuse, binding)
        except (CandidateError, OSError, ValueError):
            raise HTTPException(409, {"code": "CANDIDATE_REFINE_REUSE_INVALID"}) from None
