"""Private single-process worker; probes never call a paid provider."""

import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import logging
import subprocess

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

MAX_BODY = 160 * 1024 * 1024
LOG = logging.getLogger(__name__)


def create_app(engine=None, token="", *, factory=None):
    if len(token) < 32:
        raise ValueError("scene worker requires a private token (32+ chars)")
    if (engine is None) == (factory is None):
        raise ValueError("provide exactly one engine or factory")
    state = {
        "engine": engine,
        "phase": "ready" if engine else "loading",
        "reason": None,
    }
    busy = False

    async def initialize():
        try:
            state["engine"] = await asyncio.to_thread(factory)
            state["phase"] = "ready"
        except Exception as exc:
            # Do not log credentials, image data, operator paths or model text.
            code = getattr(exc, "code", "SCENE_STARTUP_FAILED")
            if code not in {
                "SCENE_ASSET_INVALID",
                "SCENE_CONFIG_INVALID",
                "SCENE_POLICY_INVALID",
                "SCENE_BLENDER_PREFLIGHT_FAILED",
                "SCENE_MODELS_NOT_READY",
                "SCENE_ENGINE_NOT_READY",
            }:
                code = "SCENE_STARTUP_FAILED"
            LOG.error("scene startup failed: %s (%s)", code, type(exc).__name__)
            state.update(phase="failed", reason=code)

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(initialize()) if factory else None
        try:
            yield
        finally:
            if task:
                await asyncio.shield(task)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/healthz")
    async def health():
        return {"ok": True, "service": "scene-refinement-v4"}

    @app.get("/readyz")
    async def ready():
        ok = state["phase"] == "ready"
        return JSONResponse(
            {
                "ok": ok,
                "phase": state["phase"],
                "busy": busy,
                "reason": state["reason"],
            },
            status_code=200 if ok else 503,
        )

    @app.post("/scene/{stage}")
    async def scene(stage: str, request: Request):
        nonlocal busy
        if not hmac.compare_digest(
            request.headers.get("authorization", "").encode("utf-8"),
            ("Bearer " + token).encode("utf-8"),
        ):
            raise HTTPException(401, "unauthorized")
        if stage not in ("plan", "build"):
            raise HTTPException(404)
        if state["phase"] != "ready":
            raise HTTPException(503, {"code": "SCENE_WORKER_NOT_READY"})
        # No await between admission check and reservation, including body reads.
        # The BFF transport identifies this exact legacy detail as SCENE_BUSY.
        if busy:
            raise HTTPException(503, "scene worker busy", headers={"Retry-After": "5"})
        busy = True
        try:
            data = bytearray()
            async with asyncio.timeout(60):
                async for chunk in request.stream():
                    if len(data) + len(chunk) > MAX_BODY:
                        raise HTTPException(413, {"code": "SCENE_INPUT_TOO_LARGE"})
                    data.extend(chunk)
            payload = json.loads(data)
            if not isinstance(payload, dict):
                raise ValueError("object required")
            target = state["engine"]
            task = asyncio.create_task(
                asyncio.to_thread(
                    target.prepare if stage == "plan" else target.build, payload
                )
            )
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # A BFF timeout must not admit another Blender job while this
                # thread still mutates process-local plan state.
                try:
                    await asyncio.shield(task)
                except Exception:
                    pass
                raise
        except (TimeoutError, subprocess.TimeoutExpired):
            raise HTTPException(504, {"code": "SCENE_WORKER_TIMEOUT"}) from None
        except (ValueError, KeyError, TypeError, AssertionError):
            raise HTTPException(422, {"code": "SCENE_EVIDENCE_REJECTED"}) from None
        except HTTPException:
            raise
        except Exception as exc:
            LOG.warning("scene failed: %s", type(exc).__name__)
            raise HTTPException(502, {"code": "SCENE_WORKER_FAILED"}) from None
        finally:
            busy = False

    return app
