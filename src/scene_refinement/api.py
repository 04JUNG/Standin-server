"""Private, default-off scene worker. Launch separately from /analyze.

Operator provides all model/library/rig paths. Public requests cannot set them.
Requests require a private bearer token; do not expose this port to clients.
"""

import asyncio, hmac, json, os, logging
from fastapi import FastAPI, Request, HTTPException


def create_app(engine, token):
    if len(token) < 32:
        raise ValueError("scene worker requires a private token (32+ chars)")
    app = FastAPI()
    lock = asyncio.Lock()

    @app.post("/scene/{stage}")
    async def scene(stage: str, request: Request):
        if not hmac.compare_digest(
            request.headers.get("authorization", ""), "Bearer " + token
        ):
            raise HTTPException(401, "unauthorized")
        if stage not in ("plan", "build"):
            raise HTTPException(404)
        if lock.locked():
            raise HTTPException(503, "scene worker busy")
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > 160 * 1024 * 1024:
                raise HTTPException(413, "scene input too large")
        try:
            async with lock:
                return await asyncio.to_thread(
                    engine.prepare if stage == "plan" else engine.build,
                    json.loads(data),
                )
        except (ValueError, KeyError, TypeError, AssertionError) as exc:
            logging.getLogger(__name__).warning(
                "scene rejected: %s: %s", type(exc).__name__, str(exc)[:300]
            )
            raise HTTPException(422, "scene evidence or artifact was rejected")

    return app
