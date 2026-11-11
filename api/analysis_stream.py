"""Opt-in NDJSON transport around the SAME analyze call, with bounded buffering."""
from contextvars import copy_context
import json
from threading import Condition, Thread

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from src.analysis_progress import capture_progress

MEDIA_TYPE = "application/x-ndjson"


def stream_analysis(run):
    ready = Condition()
    context = copy_context()
    latest = fallback = None
    closed = terminal = False

    def offer(event):
        nonlocal latest, fallback, terminal
        with ready:
            if closed or terminal:
                return
            if event["type"] == "fallback":
                fallback = event
            else:
                latest = event
                terminal = event["type"] in ("result", "error")
            ready.notify()

    def execute():
        try:
            with capture_progress(lambda p: offer(dict(type="progress", progress=p))):
                result = run(lambda cut: offer(dict(type="fallback", result=cut)))
            offer(dict(type="result", result=result.model_dump(mode="json")))
        except HTTPException as error:
            offer(dict(type="error", status=error.status_code, detail=error.detail))
        except Exception:
            from src.logging_setup import log_error
            log_error("analysis_stream", "analysis failed", exc_info=True)
            # Never stream exception text, provider payloads or private traces.
            offer(dict(type="error", status=500, detail="internal error"))

    def records():
        nonlocal latest, fallback, closed
        Thread(target=context.run, args=(execute,), daemon=True,
               name="analysis-progress").start()
        try:
            while True:
                with ready:
                    ready.wait_for(lambda: latest is not None or fallback is not None)
                    if fallback is not None and not terminal:
                        event, fallback = fallback, None
                    else:
                        event, latest = latest, None
                yield json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
                if event["type"] in ("result", "error"):
                    return
        finally:
            with ready:
                closed = True

    return StreamingResponse(records(), media_type=MEDIA_TYPE,
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
