"""Opt-in NDJSON transport around the SAME analyze call, with bounded buffering."""
from contextvars import copy_context
import json
from queue import Queue, Empty, Full
from threading import Event, Thread, Lock

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from src.analysis_progress import capture_progress

MEDIA_TYPE = "application/x-ndjson"


def stream_analysis(run):
    queue = Queue(maxsize=1)
    closed = Event()
    context = copy_context()

    terminal = Event()
    offer_lock = Lock()

    def offer(event):
        with offer_lock:
            if closed.is_set() or terminal.is_set():
                return
            if event["type"] in ("result", "error"):
                terminal.set()
            try:
                queue.put_nowait(event)
            except Full:
                try:
                    queue.get_nowait()
                except Empty:
                    pass
                try:
                    queue.put_nowait(event)
                except Full:
                    pass

    def execute():
        try:
            with capture_progress(lambda p: offer(dict(type="progress", progress=p))):
                result = run()
            offer(dict(type="result", result=result.model_dump(mode="json")))
        except HTTPException as error:
            offer(dict(type="error", status=error.status_code, detail=error.detail))
        except Exception:
            from src.logging_setup import log_error
            log_error("analysis_stream", "analysis failed", exc_info=True)
            # Never stream exception text, provider payloads or private traces.
            offer(dict(type="error", status=500, detail="internal error"))

    def records():
        Thread(target=context.run, args=(execute,), daemon=True,
               name="analysis-progress").start()
        try:
            while True:
                event = queue.get()
                yield json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
                if event["type"] in ("result", "error"):
                    return
        finally:
            closed.set()

    return StreamingResponse(records(), media_type=MEDIA_TYPE,
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
