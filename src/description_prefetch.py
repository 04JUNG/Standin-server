"""Opt-in, request-local description prefetch. Never publishes from a worker.

Only a byte-identical image and the entire original ordered slot/box request may
reuse a result. Mismatches use the existing foreground describe path instead.
"""
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, TimeoutError
from contextvars import copy_context
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import io
import os
from threading import BoundedSemaphore, Event
from time import perf_counter

from .config import CFG
from .hand_skin_parallel import request_cancelled
from .vlm import prompts

# No queue of speculative Gemini calls across simultaneous /analyze requests.
_SLOTS = BoundedSemaphore(1)
NOT_PREFETCHED = object()


def _image_key(image):
    from PIL import Image
    if not isinstance(image, Image.Image):
        raise ValueError("unsupported_prefetch_image")
    stream = io.BytesIO()
    image.convert("RGB").save(stream, format="PNG")
    return hashlib.sha256(stream.getvalue()).digest()


def _options(client):
    return (id(client), id(getattr(client, "_client", None)),
            getattr(client, "_model", None), CFG.routing_describe_timeout_ms,
            prompts.POSE_DESCRIBE_VERSION, prompts.POSE_DESCRIBE_SYSTEM,
            id(prompts.pose_describe_prompt))


def _boxes_key(boxes):
    return tuple(None if box is None else tuple(box) for box in boxes)


@dataclass
class _Completed:
    value: object
    error: Exception | None
    elapsed_ms: float


@dataclass
class DescriptionPrefetch:
    status: str
    future: Future | None = None
    image_key: bytes | None = None
    boxes_key: tuple = ()
    slot_keys: tuple = ()
    options: tuple = ()
    wait_seconds: float = 0
    elapsed_ms: float | None = None
    join_wait_ms: float = 0
    cancelled: Event | None = field(default=None, repr=False)
    _closed: bool = field(default=False, repr=False)

    def close(self):
        self._closed = True
        if self.future is not None:
            self.future.cancel()  # Running HTTP calls retain their existing SDK timeout.

    def consume(self, client, image, boxes, slot_keys):
        self._check_cancelled()
        if self.future is None or self._closed:
            return NOT_PREFETCHED
        try:
            matches = (tuple(slot_keys) == self.slot_keys
                       and _boxes_key(boxes) == self.boxes_key
                       and _options(client) == self.options
                       and _image_key(image) == self.image_key)
        except Exception:
            matches = False
        if not matches:
            self.status = "discarded_request_changed"
            self.close()
            return NOT_PREFETCHED
        if self.future.cancelled():
            self.status = "cancelled_before_start"
            return NOT_PREFETCHED
        # A full foreground-sized allowance from join, not a shortened remainder
        # from launch. The SDK's retry and independent-side-review limits are intact.
        joined = perf_counter()
        try:
            deadline = joined + self.wait_seconds
            while True:
                self._check_cancelled()
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    raise TimeoutError()
                try:
                    completed = self.future.result(timeout=min(.05, remaining))
                    break
                except TimeoutError:
                    continue
            self._check_cancelled()
        except Exception:
            if self.status != "cancelled":
                self.status = "join_failed"
            raise
        finally:
            self.join_wait_ms = (perf_counter() - joined) * 1000
        self.elapsed_ms = completed.elapsed_ms
        self.status = "used_error" if completed.error else "used"
        if completed.error is not None:
            raise completed.error  # Same foreground error normalization, no extra retry.
        return deepcopy(completed.value)

    def _check_cancelled(self):
        if self.cancelled is not None and self.cancelled.is_set():
            self.status = "cancelled"
            self.close()
            raise CancelledError()

    def report(self):
        return dict(status=self.status, call_elapsed_ms=(
            round(self.elapsed_ms, 3) if self.elapsed_ms is not None else None),
            join_wait_ms=round(self.join_wait_ms, 3))


def start_description_prefetch(client, image, analysis, img_w, img_h, *, fullbody):
    if not fullbody or os.getenv("PARALLEL_POSE_DESCRIPTION", "0") != "1":
        return None
    # An optional optimization must not break the existing pose path if request
    # preparation is unsupported. Actual provider failures are NOT swallowed here.
    try:
        return _start(client, image, analysis, img_w, img_h)
    except Exception:
        return DescriptionPrefetch("skipped_prepare")


def _start(client, image, analysis, img_w, img_h):
    cancelled = request_cancelled.get()
    if cancelled is not None and cancelled.is_set():
        return DescriptionPrefetch("cancelled", cancelled=cancelled)
    if analysis.num_people <= 0 or analysis.relationship.is_entangled:
        return DescriptionPrefetch("skipped_no_eligible_people")
    # The late request prefers validated VLM boxes. Do not speculate for missing,
    # invalid or cropped fallback boxes, nor alter the late request's batch.
    from .skeleton_extraction import validate_vlm_boxes
    raw = list(analysis.approx_boxes or [])
    if len(raw) != analysis.num_people:
        return DescriptionPrefetch("skipped_boxes")
    valid, reasons = validate_vlm_boxes(raw, img_w, img_h, CFG.slot_min_box_area_ratio)
    if reasons or any(box is None for box in valid):
        return DescriptionPrefetch("skipped_boxes")
    boxes = [(b.x1 / img_w, b.y1 / img_h, b.x2 / img_w, b.y2 / img_h) for b in valid]
    snapshot = None
    try:
        image_key = _image_key(image)
        snapshot = image.copy()
        if _image_key(snapshot) != image_key:
            raise ValueError("image_snapshot_changed")
        options = _options(client)
    except Exception:
        if snapshot is not None:
            snapshot.close()
        return DescriptionPrefetch("skipped_snapshot")
    slots = _SLOTS
    if not slots.acquire(blocking=False):
        snapshot.close()
        return DescriptionPrefetch("skipped_capacity")
    context = copy_context()

    def run():
        started = perf_counter()
        try:
            if cancelled is not None and cancelled.is_set():
                raise CancelledError()
            value = client.describe_people(snapshot, deepcopy(boxes))
            return _Completed(value, None, (perf_counter() - started) * 1000)
        except Exception as error:
            return _Completed(None, error, (perf_counter() - started) * 1000)

    def release(_):
        try:
            snapshot.close()
        finally:
            slots.release()

    pool = None
    try:
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pose-description")
        future = pool.submit(context.run, run)
        future.add_done_callback(release)
    except Exception:
        snapshot.close()
        slots.release()
        return DescriptionPrefetch("skipped_schedule")
    finally:
        if pool is not None:
            pool.shutdown(wait=False)
    return DescriptionPrefetch(
        "pending", future=future, image_key=image_key, boxes_key=_boxes_key(boxes),
        slot_keys=tuple(("vlm", i) for i in range(analysis.num_people)), options=options,
        wait_seconds=2 * max(0.1, CFG.routing_describe_timeout_ms / 1000) + 27,
        cancelled=cancelled,
    )
