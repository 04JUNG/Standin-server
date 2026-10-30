"""Per-person description crops with bounded concurrency and one shared deadline."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import copy_context, ContextVar
import math
from threading import BoundedSemaphore, Event

from ..config import CFG
from ..hand_skin_parallel import request_cancelled
from ..logging_setup import log_info
from ..search_routing import DescribeOutcome

VERSION = 'person-context-crop-v1'
active_person_crop = ContextVar('active_person_crop', default=False)
# Process-wide: multiple requests must not multiply the new crop concurrency.
_PERSON_SLOTS = BoundedSemaphore(2)


class _Stop:
    def __init__(self, *events):
        self.events = events

    def is_set(self):
        return any(e is not None and e.is_set() for e in self.events)


def person_crop(image, box, padding=.2):
    """No mirroring or invented coordinates; remap the original target into crop."""
    from PIL import Image
    if not isinstance(image, Image.Image):
        raise ValueError('crop_image_unsupported')
    if box is None or len(box) != 4:
        raise ValueError('crop_box_missing')
    x1, y1, x2, y2 = map(float, box)
    if not all(math.isfinite(v) for v in (x1,y1,x2,y2)) or not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError('crop_box_invalid')
    w,h = image.size
    dx,dy = max(8., (x2-x1)*w*padding), max(8., (y2-y1)*h*padding)
    rect = (max(0,math.floor(x1*w-dx)),max(0,math.floor(y1*h-dy)),
            min(w,math.ceil(x2*w+dx)),min(h,math.ceil(y2*h+dy)))
    left,top,right,bottom = rect
    if right-left < 2 or bottom-top < 2:
        raise ValueError('crop_too_small')
    target = ((x1*w-left)/(right-left),(y1*h-top)/(bottom-top),
              (x2*w-left)/(right-left),(y2*h-top)/(bottom-top))
    return image.crop(rect), target, dict(version=VERSION, source_size=[w,h],
        source_box=list(box), crop_rect=list(rect), target_box=list(target), padding=padding)


def describe_cropped_people(client, image, boxes, deadline):
    started = client._clock()
    closed = Event()
    stop = _Stop(closed, request_cancelled.get())
    outcomes, reports, futures = {}, {}, {}
    workers = max(1, min(2, int(CFG.routing_describe_person_workers)))
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='person-description')

    def failure(reason):
        return DescribeOutcome('error', error_reason=reason, retryable=False,
                               attempts=0, provider='gemini')

    def run(crop, target, index):
        acquired = False
        try:
            while not stop.is_set():
                left = deadline-client._clock()
                if left <= 0:
                    return failure('timeout')
                if _PERSON_SLOTS.acquire(timeout=min(.05,left)):
                    acquired = True
                    break
            if not acquired or stop.is_set():
                return failure('request_cancelled' if stop.is_set() else 'timeout')
            log_info('gemini_person_describe_start', personIndex=index,
                     cropVersion=VERSION, remainingBudgetMs=max(0,int((deadline-client._clock())*1000)))
            token = active_person_crop.set(True)
            try:
                return client._describe_people_bounded(crop, [target], deadline=deadline, stop=stop)
            finally:
                active_person_crop.reset(token)
        finally:
            if acquired:
                _PERSON_SLOTS.release()

    try:
        for index, box in enumerate(boxes):
            if stop.is_set() or client._clock() >= deadline:
                outcomes[index] = failure('request_cancelled' if stop.is_set() else 'timeout')
                continue
            try:
                crop,target,metadata = person_crop(image,box)
            except (ValueError,TypeError,OverflowError):
                outcomes[index] = failure('invalid_person_crop')
                continue
            reports[index] = metadata
            try:
                future = pool.submit(copy_context().run, run, crop, target, index)
            except Exception:
                crop.close()
                raise
            future.add_done_callback(lambda _, owned=crop: owned.close())
            futures[future] = index
        pending = set(futures)
        while pending and not stop.is_set():
            remaining = deadline-client._clock()
            if remaining <= 0:
                break
            done,pending = wait(pending,timeout=min(.05,remaining),return_when=FIRST_COMPLETED)
            for future in done:
                index = futures[future]
                try:
                    outcomes[index] = future.result()
                except Exception as exc:
                    outcomes[index] = failure('describe_exception:'+type(exc).__name__)
        # Late results are never committed to the returned aggregate.
        for future in pending:
            outcomes[futures[future]] = failure('request_cancelled' if stop.is_set() else 'timeout')
        closed.set()
    finally:
        closed.set()
        pool.shutdown(wait=False,cancel_futures=True)

    descriptions, errors = {}, {}
    for index in range(len(boxes)):
        result = outcomes.get(index) or failure('description_missing')
        if result.status == 'ok' and 0 in result.descriptions:
            descriptions[index] = result.descriptions[0]
        else:
            errors[index] = dict(error_reason=result.error_reason or 'description_missing',
                retryable=result.retryable, retry_after_seconds=result.retry_after_seconds,
                attempts=result.attempts)
        reports.setdefault(index,{})['result'] = dict(status=result.status,
            attempts=result.attempts,elapsed_ms=result.elapsed_ms,error_reason=result.error_reason)
    status = 'ok' if not errors else 'partial_error' if descriptions else 'error'
    log_info('gemini_person_describe_summary',status=status,people=len(boxes),
        successfulPeople=len(descriptions),failedPeople=len(errors),workers=workers,
        elapsedMs=round((client._clock()-started)*1000),cropVersion=VERSION)
    return DescribeOutcome(status,descriptions=descriptions,person_errors=errors,
        person_reports=reports,error_reason='person_description_failed' if errors else None,
        retryable=any(e['retryable'] for e in errors.values()) if errors else None,
        attempts=sum(r.attempts for r in outcomes.values()),
        elapsed_ms=(client._clock()-started)*1000,provider='gemini')
