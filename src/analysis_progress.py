"""Best-effort, request-local cumulative milestones. No model or renderer calls."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
import math
from functools import wraps
from uuid import uuid4

VERSION = "pose-analysis-progress.v1"
STAGES = ("analyzing", "extracting", "understanding", "searching", "refining", "ranking")
_current = ContextVar("analysis_progress", default=None)
_person = ContextVar("analysis_progress_person", default=None)
_slot = ContextVar("analysis_progress_slot", default=None)
_UNSET = object()


class Progress:
    def __init__(self, sink):
        self.sink = sink
        self.lock = Lock()
        self.search_modes = {}
        self.descriptor_modes = {}
        self.focus_id = None
        self.focus_slot = None
        self.state = dict(version=VERSION, sequence=0, stage="analyzing",
                          completedStages=[], skippedStages=[], people=[])

    def emit(self, stage=None, *, completed=(), skipped=(), people=None, person=None, focus=_UNSET, scene=_UNSET):
        try:
            with self.lock:
                s = self.state
                if stage and STAGES.index(stage) >= STAGES.index(s["stage"]):
                    s["stage"] = stage
                for key, values in (("completedStages", completed), ("skippedStages", skipped)):
                    s[key] = [v for v in STAGES if v in set(s[key]) | set(values)]
                # Job-level completion means at least one person actually ran it.
                # A skipped path on another person cannot undo that evidence.
                s["skippedStages"] = [v for v in s["skippedStages"] if v not in s["completedStages"]]
                if scene is not _UNSET:
                    if scene:
                        s["scene"] = deepcopy(scene)
                    else:
                        s.pop("scene", None)
                if focus is not _UNSET:
                    if focus is None:
                        s.pop("focus", None)
                    else:
                        s["focus"] = deepcopy(focus)
                if people is not None:
                    s["people"] = deepcopy(people)
                    if people:
                        s.pop("focus", None)
                if person is not None:
                    index = person["personIndex"]
                    previous = next((p for p in s["people"] if p["personIndex"] == index), None)
                    if previous is None:
                        s["people"].append(deepcopy(person))
                    else:
                        previous.update(deepcopy(person))
                    s["people"].sort(key=lambda p: p["personIndex"])
                    s["activePersonIndex"] = index
                s["sequence"] += 1
                s["updatedAt"] = datetime.now(timezone.utc).isoformat()
                self.sink(deepcopy(s))
        except Exception:
            # Telemetry must never fail the existing analysis.
            pass


@contextmanager
def capture_progress(sink):
    token = _current.set(Progress(sink))
    try:
        yield
    finally:
        _current.reset(token)


@contextmanager
def person_scope(index, slot=None):
    token = _person.set(index)
    slot_token = _slot.set(slot)
    try:
        yield
    finally:
        _slot.reset(slot_token)
        _person.reset(token)


def emit(stage=None, **kwargs):
    progress = _current.get()
    if progress:
        progress.emit(stage, **kwargs)


def person_update(**kwargs):
    index = _person.get()
    if index is not None:
        emit(person=dict(personIndex=index, **kwargs))
    else:
        current = _current.get()
        if (current and current.focus_slot is not None and _slot.get() is current.focus_slot
                and not current.state["people"] and "focus" in current.state):
            emit(focus={**current.state["focus"], **kwargs})


def best_effort(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            return None
    return wrapped


@best_effort
def scene_observations(vlm):
    # Control enums have defaults; only validated first-call declarations are
    # scene observations. Never borrow a representative person's view.
    tags = vlm.stated_tags if isinstance(vlm.stated_tags, dict) else {}
    result = {}
    for name, allowed in (("shot", ("full_half", "bust", "face")),
                          ("view", ("front", "side", "back", "three_quarter"))):
        value = tags.get(name)
        if isinstance(value, str) and value in allowed:
            result[name] = value
    raw = vlm.raw if isinstance(vlm.raw, dict) else {}
    count = raw.get("num_people")
    # _coerce may clamp/coerce/default num_people from boxes. Require the actual
    # original numeric integer to agree with the existing parsed analysis count.
    if (type(count) in (int, float) and 0 <= count <= 20 and math.isfinite(count)
            and int(count) == count and type(vlm.num_people) is int and count == vlm.num_people):
        result["personCount"] = int(count)
    return result or None


def start_search(routing):
    description = ((routing or {}).get("summary") or {}).get("describe") or {}
    understanding = description.get("status") not in (None, "not_requested")
    emit("searching", completed=("understanding",) if understanding else (),
         skipped=() if understanding else ("understanding",))


def finish():
    current = _current.get()
    if current and current.state["stage"] == "ranking":
        emit(completed=("ranking",))


@best_effort
def skeleton_overlay(slot, width, height):
    skel, ev = slot.skeleton, slot.evidence
    if skel is None or ev is None or slot.state in ("invalid", "missing", "suspect"):
        return None
    points = [None] * 17
    # All pose backends and crop recovery already return ORIGINAL image pixels.
    for i in range(5, 17):
        x, y = map(float, skel.keypoints[i])
        if bool(ev.valid_joint_mask[i]) and math.isfinite(x) and math.isfinite(y):
            if 0 <= x <= width and 0 <= y <= height:
                points[i] = [x / width, y / height]
    return dict(format="coco17", coordinateSpace="image-normalized", points=points)


@best_effort
def publish_focus(slots, width, height):
    current = _current.get()
    if current is None or len(slots) <= 1:
        return
    # This observational choice never reorders/modifies slots or assigns an index.
    selected, selected_slot, count = None, None, 0
    for slot in slots:
        skeleton = skeleton_overlay(slot, width, height)
        observed = sum(point is not None for point in skeleton["points"]) if skeleton else 0
        if observed > count:
            selected, selected_slot, count = skeleton, slot, observed
    if selected is None:
        return
    if current.focus_id is None:
        current.focus_id = f"focus_{uuid4().hex}"
    current.focus_slot = selected_slot
    emit("extracting", focus=dict(id=current.focus_id, skeleton=selected))


@best_effort
def observations(description):
    # Exact VLM observation fields only, never search text/context/reasoning.
    if not isinstance(description, dict):
        return {}
    from .body_language import validated_observations
    return validated_observations(description.get("body_observations"))


@best_effort
def publish_focus_observations(slots, semantics):
    current = _current.get()
    if current is None or current.focus_slot is None or current.state["people"]:
        return
    # Both dictionaries and scopes refer to the same live slot object. Never use
    # an output index or a nearby person's description as an association.
    if not any(slot is current.focus_slot for slot in slots):
        return
    semantic = semantics.get(id(current.focus_slot))
    observed = observations(semantic.description if semantic else None)
    if observed and "focus" in current.state:
        emit(focus={**current.state["focus"], "observations": observed})


@best_effort
def candidates(rows, data_dir, source="pose"):
    from .thumbnails import thumbnail_url
    result = []
    for rank, row in enumerate(rows, 1):
        if isinstance(row, dict):
            pose_id, view = row.get("pose_id"), row.get("view", "front")
        else:
            pose_id, view = row.pose_id, row.view.value
        if not isinstance(pose_id, str) or not isinstance(view, str):
            continue
        url = thumbnail_url(data_dir, pose_id, view)
        if url:
            result.append(dict(id=f"semantic::{pose_id}" if source == "semantic" else f"{pose_id}::{view}", rank=rank, thumbnailUrl=url))
    return result


@best_effort
def publish_ranked(processed, width, height, routing, data_dir):
    if _current.get() is None:
        return
    progress_people = []
    for index, (slot, outcome) in enumerate(processed):
        person = dict(personIndex=index)
        overlay = skeleton_overlay(slot, width, height)
        if overlay:
            person["skeleton"] = overlay
        record = outcome.routing or {}
        report = record.get("fullbody_search") or {}
        observed = observations((record.get("semantic_evidence") or {}).get("description"))
        if observed:
            person["observations"] = observed
        mode = _current.get().search_modes.get(id(report))
        if mode is None:
            mode = _current.get().descriptor_modes.get(id(outcome.descriptor))
        if mode:
            person["searchMode"] = mode
        # Dynamic IDs require the existing API asset/camera binding. Publish them
        # from that binding below, never invent a temporary selectable identity.
        preview_rows = [] if report else (outcome.semantic_candidates or outcome.candidates)
        previews = candidates(preview_rows, data_dir, "semantic" if outcome.semantic_candidates else "pose")
        if previews:
            person["candidates"] = previews
        progress_people.append(person)
    ran = any("searchMode" in person for person in progress_people)
    emit("ranking", completed=("searching",) if ran else (),
         skipped=("refining",) if ran else ("searching", "refining"), people=progress_people, focus=None)


@best_effort
def presentation_previews(presentation):
    rows = list(presentation.library_candidates) + list(presentation.review_candidates)
    return [dict(id=row.candidate_id, rank=rank, thumbnailUrl=row.asset.thumbnail_url)
            for rank, row in enumerate(rows, 1) if row.asset.thumbnail_url]


def record_search_mode(report, mode):
    current = _current.get()
    if current:
        current.search_modes[id(report)] = mode
    person_update(searchMode=mode)


def record_descriptor_search(descriptor, mode):
    current = _current.get()
    if current:
        current.descriptor_modes[id(descriptor)] = mode
    person_update(searchMode=mode)
