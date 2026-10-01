"""Side-effect-isolated router observer: append audit events, never mutate CutResult."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid

from ..vlm.rough_slots import PersonSemantics, SCHEMA_VERSION, parse_slots, valid_normalized_box
from .rough_router import Observation, POLICY_VERSION, plan_search


def append_event(path, event):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode()
    fd = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        with os.fdopen(fd, 'ab', closefd=False) as stream:
            stream.write(payload)
            stream.flush()
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def plans_from_result(vlm, result):
    parsed = parse_slots(vlm.raw, vlm.num_people)
    descriptors = {}
    extras = []
    for desc in result.descriptors:
        index = desc.quality_trace.get('vlm_person_index')
        if type(index) is int and 0 <= index < vlm.num_people:
            descriptors.setdefault(index, []).append(desc)
        else:
            extras.append({'route': 'unresolved_person', 'reason': 'no_vlm_person_ownership',
                           'execution': 'not_executed_planning_only'})
    plans = []
    for i in range(vlm.num_people):
        people_desc = descriptors.get(i, [])
        desc = people_desc[0] if len(people_desc) == 1 else None
        person = parsed.get(i) or PersonSemantics(
            i, framing='face' if vlm.shot.value == 'face' else 'unknown',
            issues=parsed.issues,
        )
        box = vlm.approx_boxes[i] if i < len(vlm.approx_boxes) else None
        ownership = (len(people_desc) <= 1 and box is not None
                     and box.x2 > box.x1 and box.y2 > box.y1)
        if 'approx_boxes' in vlm.raw:
            raw_boxes = vlm.raw['approx_boxes']
            ownership = ownership and isinstance(raw_boxes, list) and i < len(raw_boxes) and valid_normalized_box(raw_boxes[i])
        if desc is not None:
            ownership = ownership and desc.slot_origin == 'vlm'
            observation = Observation(
                keypoints=desc.skeleton.keypoints if desc.skeleton is not None else None,
                valid_mask=desc.quality_trace.get('search_valid_joint_mask'),
                ownership_valid=ownership,
                state=desc.skeleton_state,
            )
        else:
            observation = Observation(ownership_valid=ownership)
        plan = plan_search(person, observation)
        row = plan.to_dict()
        row['plan_id'] = plan.plan_id
        row['observation_status'] = ('available' if desc is not None else
                                      'not_run_legacy_' + result.route)
        row['slot_issues'] = list(parsed.issues)
        row['capability_status'] = {
            'retrieval': 'not_executed', 'partial_feature_index': 'not_implemented',
            'compose': 'not_implemented', 'camera_validation': 'pending',
        }
        plans.append(row)
    return plans + extras


def _image_hash(image):
    if isinstance(image, (bytes, bytearray, memoryview)):
        data = bytes(image)
    elif isinstance(image, str) and Path(image).is_file():
        data = Path(image).read_bytes()
    elif callable(getattr(image, 'tobytes', None)):
        data = (str(getattr(image, 'size', None)) + str(getattr(image, 'mode', None))).encode() + image.tobytes()
    else:
        # Mock inputs are never advertised as real image observations.
        return None
    return hashlib.sha256(data).hexdigest()


def observe(vlm, result, *, image, path, provider, slots_enabled):
    """Caller also guards import/errors. Audit failure must not replace baseline output."""
    parsed = parse_slots(vlm.raw, vlm.num_people)
    event = {
        'event': 'rough_router_plan', 'event_id': uuid.uuid4().hex,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'policy_version': POLICY_VERSION, 'slot_schema_version': SCHEMA_VERSION,
        'mode': 'shadow', 'slots_prompt_enabled': slots_enabled,
        'provider': provider, 'mock': bool(vlm.raw.get('mock', False)),
        'semantic_snapshot': [p.to_dict() for p in parsed.people],
        'slot_parser_issues': list(parsed.issues),
        'baseline_route': result.route, 'candidates_changed': False,
        'execution': 'plan_only', 'image_sha256': _image_hash(image),
        'plans': plans_from_result(vlm, result),
    }
    if result.rough_semantic:
        event["semantic_execution"] = result.rough_semantic
    append_event(path, event)
    return event
