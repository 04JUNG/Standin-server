"""Immutable per-joint lineage; scores never certify direct visual observation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Optional
import numpy as np

SOURCES = frozenset({"full_image", "crop_retry", "fallback_full_image", "vlm_visible",
                     "vlm_estimated", "missing", "unknown"})
VISIBILITY = frozenset({"direct_observed", "model_inferred", "occluded_estimated", "absent", "unknown"})


@dataclass(frozen=True)
class JointObservation:
    index: int
    name: str
    source: str
    visibility: str
    original: Optional[tuple[float, float]]
    current: Optional[tuple[float, float]]
    original_score: Optional[float]
    current_score: Optional[float]
    reason: str
    person_id: Optional[int] = None
    version: str = "joint-observation-v1"

    def to_serializable(self):
        return asdict(self)


def _point(point):
    return tuple(float(x) for x in point) if np.isfinite(point).all() else None


def _score(score):
    return float(score) if np.isfinite(score) else None


def initialize_joint_provenance(skeleton, source="full_image", person_id=None):
    """Snapshot once before crop/rescue/repair. Existing immutable lineage is preserved.

    A detector's confidence is model_inferred. Regional policy must additionally
    check ownership/structure and its independently certified observation mask.
    """
    from .schema import COCO17
    if skeleton.joint_provenance is not None:
        return tuple(skeleton.joint_provenance)
    source = source if source in SOURCES else "unknown"
    pts = np.asarray(skeleton.keypoints, float).reshape(17, 2)
    scores = np.asarray(skeleton.scores, float).reshape(17)
    result = []
    for i, name in enumerate(COCO17):
        point, score = _point(pts[i]), _score(scores[i])
        present = point is not None and score is not None and score > 0
        result.append(JointObservation(i, name, source if present else "missing",
                      "model_inferred" if present else "absent", point, point, score, score,
                      "initial_model_observation" if present else "initial_missing", person_id))
    return tuple(result)


def record_joint_update(previous, skeleton, *, source, visibility, reason, changed_indices=None):
    """Preserve original snapshots and unchanged records across an observation update."""
    pts = np.asarray(skeleton.keypoints, float).reshape(17, 2)
    scores = np.asarray(skeleton.scores, float).reshape(17)
    if source not in SOURCES or visibility not in VISIBILITY:
        raise ValueError("invalid_joint_lineage_enum")
    changed = set(range(17)) if changed_indices is None else set(changed_indices)
    return tuple(replace(record, current=_point(pts[record.index]),
                         current_score=_score(scores[record.index]), source=source,
                         visibility=visibility, reason=reason)
                 if record.index in changed else record for record in previous)


def transfer_joint_provenance(old, current, *, source, person_id=None):
    """Record crop/rescue replacement without overwriting the initial observation."""
    if old is None:
        empty = type(current)(np.zeros((17, 2)), np.zeros(17))
        previous = initialize_joint_provenance(empty, source="missing", person_id=person_id)
    else:
        previous = initialize_joint_provenance(old, source="unknown", person_id=person_id)
    pts = np.asarray(current.keypoints, float).reshape(17, 2)
    scores = np.asarray(current.scores, float).reshape(17)
    records = previous
    for i in range(17):
        point, score = _point(pts[i]), _score(scores[i])
        if point == previous[i].current and score == previous[i].current_score:
            continue
        present = point is not None and score is not None and score > 0
        records = record_joint_update(records, current, source=source if present else "missing",
                                      visibility="model_inferred" if present else "absent",
                                      reason="observation_replaced:" + source, changed_indices=(i,))
    return records
