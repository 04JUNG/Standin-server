"""Fail-closed assignment of Human-Art candidates to selected VLM slots."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .config import CFG
from .skeleton_extraction import (
    PersonSlot,
    _hungarian,
    analyze_skeleton,
    bbox_iou,
    duplicate_skeleton_distance,
    finalize_slot,
    skeleton_bbox,
    sort_slots_left_to_right,
    torso_center,
)


@dataclass(frozen=True)
class RescueRequest:
    mode: str = "auto"  # auto | all | selected
    person_indices: tuple[int, ...] = ()

    @property
    def manual(self) -> bool:
        return self.mode != "auto"


@dataclass
class RescueReport:
    triggered: bool = False
    trigger: str = "auto"
    unresolved_before: int = 0
    target_count: int = 0
    candidate_count: int = 0
    accepted: int = 0
    would_accept: int = 0
    rejected_reasons: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    model_init_ms: float = 0.0
    error: str | None = None
    stage: str = "off"

    def to_trace(self) -> dict:
        """Machine-readable cut summary copied into targeted person traces."""
        return {
            "triggered": bool(self.triggered),
            "trigger": self.trigger,
            "stage": self.stage,
            "unresolved_before": int(self.unresolved_before),
            "target_count": int(self.target_count),
            "candidate_count": int(self.candidate_count),
            "accepted": int(self.accepted),
            "would_accept": int(self.would_accept),
            "rejected_reasons": list(dict.fromkeys(self.rejected_reasons)),
            "elapsed_ms": round(float(self.elapsed_ms), 3),
            "model_init_ms": round(float(self.model_init_ms), 3),
            "error": self.error,
        }


def parse_rescue_request(value: str | RescueRequest | None) -> RescueRequest:
    if isinstance(value, RescueRequest):
        return value
    raw = "" if value is None else str(value).strip().lower()
    if raw in {"", "auto"}:
        return RescueRequest()
    if raw == "all":
        return RescueRequest(mode="all")
    if len(raw) > 100:
        raise ValueError("rescue selector is too long")
    tokens = raw.split(",")
    if not tokens or any(not token or not token.isdigit() for token in tokens):
        raise ValueError("rescue must be empty, 'all', or comma-separated person indexes")
    indexes = tuple(int(token) for token in tokens)
    if len(indexes) > 20 or len(indexes) != len(set(indexes)):
        raise ValueError("rescue person indexes must be unique and contain at most 20 items")
    if any(index > 99 for index in indexes):
        raise ValueError("rescue person index must be between 0 and 99")
    return RescueRequest(mode="selected", person_indices=indexes)


def _needs_auto_rescue(slot: PersonSlot) -> bool:
    return bool(
        slot.slot_origin == "vlm"
        and slot.vlm_box is not None
        and (
            slot.skeleton is None
            or slot.state in {"missing", "invalid"}
            or slot.evidence is None
            or slot.evidence.coverage_class == "insufficient"
        )
    )


def _manual_targets(slots: list[PersonSlot], request: RescueRequest
                    ) -> tuple[list[PersonSlot], list[str]]:
    ordered = sort_slots_left_to_right([
        slot for slot in slots if slot.slot_origin == "vlm"
    ])
    if request.mode == "all":
        return (
            [slot for slot in ordered if slot.vlm_box is not None],
            [
                f"unusable_person_index:{index}"
                for index, slot in enumerate(ordered)
                if slot.vlm_box is None
            ],
        )
    rejected = []
    targets = [
        ordered[index] for index in request.person_indices
        if index < len(ordered) and ordered[index].vlm_box is not None
    ]
    for index in request.person_indices:
        if index >= len(ordered):
            rejected.append(f"unknown_person_index:{index}")
        elif ordered[index].vlm_box is None:
            rejected.append(f"unusable_person_index:{index}")
    return targets, rejected


def _assignment_cost(slot: PersonSlot, candidate, candidate_box, evidence) -> float:
    if slot.vlm_box is None or candidate_box is None:
        return float("inf")
    center = torso_center(candidate, evidence.valid_joint_mask)
    if center is None:
        center = np.asarray([
            (candidate_box.x1 + candidate_box.x2) * 0.5,
            (candidate_box.y1 + candidate_box.y2) * 0.5,
        ])
    slot_center = np.asarray([
        (slot.vlm_box.x1 + slot.vlm_box.x2) * 0.5,
        (slot.vlm_box.y1 + slot.vlm_box.y2) * 0.5,
    ])
    diagonal = max(1.0, float(np.hypot(
        slot.vlm_box.x2 - slot.vlm_box.x1,
        slot.vlm_box.y2 - slot.vlm_box.y1,
    )))
    center_distance = min(
        2.0, float(np.linalg.norm(center - slot_center) / diagonal)
    )
    return (
        0.65 * (1.0 - bbox_iou(slot.vlm_box, candidate_box))
        + 0.35 * center_distance
    )


def _is_duplicate_of_resolved(candidate, candidate_box,
                              resolved: list[PersonSlot], cfg) -> bool:
    for slot in resolved:
        if slot.skeleton is None or slot.skeleton_box is None:
            continue
        if bbox_iou(candidate_box, slot.skeleton_box) < cfg.pose_fallback_duplicate_iou:
            continue
        # The measured safe policy deliberately uses the primary 0.30 profile
        # for both models.  Raising only Human-Art to 0.35 reduced same-person
        # recall without reducing different-person false positives.
        distance = duplicate_skeleton_distance(
            candidate, slot.skeleton, candidate_box, slot.skeleton_box,
            cfg.skeleton_kpt_threshold,
        )
        if distance <= cfg.pose_fallback_duplicate_distance:
            return True
    return False


def _reject(slot: PersonSlot, reason: str, trace: dict) -> None:
    trace.update({"accepted": False, "rejected_reason": reason})
    slot.rescue_trace = trace
    slot.reasons.append(f"fallback_rejected:{reason}")


def rescue_slots(slots: list[PersonSlot], pose_model, image,
                 img_w: int, img_h: int, cfg=CFG,
                 request: RescueRequest | str | None = None,
                 rescue_context=None) -> RescueReport:
    """Run at most one full-image fallback and mutate only accepted targets."""
    request = parse_rescue_request(request)
    unresolved = [slot for slot in slots if _needs_auto_rescue(slot)]
    report = RescueReport(
        trigger="manual" if request.manual else "auto",
        unresolved_before=len(unresolved),
    )
    if request.manual:
        targets, manual_rejections = _manual_targets(slots, request)
    else:
        targets, manual_rejections = unresolved, []
    report.target_count = len(targets)
    report.rejected_reasons.extend(manual_rejections)
    stage_method = getattr(pose_model, "rescue_stage", None)
    report.stage = stage_method() if callable(stage_method) else "off"
    if not targets or report.stage == "off":
        return report

    report.triggered = True
    started = perf_counter()
    try:
        contextual = getattr(pose_model, "rescue_candidates_with_context", None)
        if rescue_context is not None and callable(contextual):
            candidates = list(contextual(
                image, img_w, img_h, rescue_context
            ))
        else:
            candidates = list(pose_model.rescue_candidates(image, img_w, img_h))
        threshold = float(pose_model.fallback_kpt_threshold())
    except Exception as exc:
        report.error = f"fallback_error:{type(exc).__name__}"
        report.rejected_reasons.append(report.error)
        for slot in targets:
            _reject(slot, report.error, {
                "triggered": True,
                "trigger": report.trigger,
                "stage": report.stage,
                "model_id": "humanart-m",
            })
        report.elapsed_ms = (perf_counter() - started) * 1000.0
        init_method = getattr(pose_model, "fallback_init_ms", None)
        report.model_init_ms = init_method() if callable(init_method) else 0.0
        return report

    report.candidate_count = len(candidates)
    candidate_boxes = [skeleton_bbox(candidate, threshold) for candidate in candidates]
    target_ids = {id(slot) for slot in targets}
    resolved = [
        slot for slot in slots
        if id(slot) not in target_ids
        and slot.skeleton is not None
        and slot.evidence is not None
        and slot.state not in {"missing", "invalid"}
        and slot.evidence.coverage_class != "insufficient"
    ]
    kept_indices = []
    for index, (candidate, box) in enumerate(zip(candidates, candidate_boxes)):
        if box is None:
            report.rejected_reasons.append("invalid_candidate_box")
        elif _is_duplicate_of_resolved(candidate, box, resolved, cfg):
            report.rejected_reasons.append("duplicate_of_resolved")
        else:
            kept_indices.append(index)
    candidates = [candidates[index] for index in kept_indices]
    candidate_boxes = [candidate_boxes[index] for index in kept_indices]

    if not candidates:
        for slot in targets:
            _reject(slot, "no_eligible_candidates", {
                "triggered": True,
                "trigger": report.trigger,
                "stage": report.stage,
                "model_id": "humanart-m",
            })
        report.rejected_reasons.append("no_eligible_candidates")
        report.elapsed_ms = (perf_counter() - started) * 1000.0
        init_method = getattr(pose_model, "fallback_init_ms", None)
        report.model_init_ms = init_method() if callable(init_method) else 0.0
        return report

    evidence_grid = []
    real_cost = np.full((len(targets), len(candidates)), np.inf, dtype=np.float64)
    for row, slot in enumerate(targets):
        peer_boxes = [
            other.vlm_box for other in slots
            if other is not slot and other.vlm_box is not None
        ]
        row_evidence = []
        for column, (candidate, box) in enumerate(zip(candidates, candidate_boxes)):
            evidence = analyze_skeleton(
                candidate, box, threshold, cfg.skeleton_torso_min_box_ratio,
                owner_box=slot.vlm_box, peer_boxes=peer_boxes, cfg=cfg,
            )
            row_evidence.append(evidence)
            real_cost[row, column] = _assignment_cost(
                slot, candidate, box, evidence
            )
        evidence_grid.append(row_evidence)

    padded = np.full(
        (len(targets), len(candidates) + len(targets)),
        float(cfg.slot_assignment_max_cost), dtype=np.float64,
    )
    padded[:, :len(candidates)] = np.where(
        np.isfinite(real_cost), real_cost, cfg.slot_assignment_max_cost + 1.0
    )
    assignment = _hungarian(padded)
    for row, column in enumerate(assignment):
        slot = targets[row]
        trace = {
            "triggered": True,
            "trigger": report.trigger,
            "stage": report.stage,
            "model_id": "humanart-m",
        }
        if column < 0 or column >= len(candidates):
            reason = "unmatched_slot"
            _reject(slot, reason, trace)
            report.rejected_reasons.append(reason)
            continue
        cost = float(real_cost[row, column])
        row_costs = sorted(
            float(value) for value in real_cost[row] if np.isfinite(value)
        )
        row_best = row_costs[0] if row_costs else float("inf")
        margin = (
            row_costs[1] - row_costs[0]
            if len(row_costs) >= 2 else float("inf")
        )
        column_costs = sorted(
            float(value) for value in real_cost[:, column] if np.isfinite(value)
        )
        trace.update({
            "candidate_index": kept_indices[column],
            "assignment_cost": cost,
            "assignment_margin": None if np.isinf(margin) else margin,
        })
        evidence = evidence_grid[row][column]
        if not np.isfinite(cost) or cost > cfg.slot_assignment_max_cost:
            reason = "assignment_cost_exceeded"
        elif cost - row_best > cfg.slot_assignment_ambiguity_margin:
            reason = "assignment_competition"
        elif margin < cfg.slot_assignment_ambiguity_margin:
            reason = "ambiguous_margin"
        elif (len(column_costs) >= 2
              and column_costs[1] - column_costs[0]
              < cfg.slot_assignment_ambiguity_margin):
            reason = "ambiguous_owner"
        elif evidence.coverage_class not in {"full", "reduced"}:
            reason = "coverage_below_reduced"
        elif any("cross_slot" in value for value in evidence.reasons):
            reason = "cross_slot_ownership"
        elif evidence.state not in {"valid", "partial"}:
            reason = "invalid_structure"
        else:
            reason = None
        if reason is not None:
            _reject(slot, reason, trace)
            report.rejected_reasons.append(reason)
            continue

        report.would_accept += 1
        trace.update({
            "accepted": report.stage != "shadow",
            "would_accept": True,
            "coverage_class": evidence.coverage_class,
        })
        if report.stage == "shadow":
            trace["rejected_reason"] = "shadow_not_applied"
            slot.rescue_trace = trace
            slot.reasons.append("fallback_shadow_candidate")
            continue
        slot.skeleton = candidates[column]
        slot.skeleton_box = candidate_boxes[column]
        slot.evidence = evidence
        slot.state = evidence.state
        slot.skeleton_source = "fallback_full_image"
        slot.assignment_cost = cost
        slot.assignment_margin = None if np.isinf(margin) else margin
        slot.reasons.extend(evidence.reasons)
        slot.reasons.append("fallback:humanart-m")
        slot.rescue_trace = trace
        finalize_slot(slot, cfg)
        report.accepted += 1

    report.elapsed_ms = (perf_counter() - started) * 1000.0
    init_method = getattr(pose_model, "fallback_init_ms", None)
    report.model_init_ms = init_method() if callable(init_method) else 0.0
    return report


__all__ = [
    "RescueReport", "RescueRequest", "parse_rescue_request", "rescue_slots",
]
