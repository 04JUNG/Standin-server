"""러프함에 따른 인물별 검색 경로 분기(04J-286).

한 컷 안에서 인물마다 따로 정한다.

    geometric     관절 근거가 충분 → 기존 기하 kNN(동작 변경 없음)
    semantic      관절은 부족하지만 VLM 자세 설명이 구체적 → 시맨틱 검색
    undetermined  관절·의미 근거 모두 부족 → 검색하지 않고 판정 수치만 남긴다
    error         관절이 부족해 설명을 물었지만 VLM 호출 자체가 실패 → 오류로 남긴다
                  ("의미 근거 없음"과 섞지 않는다. 04J-294 reason이 있으면 그대로 싣는다)

이 모듈은 **순수 판정**만 한다. 같은 입력은 항상 같은 경로를 낸다. 검색 실행·VLM 호출은
pipeline이 맡고, 여기서는 판정에 쓴 수치와 기준값을 그대로 기록으로 돌려준다.

불변식과의 관계(CLAUDE.md):
  · 관절 근거 판정은 rtmlib raw score를 문턱으로 쓰지 않는다(불변식 2 — 모델마다 scale이
    다르다). 모델별로 보정된 구조 mask(valid bone/torso bone/coverage)만 문턱으로 쓰고,
    평균 score는 기록용으로만 남긴다.
  · VLM 의미는 여전히 **기하 매칭의 입력이 아니다**(불변식 1). 의미 설명은 기하 근거가
    없는 인물에 한해 별도 시맨틱 경로에서만 쓰인다. geometric 경로의 후보·순위는 그대로다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from .body_language import query_observations, specific_observation_count

ROUTES = ("geometric", "semantic", "undetermined", "error")

# COCO17 중 몸통·팔·다리 12관절(얼굴 5점 제외). 기록용 집계에만 쓴다.
_BODY_JOINTS = tuple(range(5, 17))


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else int(raw)


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class RoutingConfig:
    """판정 기준값. 전부 env로 바꿀 수 있다(기본값 근거: docs/SEARCH_ROUTING.md §3)."""

    # ---- 관절 근거 ----
    # finalize_slot이 내린 coverage 중 기하 검색을 허용하는 등급.
    joint_allowed_coverage: tuple[str, ...] = ("full", "reduced", "sparse", "upper_only")
    # 전신 계열(full/reduced/sparse): 유효 뼈 수와 몸통 뼈 수(최대 4).
    joint_min_valid_bones: int = 6
    joint_min_torso_bones: int = 4
    # 상체 전용(upper_only)은 몸통 뼈가 1개로 기록되므로 별도 기준.
    joint_min_upper_bones: int = 3
    # 인물 박스(스켈레톤 박스, 없으면 VLM 박스) 최소 크기. 둘 다 만족해야 한다.
    joint_min_box_px: float = 64.0
    joint_min_box_ratio: float = 0.05

    # ---- 의미 근거 ----
    # 서기·기다리기처럼 그 자체로는 포즈를 좁히지 못하는 태그. 이것만 있으면 '너무 일반적'.
    semantic_generic_postures: tuple[str, ...] = ("standing",)
    semantic_generic_actions: tuple[str, ...] = (
        "wait", "look", "rest", "listen", "talk", "step", "gesture",
    )
    semantic_min_specific_tags: int = 1
    semantic_min_text_chars: int = 8
    semantic_min_body_detail_chars: int = 6
    # Two distinctive body observations can ground a query when action is unknown.
    semantic_min_body_observations: int = 2
    # VLM이 스스로 low라고 답하면 유효로 보지 않는다.
    semantic_reject_certainty: tuple[str, ...] = ("low",)

    @classmethod
    def from_env(cls) -> "RoutingConfig":
        base = cls()
        return cls(
            joint_allowed_coverage=_env_csv(
                "ROUTING_JOINT_ALLOWED_COVERAGE", base.joint_allowed_coverage),
            joint_min_valid_bones=_env_int(
                "ROUTING_JOINT_MIN_VALID_BONES", base.joint_min_valid_bones),
            joint_min_torso_bones=_env_int(
                "ROUTING_JOINT_MIN_TORSO_BONES", base.joint_min_torso_bones),
            joint_min_upper_bones=_env_int(
                "ROUTING_JOINT_MIN_UPPER_BONES", base.joint_min_upper_bones),
            joint_min_box_px=_env_float("ROUTING_JOINT_MIN_BOX_PX", base.joint_min_box_px),
            joint_min_box_ratio=_env_float(
                "ROUTING_JOINT_MIN_BOX_RATIO", base.joint_min_box_ratio),
            semantic_generic_postures=_env_csv(
                "ROUTING_SEMANTIC_GENERIC_POSTURES", base.semantic_generic_postures),
            semantic_generic_actions=_env_csv(
                "ROUTING_SEMANTIC_GENERIC_ACTIONS", base.semantic_generic_actions),
            semantic_min_specific_tags=_env_int(
                "ROUTING_SEMANTIC_MIN_SPECIFIC_TAGS", base.semantic_min_specific_tags),
            semantic_min_text_chars=_env_int(
                "ROUTING_SEMANTIC_MIN_TEXT_CHARS", base.semantic_min_text_chars),
            semantic_min_body_detail_chars=_env_int(
                "ROUTING_SEMANTIC_MIN_BODY_DETAIL_CHARS",
                base.semantic_min_body_detail_chars),
            semantic_min_body_observations=_env_int(
                "ROUTING_SEMANTIC_MIN_BODY_OBSERVATIONS", base.semantic_min_body_observations),
            semantic_reject_certainty=_env_csv(
                "ROUTING_SEMANTIC_REJECT_CERTAINTY", base.semantic_reject_certainty),
        )

    def joint_thresholds(self) -> dict:
        return {
            "allowed_coverage": list(self.joint_allowed_coverage),
            "min_valid_bones": self.joint_min_valid_bones,
            "min_torso_bones": self.joint_min_torso_bones,
            "min_upper_bones": self.joint_min_upper_bones,
            "min_box_px": self.joint_min_box_px,
            "min_box_ratio": self.joint_min_box_ratio,
        }

    def semantic_thresholds(self) -> dict:
        return {
            "min_specific_tags": self.semantic_min_specific_tags,
            "min_text_chars": self.semantic_min_text_chars,
            "min_body_detail_chars": self.semantic_min_body_detail_chars,
            "min_body_observations": self.semantic_min_body_observations,
            "generic_postures": list(self.semantic_generic_postures),
            "generic_actions": list(self.semantic_generic_actions),
            "reject_certainty": list(self.semantic_reject_certainty),
        }


# ---------------------------------------------------------------- 관절 근거


@dataclass
class JointEvidence:
    sufficient: bool
    # 기존 정책이 검색 자체를 막는 인물(두상·얽힘). 라우터는 기존 경로를 바꾸지 않는다.
    policy_block: Optional[str]
    reasons: list[str]
    metrics: dict
    thresholds: dict

    def to_dict(self) -> dict:
        return {
            "sufficient": self.sufficient,
            "policy_block": self.policy_block,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "thresholds": dict(self.thresholds),
        }


def _box_height(slot) -> Optional[float]:
    box = getattr(slot, "result_box", None)
    if box is None:
        return None
    height = float(box.y2) - float(box.y1)
    return height if np.isfinite(height) else None


def assess_joint_evidence(slot, desc, img_h: int, rcfg: RoutingConfig,
                          policy_block: Optional[str] = None) -> JointEvidence:
    """finalize/rescue가 끝난 슬롯의 관절 근거가 기하 검색에 충분한지 본다."""
    evidence = getattr(slot, "evidence", None)
    coverage = (evidence.coverage_class if evidence is not None
                else getattr(desc, "coverage_class", "insufficient"))
    valid_mask = (np.asarray(evidence.valid_joint_mask, dtype=bool)
                  if evidence is not None else np.zeros(17, dtype=bool))
    raw_scores = (np.asarray(evidence.raw_scores, dtype=float)
                  if evidence is not None else np.zeros(17, dtype=float))
    body_valid = [index for index in _BODY_JOINTS if valid_mask[index]]
    mean_score = (float(np.mean(raw_scores[body_valid])) if body_valid else None)
    box_h = _box_height(slot)
    box_ratio = (box_h / float(img_h)) if box_h is not None and img_h else None
    metrics = {
        "skeleton_state": slot.state,
        "coverage_class": coverage,
        "valid_bone_count": int(evidence.valid_bone_count) if evidence is not None else 0,
        "torso_bone_count": int(evidence.torso_bone_count) if evidence is not None else 0,
        "valid_body_joints": len(body_valid),
        # 기록용. rtmlib score scale이 모델마다 달라 문턱으로 쓰지 않는다.
        "mean_valid_body_score": None if mean_score is None else round(mean_score, 4),
        "box_height_px": None if box_h is None else round(box_h, 1),
        "box_height_ratio": None if box_ratio is None else round(box_ratio, 4),
        "skeleton_source": slot.skeleton_source,
        "humanart_fallback": slot.skeleton_source == "fallback_full_image",
        "crop_retry": int(slot.retry_count),
        "slot_origin": slot.slot_origin,
        "output_scope": (getattr(getattr(desc, "output_scope", None), "detected", None)
                         and desc.output_scope.detected.value),
        "lower_body_visibility_known": bool(slot.lower_body_visibility_known),
        "lower_body_observed": bool(slot.lower_body_observed),
    }
    reasons: list[str] = []
    if slot.skeleton is None or slot.state == "missing":
        reasons.append("skeleton_missing")
    elif slot.state == "invalid":
        reasons.append("skeleton_invalid")
    if coverage not in rcfg.joint_allowed_coverage:
        reasons.append(f"coverage_{coverage}")
    if coverage == "upper_only":
        if metrics["valid_bone_count"] < rcfg.joint_min_upper_bones:
            reasons.append("upper_bones_below_min")
    elif coverage in rcfg.joint_allowed_coverage:
        if metrics["valid_bone_count"] < rcfg.joint_min_valid_bones:
            reasons.append("valid_bones_below_min")
        if metrics["torso_bone_count"] < rcfg.joint_min_torso_bones:
            reasons.append("torso_bones_below_min")
    if box_h is None:
        reasons.append("box_missing")
    else:
        if box_h < rcfg.joint_min_box_px:
            reasons.append("box_px_below_min")
        if box_ratio is not None and box_ratio < rcfg.joint_min_box_ratio:
            reasons.append("box_ratio_below_min")
    return JointEvidence(
        sufficient=not reasons and policy_block is None,
        policy_block=policy_block,
        reasons=reasons,
        metrics=metrics,
        thresholds=rcfg.joint_thresholds(),
    )


# ---------------------------------------------------------------- 의미 근거


@dataclass
class PoseDescription:
    """VLM이 한 인물에 대해 말한 자세 설명. 태그 값은 search-taxonomy-v2 키다."""

    visible: bool = True
    postures: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    activities: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    emotions: list[str] = field(default_factory=list)
    body_details: str = ""
    search_text: str = ""
    certainty: Optional[str] = None
    # 어휘 밖이라 버린 값(기록용).
    dropped: dict = field(default_factory=dict)
    # Qualitative observations, not VLM-generated joint coordinates.
    body_observations: dict = field(default_factory=dict)
    context_evidence: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DescribeOutcome:
    """인물별 설명 호출 결과. status: ok | error | not_requested."""

    status: str
    descriptions: dict = field(default_factory=dict)   # 요청 인덱스 → PoseDescription
    error_reason: Optional[str] = None
    retryable: Optional[bool] = None
    retry_after_seconds: Optional[float] = None
    elapsed_ms: float = 0.0
    provider: Optional[str] = None


@dataclass
class SemanticEvidence:
    # valid | too_generic | empty | not_visible | low_certainty | vlm_error | not_requested
    status: str
    reasons: list[str]
    metrics: dict
    thresholds: dict
    description: Optional[dict] = None
    error_reason: Optional[str] = None

    @property
    def valid(self) -> bool:
        return self.status == "valid"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "thresholds": dict(self.thresholds),
            "description": self.description,
            "error_reason": self.error_reason,
        }


def assess_semantic_evidence(description: Optional[PoseDescription], rcfg: RoutingConfig,
                             *, vlm_error: Optional[str] = None) -> SemanticEvidence:
    """VLM 설명이 해당 인물의 자세를 실제로 좁혀 주는지 본다."""
    thresholds = rcfg.semantic_thresholds()
    if vlm_error is not None:
        return SemanticEvidence("vlm_error", ["vlm_call_failed"], {}, thresholds,
                                error_reason=vlm_error)
    if description is None:
        return SemanticEvidence("empty", ["no_description_for_person"], {}, thresholds)
    generic_postures = set(rcfg.semantic_generic_postures)
    generic_actions = set(rcfg.semantic_generic_actions)
    specific = (
        [value for value in description.postures if value not in generic_postures]
        + [value for value in description.actions if value not in generic_actions]
        + list(description.activities)
    )
    text = (description.search_text or "").strip()
    body = (description.body_details or "").strip()
    observations = query_observations(body, description.body_observations)
    specific_body = specific_observation_count(observations)
    metrics = {
        "specific_tag_count": len(specific),
        "specific_tags": specific,
        "tag_count": sum(len(values) for values in (
            description.postures, description.actions, description.domains,
            description.activities, description.objects, description.intents,
            description.emotions)),
        "search_text_chars": len(text),
        "body_detail_chars": len(body),
        "certainty": description.certainty,
        "visible": description.visible,
        "body_observations": observations,
        "specific_body_observations": specific_body,
    }
    reasons: list[str] = []
    status = "valid"
    if not description.visible:
        status, reasons = "not_visible", ["vlm_says_pose_not_readable"]
    elif metrics["tag_count"] == 0 and not text:
        status, reasons = "empty", ["no_tags_no_text"]
    else:
        if description.certainty in rcfg.semantic_reject_certainty:
            reasons.append("certainty_low")
        if (len(specific) < rcfg.semantic_min_specific_tags
                and specific_body < rcfg.semantic_min_body_observations):
            reasons.append("only_generic_tags")
        if len(text) < rcfg.semantic_min_text_chars:
            reasons.append("search_text_too_short")
        if len(body) < rcfg.semantic_min_body_detail_chars:
            reasons.append("body_details_too_short")
        if reasons:
            status = ("low_certainty" if reasons == ["certainty_low"] else "too_generic")
    return SemanticEvidence(status, reasons, metrics, thresholds,
                            description=description.to_dict())


def not_requested_semantic(rcfg: RoutingConfig, reason: str) -> SemanticEvidence:
    return SemanticEvidence("not_requested", [reason], {}, rcfg.semantic_thresholds())


# ---------------------------------------------------------------- 결정


@dataclass
class RoutingDecision:
    route: str
    reason: str
    joint: JointEvidence
    semantic: SemanticEvidence

    def to_dict(self) -> dict:
        return {
            "route": self.route,
            "reason": self.reason,
            "joint_evidence": self.joint.to_dict(),
            "semantic_evidence": self.semantic.to_dict(),
        }


def decide_route(joint: JointEvidence, semantic: SemanticEvidence) -> RoutingDecision:
    """순수 함수. 우선순위: 기존 정책 차단 > 관절 충분 > VLM 오류 > 의미 유효 > 판단 불가."""
    if joint.policy_block is not None:
        return RoutingDecision("undetermined", f"existing_policy:{joint.policy_block}",
                               joint, semantic)
    if joint.sufficient:
        return RoutingDecision("geometric", "joint_evidence_sufficient", joint, semantic)
    if semantic.status == "vlm_error":
        return RoutingDecision(
            "error", f"vlm_describe_failed:{semantic.error_reason or 'unknown'}",
            joint, semantic)
    if semantic.valid:
        return RoutingDecision("semantic", "joint_insufficient+semantic_valid",
                               joint, semantic)
    return RoutingDecision("undetermined",
                           f"joint_insufficient+semantic_{semantic.status}",
                           joint, semantic)
