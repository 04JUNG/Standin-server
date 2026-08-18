"""Internal semantic pose search runtime.

The runtime keeps three evidence layers separate:

* dense/FTS text retrieval proposes semantic units;
* deterministic PoseCode measurements establish observable pose constraints;
* source action metadata can only produce contextual candidates.

It intentionally does not modify the existing image-to-geometry search path.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Literal

import numpy as np

from .semantic_embedding import OnnxE5Encoder, load_embedding_profile, model_directory
from .semantic_index import SEMANTIC_DB_SCHEMA_VERSION, validate_semantic_index


MatchState = Literal["exact", "violation", "unknown"]


@dataclass(frozen=True)
class ConstraintResult:
    state: MatchState
    margin: float | None
    missing_measurements: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedSemanticQuery:
    raw_query: str
    normalized_query: str
    intent: Literal["observable_constraints", "contextual_retrieval", "clarify"]
    constraint_ids: tuple[str, ...]
    context_key: str | None
    context_expectation: Literal["required", "allowed", "partial", "none"]
    unsupported_concepts: tuple[str, ...]
    clarification_question: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "normalized_query": self.normalized_query,
            "intent": self.intent,
            "constraint_ids": list(self.constraint_ids),
            "constraints": [CONSTRAINTS[item] for item in self.constraint_ids],
            "context_key": self.context_key,
            "context_expectation": self.context_expectation,
            "unsupported_concepts": list(self.unsupported_concepts),
            "clarification_question": self.clarification_question,
            "parser_version": 1,
        }


def _cmp(measurement: str, operator: str, value: float) -> dict[str, Any]:
    return {"op": operator, "measurement": measurement, "value": value}


def _all(*children: dict[str, Any]) -> dict[str, Any]:
    return {"op": "all", "children": list(children)}


def _any(*children: dict[str, Any]) -> dict[str, Any]:
    return {"op": "any", "children": list(children)}


LEFT_WRIST_HEIGHT = "left_wrist_height_from_shoulder_torso_units"
RIGHT_WRIST_HEIGHT = "right_wrist_height_from_shoulder_torso_units"
LEFT_ANKLE_HEIGHT = "left_ankle_height_from_pelvis_torso_units"
RIGHT_ANKLE_HEIGHT = "right_ankle_height_from_pelvis_torso_units"
LEFT_ANKLE_FORWARD = "left_ankle_forward_from_pelvis_torso_units"
RIGHT_ANKLE_FORWARD = "right_ankle_forward_from_pelvis_torso_units"

SEATED_EXPRESSION = _all(
    _cmp("left_knee_flexion_deg", "lt", 110),
    _cmp("right_knee_flexion_deg", "lt", 110),
    _cmp(LEFT_ANKLE_HEIGHT, "gt", -1.35),
    _cmp(RIGHT_ANKLE_HEIGHT, "gt", -1.35),
)


CONSTRAINTS: dict[str, dict[str, Any]] = {
    "torso_upright": {
        "evidence": "observed",
        "expression": _cmp("torso_forward_lean_deg", "abs_lt", 16),
    },
    "torso_forward_deep": {
        "evidence": "observed",
        "expression": _cmp("torso_forward_lean_deg", "gt", 35),
    },
    "torso_backward_deep": {
        "evidence": "observed",
        "expression": _cmp("torso_forward_lean_deg", "lt", -35),
    },
    "feet_wide": {
        "evidence": "observed",
        "expression": _cmp("foot_spacing_torso_units", "gt", 1.35),
    },
    "feet_narrow": {
        "evidence": "observed",
        "expression": _cmp("foot_spacing_torso_units", "lt", 0.45),
    },
    "arms_wide": {
        "evidence": "observed",
        "expression": _cmp("wrist_span_torso_units", "gt", 1.75),
    },
    "both_knees_deep": {
        "evidence": "observed",
        "expression": _all(
            _cmp("left_knee_flexion_deg", "lt", 90),
            _cmp("right_knee_flexion_deg", "lt", 90),
        ),
    },
    "either_knee_deep": {
        "evidence": "observed",
        "expression": {
            "op": "min_lt",
            "measurements": ["left_knee_flexion_deg", "right_knee_flexion_deg"],
            "value": 90,
        },
    },
    "both_knees_extended": {
        "evidence": "observed",
        "expression": _all(
            _cmp("left_knee_flexion_deg", "gt", 155),
            _cmp("right_knee_flexion_deg", "gt", 155),
        ),
    },
    "left_leg_back_raised": {
        "evidence": "observed",
        "side_specific": True,
        "expression": _all(
            _cmp(LEFT_ANKLE_FORWARD, "lt", -0.35),
            _cmp(LEFT_ANKLE_HEIGHT, "gt", -1.20),
        ),
    },
    "both_arms_above_shoulders": {
        "evidence": "observed",
        "expression": _all(
            _cmp(LEFT_WRIST_HEIGHT, "gt", 0.10),
            _cmp(RIGHT_WRIST_HEIGHT, "gt", 0.10),
        ),
    },
    "one_foot_raised": {
        "evidence": "observed",
        "expression": {
            "op": "difference_abs_gt",
            "measurements": [LEFT_ANKLE_HEIGHT, RIGHT_ANKLE_HEIGHT],
            "value": 0.35,
        },
    },
    "hands_near_head": {
        "evidence": "observed",
        "expression": {
            "op": "min_lt",
            "measurements": [
                "left_hand_to_head_torso_units",
                "right_hand_to_head_torso_units",
            ],
            "value": 0.45,
        },
    },
    "both_elbows_deep": {
        "evidence": "observed",
        "expression": _all(
            _cmp("left_elbow_flexion_deg", "lt", 90),
            _cmp("right_elbow_flexion_deg", "lt", 90),
        ),
    },
    "one_arm_forward": {
        "evidence": "observed",
        "expression": {
            "op": "max_gt",
            "measurements": [
                "left_wrist_forward_from_pelvis_torso_units",
                "right_wrist_forward_from_pelvis_torso_units",
            ],
            "value": 0.85,
        },
    },
    "both_elbows_extended": {
        "evidence": "observed",
        "expression": _all(
            _cmp("left_elbow_flexion_deg", "gt", 155),
            _cmp("right_elbow_flexion_deg", "gt", 155),
        ),
    },
    "right_arm_up_left_down": {
        "evidence": "observed",
        "side_specific": True,
        "expression": _all(
            _cmp(RIGHT_WRIST_HEIGHT, "gt", 0.10),
            _cmp(LEFT_WRIST_HEIGHT, "lt", -0.35),
        ),
    },
    "left_arm_up_right_down": {
        "evidence": "observed",
        "side_specific": True,
        "expression": _all(
            _cmp(LEFT_WRIST_HEIGHT, "gt", 0.10),
            _cmp(RIGHT_WRIST_HEIGHT, "lt", -0.35),
        ),
    },
    "left_leg_only_back": {
        "evidence": "observed",
        "side_specific": True,
        "expression": _all(
            _cmp(LEFT_ANKLE_FORWARD, "lt", -0.45),
            _cmp(RIGHT_ANKLE_FORWARD, "gt", -0.10),
        ),
    },
    "right_hand_on_hip_only": {
        "evidence": "observed",
        "side_specific": True,
        "expression": _all(
            _cmp("right_hand_to_hip_torso_units", "lt", 0.30),
            _cmp("left_hand_to_hip_torso_units", "gt", 0.60),
        ),
    },
    "torso_lean_left": {
        "evidence": "observed",
        "side_specific": True,
        "expression": _cmp("torso_lateral_lean_deg", "lt", -25),
    },
    "exactly_one_arm_above": {
        "evidence": "observed",
        "expression": {
            "op": "xor",
            "children": [
                _cmp(LEFT_WRIST_HEIGHT, "gt", 0.10),
                _cmp(RIGHT_WRIST_HEIGHT, "gt", 0.10),
            ],
        },
    },
    "not_seated_like": {
        "evidence": "observed",
        "expression": {"op": "not", "child": SEATED_EXPRESSION},
    },
    "both_arms_not_raised": {
        "evidence": "observed",
        "expression": _all(
            _cmp(LEFT_WRIST_HEIGHT, "lte", 0.10),
            _cmp(RIGHT_WRIST_HEIGHT, "lte", 0.10),
        ),
    },
    "hands_far_from_head": {
        "evidence": "observed",
        "expression": _all(
            _cmp("left_hand_to_head_torso_units", "gt", 1.20),
            _cmp("right_hand_to_head_torso_units", "gt", 1.20),
        ),
    },
    "seated_like": {"evidence": "observed_heuristic", "expression": SEATED_EXPRESSION},
    "crouched_low": {
        "evidence": "observed_heuristic",
        "expression": _all(
            _cmp("left_knee_flexion_deg", "lt", 90),
            _cmp("right_knee_flexion_deg", "lt", 90),
            _cmp(LEFT_ANKLE_HEIGHT, "gt", -1.20),
            _cmp(RIGHT_ANKLE_HEIGHT, "gt", -1.20),
        ),
    },
    "torso_near_horizontal": {
        "evidence": "observed_heuristic",
        "expression": {
            "op": "hypot_gt",
            "measurements": ["torso_forward_lean_deg", "torso_lateral_lean_deg"],
            "value": 60,
        },
    },
    "stride_front_back": {
        "evidence": "observed_heuristic",
        "expression": _any(
            _all(
                _cmp(LEFT_ANKLE_FORWARD, "gt", 0.30),
                _cmp(RIGHT_ANKLE_FORWARD, "lt", -0.30),
            ),
            _all(
                _cmp(RIGHT_ANKLE_FORWARD, "gt", 0.30),
                _cmp(LEFT_ANKLE_FORWARD, "lt", -0.30),
            ),
        ),
    },
}


def _missing(measurements: dict[str, float], keys: list[str]) -> tuple[str, ...]:
    return tuple(sorted(key for key in keys if key not in measurements))


def evaluate_expression(
    expression: dict[str, Any], measurements: dict[str, float]
) -> ConstraintResult:
    """Evaluate an expression without collapsing missing data into false."""
    op = expression["op"]
    if op in {"lt", "lte", "gt", "gte", "abs_lt", "abs_gt"}:
        key = expression["measurement"]
        if key not in measurements:
            return ConstraintResult("unknown", None, (key,))
        actual = float(measurements[key])
        threshold = float(expression["value"])
        if op in {"lt", "lte"}:
            margin = threshold - actual
            exact = actual < threshold if op == "lt" else actual <= threshold
        elif op in {"gt", "gte"}:
            margin = actual - threshold
            exact = actual > threshold if op == "gt" else actual >= threshold
        elif op == "abs_lt":
            margin = threshold - abs(actual)
            exact = abs(actual) < threshold
        else:
            margin = abs(actual) - threshold
            exact = abs(actual) > threshold
        return ConstraintResult("exact" if exact else "violation", margin)

    if op in {"min_lt", "max_gt", "difference_abs_gt", "hypot_gt"}:
        keys = list(expression["measurements"])
        missing = _missing(measurements, keys)
        if missing:
            return ConstraintResult("unknown", None, missing)
        values = [float(measurements[key]) for key in keys]
        threshold = float(expression["value"])
        if op == "min_lt":
            actual = min(values)
            margin = threshold - actual
            exact = actual < threshold
        elif op == "max_gt":
            actual = max(values)
            margin = actual - threshold
            exact = actual > threshold
        elif op == "difference_abs_gt":
            actual = abs(values[0] - values[1])
            margin = actual - threshold
            exact = actual > threshold
        else:
            actual = math.hypot(*values)
            margin = actual - threshold
            exact = actual > threshold
        return ConstraintResult("exact" if exact else "violation", margin)

    if op == "not":
        child = evaluate_expression(expression["child"], measurements)
        state: MatchState = {
            "exact": "violation",
            "violation": "exact",
            "unknown": "unknown",
        }[child.state]
        return ConstraintResult(
            state,
            None if child.margin is None else -child.margin,
            child.missing_measurements,
        )

    children = [evaluate_expression(child, measurements) for child in expression["children"]]
    missing = tuple(sorted({key for child in children for key in child.missing_measurements}))
    known_margins = [child.margin for child in children if child.margin is not None]
    if op == "all":
        if any(child.state == "violation" for child in children):
            state = "violation"
        elif any(child.state == "unknown" for child in children):
            state = "unknown"
        else:
            state = "exact"
        margin = min(known_margins) if known_margins else None
    elif op == "any":
        if any(child.state == "exact" for child in children):
            state = "exact"
        elif any(child.state == "unknown" for child in children):
            state = "unknown"
        else:
            state = "violation"
        margin = max(known_margins) if known_margins else None
    elif op == "xor":
        exact_count = sum(child.state == "exact" for child in children)
        if any(child.state == "unknown" for child in children):
            state = "unknown"
        else:
            state = "exact" if exact_count == 1 else "violation"
        if len(known_margins) == len(children):
            margin = min(abs(value) for value in known_margins)
            if state == "violation":
                margin = -margin
        else:
            margin = None
    else:
        raise ValueError(f"unknown constraint operator: {op}")
    return ConstraintResult(state, margin, missing)


def evaluate_constraints(
    constraint_ids: tuple[str, ...], measurements: dict[str, float]
) -> tuple[MatchState, float | None, list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    for constraint_id in constraint_ids:
        result = evaluate_expression(CONSTRAINTS[constraint_id]["expression"], measurements)
        results.append(
            {
                "constraint_id": constraint_id,
                "state": result.state,
                "margin": result.margin,
                "missing_measurements": list(result.missing_measurements),
            }
        )
    if any(row["state"] == "violation" for row in results):
        state: MatchState = "violation"
    elif any(row["state"] == "unknown" for row in results):
        state = "unknown"
    else:
        state = "exact"
    margins = [float(row["margin"]) for row in results if row["margin"] is not None]
    return state, (min(margins) if margins else None), results


_OBSERVABLE_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    (r"상체를\s*(곧게\s*)?(세우|세운)|상체를\s*(곧게|꼿꼿)", ("torso_upright",)),
    (r"앞으로\s*(깊이\s*)?숙", ("torso_forward_deep",)),
    (r"뒤로\s*젖", ("torso_backward_deep",)),
    (r"두\s*발을\s*(넓게|멀리)|두\s*발.*간격.*(넓|크)", ("feet_wide",)),
    (r"두\s*발을\s*(거의\s*)?붙", ("feet_narrow",)),
    (
        r"양팔(을|은)?[^,.]{0,12}(넓게|활짝|크게)[^,.]{0,8}(벌|편|펼)"
        r"|양팔을\s*(벌|옆으로\s*뻗)",
        ("arms_wide",),
    ),
    (r"두?\s*무릎을\s*(깊이|깊게)\s*굽", ("both_knees_deep",)),
    (r"두\s*무릎을\s*(곧게|쭉)\s*편", ("both_knees_extended",)),
    (r"왼쪽\s*다리[^,.]{0,12}뒤(로|쪽)[^,.]{0,10}(들|올)", ("left_leg_back_raised",)),
    (r"양팔을\s*어깨\s*위로", ("both_arms_above_shoulders",)),
    (r"한\s*발을\s*들", ("one_foot_raised",)),
    (r"한\s*발로\s*균형", ("one_foot_raised", "torso_upright")),
    (r"한쪽\s*무릎을\s*(깊이|깊게)\s*굽", ("either_knee_deep",)),
    (r"손을?\s*머리\s*근처", ("hands_near_head",)),
    (r"팔꿈치를\s*(깊이|깊게)\s*굽", ("both_elbows_deep",)),
    (r"한쪽\s*팔을\s*앞으로\s*뻗", ("one_arm_forward",)),
    (r"팔(은|을)?\s*(곧게\s*)?편", ("both_elbows_extended",)),
    (r"오른팔은\s*들고\s*왼팔은\s*내", ("right_arm_up_left_down",)),
    (r"왼팔은\s*들고\s*오른팔은\s*내", ("left_arm_up_right_down",)),
    (r"왼쪽\s*다리만\s*뒤로", ("left_leg_only_back",)),
    (r"오른손만\s*골반", ("right_hand_on_hip_only",)),
    (r"왼쪽으로\s*몸을\s*기울", ("torso_lean_left",)),
    (r"한쪽\s*팔만\s*어깨\s*위", ("exactly_one_arm_above",)),
    (r"앉아\s*있지\s*않", ("not_seated_like",)),
    (r"팔을\s*들지\s*않", ("both_arms_not_raised",)),
    (r"무릎을\s*굽히지\s*않", ("both_knees_extended",)),
    (r"무릎을\s*(곧게\s*)?편", ("both_knees_extended",)),
    (r"서\s*있지만", ("torso_upright",)),
    (r"양손이\s*머리에서\s*멀", ("hands_far_from_head",)),
    (r"(?<!않은\s)앉은\s*자세", ("seated_like",)),
    (r"쪼그려\s*앉", ("crouched_low",)),
    (r"엎드리|누운", ("torso_near_horizontal",)),
    (r"한쪽\s*다리는\s*앞.*다른\s*다리는\s*뒤", ("stride_front_back",)),
]


def _normalize_query(query: str) -> str:
    normalized = re.sub(r"\s+", " ", query.strip())
    substitutions = (
        (r"\b왼\s*다리", "왼쪽 다리"),
        (r"\b오른\s*다리", "오른쪽 다리"),
        (r"\b두\s*팔", "양팔"),
        (r"몸\s*뒤쪽", "몸 뒤로"),
    )
    for pattern, replacement in substitutions:
        normalized = re.sub(pattern, replacement, normalized)
    return normalized


def parse_semantic_query(query: str) -> ParsedSemanticQuery:
    normalized = _normalize_query(query)
    if not normalized:
        return ParsedSemanticQuery(
            query, normalized, "clarify", (), None, "none", (),
            "찾고 싶은 자세의 팔·다리·몸통 상태를 알려주세요.",
        )

    if normalized in {"자세", "포즈"} or normalized in {"왼쪽", "오른쪽"}:
        return ParsedSemanticQuery(
            query, normalized, "clarify", (), None, "none", (),
            "어느 신체 부위가 어떻게 놓인 자세인지 더 알려주세요.",
        )
    if re.search(r"멋있|예쁘|좋은\s*포즈", normalized):
        return ParsedSemanticQuery(
            query, normalized, "clarify", (), None, "none", ("subjective_style",),
            "멋있음의 기준을 팔·다리·몸통 방향이나 동작 문맥으로 좁혀주세요.",
        )

    contextual_rules = [
        (r"옛\s*전통\s*춤|전통\s*춤", "dance", "allowed", ("traditional_style",)),
        (r"칼.*(직전|휘두)", "sword", "allowed", ("prop", "motion_phase")),
        (r"인사", "greeting", "allowed", ("intent",)),
        (r"슬퍼|슬픈", None, "none", ("emotion",)),
        (r"총.*겨누|권총", "pistol", "allowed", ("prop",)),
        (r"승리.*축하|축하", None, "none", ("narrative_intent",)),
        (r"키보드|타이핑", "typing", "required", ("prop", "action_context")),
        (r"복싱", "boxing", "required", ("action_context",)),
        (r"마우스", "mouse", "required", ("prop", "action_context")),
        (r"계단.*오르|클라임", "climb", "partial", ("action_context",)),
        (r"춤\s*동작|춤추는\s*동작", "dance", "required", ("action_context",)),
    ]
    for pattern, context_key, expectation, unsupported in contextual_rules:
        if re.search(pattern, normalized):
            return ParsedSemanticQuery(
                query,
                normalized,
                "contextual_retrieval",
                (),
                context_key,
                expectation,  # type: ignore[arg-type]
                unsupported,
                None,
            )

    constraints: list[str] = []
    for pattern, constraint_ids in _OBSERVABLE_PATTERNS:
        if re.search(pattern, normalized):
            constraints.extend(constraint_ids)
    constraints = list(dict.fromkeys(constraints))
    if constraints:
        return ParsedSemanticQuery(
            query, normalized, "observable_constraints", tuple(constraints), None,
            "none", (), None,
        )
    return ParsedSemanticQuery(
        query, normalized, "clarify", (), None, "none", ("unparsed_concept",),
        "팔·다리·몸통의 위치나 굽힘처럼 화면에서 보이는 조건을 더 알려주세요.",
    )


def _context_match(source_mapping: dict[str, Any], context_key: str) -> bool:
    canonical = source_mapping["canonical"]
    actions = set(canonical.get("source_action_ids", []))
    domains = set(canonical.get("action_domain", []))
    props = set(canonical.get("intended_props", []))
    raw = (source_mapping.get("raw_action_label") or "").lower()
    if context_key == "dance":
        return "dance" in domains
    if context_key == "boxing":
        return "boxing" in actions
    if context_key == "mouse":
        return "use_mouse" in actions
    if context_key == "climb":
        return "climb" in actions
    if context_key == "sword":
        return "swordplay" in actions or "sword" in props
    if context_key == "greeting":
        return bool({"greet", "wave"} & actions)
    if context_key == "pistol":
        return "pistol" in props or "pistol" in raw
    if context_key == "typing":
        return bool({"type", "use_mouse"} & actions)
    raise ValueError(f"unknown source context key: {context_key}")


class SemanticPoseSearch:
    """Read-only internal runtime over one validated immutable semantic build."""

    def __init__(
        self,
        build_dir: Path,
        *,
        profile_path: Path,
        models_root: Path,
    ) -> None:
        self.build_dir = Path(build_dir)
        self.db_path = self.build_dir / "pose_semantics.db"
        self.manifest_path = self.build_dir / "semantic-build.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("semantic_db_schema_version") != SEMANTIC_DB_SCHEMA_VERSION:
            raise ValueError("semantic runtime requires the member-measurement DB schema")
        validate_semantic_index(self.db_path, self.manifest_path, profile_path=profile_path)
        self.profile = load_embedding_profile(profile_path)
        self.encoder = OnnxE5Encoder(
            self.profile, model_directory(self.profile, models_root)
        )
        self._load_index()

    def _load_index(self) -> None:
        with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as connection:
            document_rows = connection.execute(
                """SELECT d.document_id,d.semantic_unit_id,d.document_type,d.text,
                          d.evidence_state,d.candidate_only,d.retrieval_weight,
                          e.embedding_blob
                   FROM semantic_text_documents d
                   JOIN semantic_embeddings e ON e.document_id=d.document_id
                   ORDER BY d.document_id"""
            ).fetchall()
            unit_rows = connection.execute(
                """SELECT semantic_unit_id,canonical_pose_id,mirrored_pose_id,
                          source_clip_id,source_mapping_json
                   FROM semantic_units ORDER BY semantic_unit_id"""
            ).fetchall()
            member_rows = connection.execute(
                """SELECT semantic_unit_id,pose_id,variant_kind,
                          posecode_measurements_json
                   FROM pose_semantic_members
                   ORDER BY semantic_unit_id,variant_kind"""
            ).fetchall()
        self.documents = [
            {
                "document_id": row[0],
                "semantic_unit_id": row[1],
                "document_type": row[2],
                "text": row[3],
                "evidence_state": row[4],
                "candidate_only": bool(row[5]),
                "retrieval_weight": float(row[6]),
            }
            for row in document_rows
        ]
        self.embedding_matrix = np.stack(
            [np.frombuffer(row[7], dtype="<f4") for row in document_rows]
        )
        self.units = {
            row[0]: {
                "semantic_unit_id": row[0],
                "canonical_pose_id": row[1],
                "mirrored_pose_id": row[2],
                "source_clip_id": row[3],
                "source_mapping": json.loads(row[4]),
                "members": [],
            }
            for row in unit_rows
        }
        for unit_id, pose_id, variant_kind, measurements_json in member_rows:
            self.units[unit_id]["members"].append(
                {
                    "pose_id": pose_id,
                    "variant_kind": variant_kind,
                    "measurements": json.loads(measurements_json),
                }
            )

    def _text_scores(
        self, query: str, *, contextual_only: bool = False
    ) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
        query_vector, _ = self.encoder.encode([query], kind="query")
        similarities = self.embedding_matrix @ query_vector[0]
        unit_scores: dict[str, float] = {}
        best_documents: dict[str, dict[str, Any]] = {}
        for document, similarity in zip(self.documents, similarities, strict=True):
            if contextual_only and not document["candidate_only"]:
                continue
            score = float(similarity) * document["retrieval_weight"]
            unit_id = document["semantic_unit_id"]
            if score > unit_scores.get(unit_id, -math.inf):
                unit_scores[unit_id] = score
                best_documents[unit_id] = {
                    key: document[key]
                    for key in (
                        "document_id", "document_type", "text",
                        "evidence_state", "candidate_only",
                    )
                }
        return unit_scores, best_documents

    def _lexical_ranks(self, query: str, limit: int = 50) -> dict[str, int]:
        terms = [term for term in re.findall(r"[0-9A-Za-z가-힣_]+", query) if len(term) > 1]
        if not terms:
            return {}
        match = " OR ".join(f'"{term}"' for term in terms)
        with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as connection:
            rows = connection.execute(
                """SELECT semantic_unit_id,bm25(semantic_text_documents_fts) AS score
                   FROM semantic_text_documents_fts
                   WHERE semantic_text_documents_fts MATCH ?
                   ORDER BY score LIMIT ?""",
                (match, limit * 3),
            ).fetchall()
        units: list[str] = []
        for unit_id, _ in rows:
            if unit_id not in units:
                units.append(unit_id)
            if len(units) == limit:
                break
        return {unit_id: rank for rank, unit_id in enumerate(units, start=1)}

    def _hybrid_scores(
        self, query: str, *, contextual_only: bool = False
    ) -> tuple[dict[str, float], dict[str, dict[str, Any]], set[str]]:
        dense_scores, best_documents = self._text_scores(
            query, contextual_only=contextual_only
        )
        dense_ranked = sorted(dense_scores, key=dense_scores.get, reverse=True)
        dense_ranks = {unit_id: rank for rank, unit_id in enumerate(dense_ranked, start=1)}
        lexical_ranks = self._lexical_ranks(query)
        rrf_k = int(self.profile["retrieval"]["rrf_k"])
        scores = {
            # Fuse rank positions, never incomparable cosine/BM25 raw scores.
            # The tiny dense term only breaks exact RRF ties deterministically.
            unit_id: 1.0 / (rrf_k + dense_ranks[unit_id])
            + (1.0 / (rrf_k + lexical_ranks[unit_id]) if unit_id in lexical_ranks else 0.0)
            + dense_scores[unit_id] * 1e-9
            for unit_id in dense_scores
        }
        dense_depth = int(self.profile["retrieval"]["dense_candidate_depth"])
        candidates = set(dense_ranked[:dense_depth]) | set(lexical_ranks)
        return scores, best_documents, candidates

    def search(self, query: str, *, top_k: int = 10) -> dict[str, Any]:
        if top_k < 1 or top_k > 100:
            raise ValueError("top_k must be between 1 and 100")
        parsed = parse_semantic_query(query)
        base = {
            "query": query,
            "semantic_build_id": self.manifest["semantic_build_id"],
            "parsed_query": parsed.as_dict(),
            "match_source": "semantic_user",
            "refine_allowed": False,
        }
        if parsed.intent == "clarify":
            return {
                **base,
                "status": "clarification_required",
                "exact_match_status": "not_evaluated",
                "clarification_question": parsed.clarification_question,
                "results": [],
            }

        scores, best_documents, hybrid_candidates = self._hybrid_scores(
            query, contextual_only=parsed.intent == "contextual_retrieval"
        )
        if parsed.intent == "contextual_retrieval":
            matching_units = []
            if parsed.context_key is not None:
                matching_units = [
                    unit_id
                    for unit_id, unit in self.units.items()
                    if _context_match(unit["source_mapping"], parsed.context_key)
                ]
            ranked = sorted(
                matching_units,
                key=lambda unit_id: (
                    unit_id in hybrid_candidates,
                    scores.get(unit_id, -math.inf),
                    unit_id,
                ),
                reverse=True,
            )
            results = []
            for unit_id in ranked[:top_k]:
                unit = self.units[unit_id]
                results.append(
                    {
                        "semantic_unit_id": unit_id,
                        "pose_id": unit["canonical_pose_id"],
                        "variant_kind": "original",
                        "source_clip_id": unit["source_clip_id"],
                        "score": scores[unit_id],
                        "evidence_state": "contextual",
                        "exact_pose_claim": False,
                        "context_key": parsed.context_key,
                        "context_provenance": {
                            "kind": "source_catalog_or_filename",
                            "source_context_only": True,
                        },
                        "matched_constraints": [],
                        "unknown_constraints": [],
                        "best_text_document": best_documents[unit_id],
                        "match_source": "semantic_user",
                        "refine_allowed": False,
                    }
                )
            status = "contextual_candidates" if results else "library_gap"
            return {
                **base,
                "status": status,
                "exact_match_status": "library_gap",
                "gap_reason": list(parsed.unsupported_concepts),
                "results": results,
            }

        matching: list[dict[str, Any]] = []
        unknown_members = 0
        for unit_id, unit in self.units.items():
            for member in unit["members"]:
                state, margin, details = evaluate_constraints(
                    parsed.constraint_ids, member["measurements"]
                )
                if state == "unknown":
                    unknown_members += 1
                if state == "exact":
                    matching.append(
                        {
                            "semantic_unit_id": unit_id,
                            "pose_id": member["pose_id"],
                            "variant_kind": member["variant_kind"],
                            "constraint_margin": margin or 0.0,
                            "constraint_results": details,
                        }
                    )
        # Keep one concrete member per direction-neutral unit. A side-specific
        # query can resolve to the mirrored member; a neutral tie prefers original.
        best_member_by_unit: dict[str, dict[str, Any]] = {}
        side_specific = any(CONSTRAINTS[item].get("side_specific") for item in parsed.constraint_ids)
        for member in matching:
            unit_id = member["semantic_unit_id"]
            previous = best_member_by_unit.get(unit_id)
            preference = (
                member["constraint_margin"],
                member["variant_kind"] == "original",
            )
            if previous is None or preference > (
                previous["constraint_margin"],
                previous["variant_kind"] == "original",
            ):
                best_member_by_unit[unit_id] = member
        ranked_members = sorted(
            best_member_by_unit.values(),
            key=lambda member: (
                scores.get(member["semantic_unit_id"], -math.inf),
                member["constraint_margin"],
                member["semantic_unit_id"],
            ),
            reverse=True,
        )
        results = []
        for member in ranked_members[:top_k]:
            unit_id = member["semantic_unit_id"]
            unit = self.units[unit_id]
            results.append(
                {
                    **member,
                    "source_clip_id": unit["source_clip_id"],
                    "score": scores[unit_id],
                    "evidence_state": "observed",
                    "exact_pose_claim": True,
                    "side_resolved": bool(side_specific),
                    "matched_constraints": list(parsed.constraint_ids),
                    "unknown_constraints": [],
                    "best_text_document": best_documents[unit_id],
                    "match_source": "semantic_user",
                    "refine_allowed": False,
                }
            )
        return {
            **base,
            "status": "success" if results else "library_gap",
            "exact_match_status": "exact" if results else "library_gap",
            "matching_pose_members": len(matching),
            "matching_semantic_units": len(best_member_by_unit),
            "unknown_pose_members": unknown_members,
            "results": results,
        }


def discover_semantic_build(builds_root: Path) -> Path:
    candidates: list[tuple[Path, int]] = []
    for manifest_path in Path(builds_root).glob("*/semantic-build.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("semantic_db_schema_version") == SEMANTIC_DB_SCHEMA_VERSION:
            candidates.append(
                (manifest_path.parent, int(manifest["semantic_index_builder_version"]))
            )
    if not candidates:
        raise ValueError(
            f"semantic DB schema v{SEMANTIC_DB_SCHEMA_VERSION} build not found"
        )
    newest_builder = max(version for _, version in candidates)
    newest = [path for path, version in candidates if version == newest_builder]
    if len(newest) != 1:
        raise ValueError(
            f"found {len(newest)} semantic builds for latest builder v{newest_builder}; "
            "pass --build-dir explicitly"
        )
    return newest[0]
