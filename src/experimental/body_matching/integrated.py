"""Optional body design extension to the first cut analysis.

The body block is independently validated. Invalid body output never changes
person count, boxes, routing, hand metadata or geometric pose selection.
"""

from copy import deepcopy
from .coarse import AXES, parse_response, unknown

VERSION = "body-integrated.v6.independent-axes"
PROMPT_EXTENSION = """
Additional output field: person_bodies, one entry for EACH approx_boxes person, in exactly that order.
Finish the original person count, shot and box analysis using the original rules. Each approx_box must have 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1, never 0~1000 coordinates. Check this before returning. Do not add,
remove or reorder people for body classification. Do not modify the original fields' criteria.
Then read each fictional character's face, CLOTHING (neckline, sleeves, hems, exposed skin),
and visible body design before choosing the mannequin family. Describe these visible cues
briefly in evidence=[face,clothing,body] (Korean, at most 35 characters each). Describe actual marks before any family conclusion; avoid generic statements like "clothing details are simple". Garment boundaries across the chest/waist
are not muscle divisions. Polygonal construction strokes are not anatomical detail; unfinished
hair is not a bald design. Hair length, shoulder width, angular jaw, large eyes or sitting posture
alone do not decide the family. Use their combined visible design impression, not names/text.
Target=ambiguous means the person cannot be linked to its approx_box; it does NOT mean the family is uncertain. A sole readable character is target=clear. Normal clothing, rough line art and lack of shading do not by themselves make the design unknown.
First choose masculine/feminine family; use a tentative best guess when supported, androgynous
or unknown when no preference can be read. This is fictional drawn design, not biological sex.
Ordinary build, average fullness and standard proportions are defaults. Only clearly emphasized
special traits receive clear certainty: muscle definition or exaggerated mass (not clothing folds
or angular arms), conspicuous softness/heaviness, or unmistakably elongated/compact stylization.
A smooth adult is not a child; do not infer age or exact head counts. No new positions/joints.
Do not use occlusion or loose clothing alone to invent body traits. Cropped/foreshortened limbs
may make proportion unreadable. A visible muscular upper body can support muscle despite absent legs.
Each entry has these keys (no body_id/preset selection; the server maps classes to assets):
{ "person_index": <zero-based approx_boxes index>, "evidence": ["<eye/brow/mouth styling>","<neckline, hems, exposed skin and lower garment>","<visible body design>"],
  "target": "clear|ambiguous", "representation": "character|stick|face_only|unreadable",
  "proportion_readable": true|false,
  "axes": {
    "presentation": ["masculine|feminine|androgynous|unknown", "clear|tentative|unknown"],
    "muscle": ["ordinary|muscular|very_muscular|unknown", "clear|tentative|unknown"],
    "fullness": ["slim|average|soft|heavy|unknown", "clear|tentative|unknown"],
    "proportion": ["compact|standard|elongated|unknown", "clear|tentative|unknown"]
  }
}
Each axis is a two-item array [value, certainty], in that exact order.
person_bodies must have num_people entries (zero people => []). Values are single enum choices,
not pipe-separated strings. Body fields are independent of pose actions and hand-shape recommendations.
"""


def enabled(cfg):
    return (
        getattr(cfg, "body_classification_strategy", "legacy") == "integrated"
        and getattr(cfg, "body_matching_mode", "off") != "off"
    )


def analysis_prompt(base):
    return base + PROMPT_EXTENSION


def parse_bodies(raw, count):
    """Keep all original slots; reject ambiguous indexing instead of shifting bodies."""

    def missing(index, reason):
        result = unknown("V" + str(index), reason)
        result["version"] = VERSION
        return result

    if not isinstance(raw, list) or len(raw) != count:
        return [missing(i, "integrated_body_count_mismatch") for i in range(count)]
    indices = [p.get("person_index") if isinstance(p, dict) else None for p in raw]
    if any(type(i) is not int for i in indices) or set(indices) != set(range(count)):
        return [missing(i, "integrated_body_identity_mismatch") for i in range(count)]
    by_id = {p["person_index"]: p for p in raw}
    result = []
    for index in range(count):
        p, ident = by_id[index], "V" + str(index)
        try:
            if set(p) != {
                "person_index",
                "evidence",
                "target",
                "representation",
                "proportion_readable",
                "axes",
            }:
                raise ValueError("invalid_integrated_body_fields")
            evidence = p["evidence"]
            if isinstance(evidence, list):
                if len(evidence) != 3 or any(
                    not isinstance(v, str) or not v.strip() or len(v) > 150
                    for v in evidence
                ):
                    raise ValueError("invalid_integrated_body_evidence")
                evidence = " / ".join(evidence)
            if isinstance(evidence, dict):
                if set(evidence) != {"face", "clothing", "body"} or any(
                    not isinstance(v, str) or not v.strip() or len(v) > 150
                    for v in evidence.values()
                ):
                    raise ValueError("invalid_integrated_body_evidence")
                evidence = " / ".join(evidence[k] for k in ("face", "clothing", "body"))
            if (
                not isinstance(evidence, str)
                or not evidence.strip()
                or len(evidence) > 500
            ):
                raise ValueError("invalid_integrated_body_evidence")
            if not isinstance(p["axes"], dict) or set(p["axes"]) != set(AXES):
                raise ValueError("invalid_integrated_body_axes")
            axes = {}
            for name, axis in p["axes"].items():
                if isinstance(axis, list) and len(axis) == 2:
                    axis = dict(value=axis[0], certainty=axis[1])
                if not isinstance(axis, dict) or set(axis) != {"value", "certainty"}:
                    raise ValueError("invalid_integrated_body_axis")
                axes[name] = dict(**axis, reason=evidence)
            o = {k: p[k] for k in ("target", "representation", "proportion_readable")}
            o.update(person_id=ident, axes=axes)
            parsed = parse_response({"people": [o]}, [ident])[ident]
            parsed["version"] = VERSION
            result.append(parsed)
        except (KeyError, TypeError, ValueError):
            result.append(missing(index, "invalid_integrated_body"))
    return result


def detect_body(vlm, index):
    bodies = getattr(vlm, "person_bodies", [])
    if type(index) is not int or not 0 <= index < len(bodies):
        return None
    return deepcopy(bodies[index])


SYSTEM_EXTENSION = " 인물별 얼굴·의상·몸의 그림 표현을 함께 읽어 드로잉 마네킹 체형도 분류한다. 사람 수·구도·대략 박스와 체형을 독립 항목으로 기록한다."


def response_schema():
    def enum(values):
        return {"type": "string", "enum": list(values)}

    def obj(properties):
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    def array(item):
        return {"type": "array", "items": item}

    axis = {
        k: {
            "type": "array",
            "items": enum(list(values) + ["clear", "tentative"]),
            "minItems": 2,
            "maxItems": 2,
            "description": "[value, certainty]; second item is clear, tentative or unknown",
        }
        for k, values in AXES.items()
    }
    body = obj(
        {
            "person_index": {"type": "integer"},
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 3,
                "description": "[face styling, clothing neckline/hems, visible body design]",
            },
            "target": enum(("clear", "ambiguous")),
            "representation": enum(("character", "stick", "face_only", "unreadable")),
            "proportion_readable": {"type": "boolean"},
            "axes": obj(axis),
        }
    )
    return obj(
        {
            "num_people": {"type": "integer", "minimum": 0, "maximum": 20},
            "shot": enum(("full_half", "bust", "face")),
            "action": enum(
                (
                    "standing",
                    "sitting",
                    "walking",
                    "running",
                    "reaching",
                    "lying",
                    "other",
                )
            ),
            "view": enum(("front", "side", "back", "three_quarter")),
            "relationship": enum(
                ("solo", "talking", "hugging", "holding_hands", "fighting")
            ),
            "approx_boxes": array(
                obj(
                    {
                        k: {"type": "number", "minimum": 0, "maximum": 1}
                        for k in ("x1", "y1", "x2", "y2")
                    }
                )
            ),
            "lower_body_visible": array({"type": "boolean"}),
            "body_scopes": array(
                {
                    "type": ["string", "null"],
                    "enum": ["full", "half", "bust", "shoulder_head", "head", None],
                }
            ),
            "dialogue": {"type": ["string", "null"]},
            "person_bodies": array(body),
        }
    )
