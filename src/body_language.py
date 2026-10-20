"""Qualitative body observations for semantic search; never generates coordinates.

Unknown or unmentioned attributes contribute no score. Library phrases are sparse,
so this is a conservative Korean vocabulary adapter, not a complete NLP model.
"""
from __future__ import annotations

import re

BODY_VALUES = {
    "knees": ("bent", "straight", "mixed", "unknown"),
    "legs": ("apart", "together", "staggered", "unknown"),
    "arms": ("lowered", "raised", "spread", "mixed", "unknown"),
    "torso": ("forward", "backward", "upright", "unknown"),
}
BODY_WEIGHTS = {"knees": 3.0, "legs": 3.0, "arms": 2.0, "torso": 1.5}


def validated_observations(value) -> dict:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items()
            if key in BODY_VALUES and isinstance(item, str) and item in BODY_VALUES[key]}


def _has(text, pattern):
    for match in re.finditer(pattern, text):
        # Do not turn "굽히지 않는다" or an occluded limb into positive evidence.
        span = match.group() + text[match.end():match.end() + 7]
        if not re.search(r"않|아니|불명|모호|가려|보이지", span):
            return True
    return False


_PARTS = re.compile(r"팔꿈치|발목|무릎|다리|주먹|상체|몸통|몸을|골반|어깨|허리|팔|손|발")


def _part_has(text, names, pattern):
    """A verb about hands must not become evidence about a nearby knee."""
    parts = list(_PARTS.finditer(text))
    for index, part in enumerate(parts):
        if part.group() not in names:
            continue
        end = parts[index + 1].start() if index + 1 < len(parts) else len(text)
        segment = text[part.start():end]
        if _has(segment, pattern):
            return True
    return False


def observations_from_text(text: str) -> dict:
    text = re.sub(r"\s+", "", text or "")
    gap = r"[^,.;。]{0,18}"
    result = {}
    knees = set()
    if (_part_has(text, {"무릎", "다리"}, rf"(?:무릎|다리){gap}(?:굽|구부|접)")
            or _has(text, r"쪼그(?:려|리)|낮게앉")):
        knees.add("bent")
    if _part_has(text, {"무릎", "다리"}, rf"(?:무릎|다리){gap}(?:펴|편|뻗)"):
        knees.add("straight")
    if knees:
        result["knees"] = next(iter(knees)) if len(knees) == 1 else "mixed"
    if _part_has(text, {"다리", "무릎"}, rf"(?:다리|무릎){gap}(?:앞뒤|엇갈)"):
        result["legs"] = "staggered"
    elif _part_has(text, {"다리", "무릎"}, rf"(?:다리|무릎){gap}(?:벌리|벌린|벌려|넓)"):
        result["legs"] = "apart"
    elif _part_has(text, {"다리", "무릎"}, rf"(?:다리|무릎){gap}(?:모으|모은|붙)"):
        result["legs"] = "together"
    arms = set()
    if _part_has(text, {"팔", "손", "주먹"},
                 rf"(?:팔|손|주먹){gap}(?:내리|내린|내려|내림|늘어뜨|늘어트|아래)"):
        arms.add("lowered")
    if _part_has(text, {"팔", "손", "주먹"}, rf"(?:팔|손|주먹){gap}(?:올리|올린|위로|머리위)"):
        arms.add("raised")
    if _part_has(text, {"팔"}, rf"팔{gap}(?:벌리|벌린|벌려|옆으로뻗|양옆으로뻗)"):
        arms.add("spread")
    if arms:
        result["arms"] = next(iter(arms)) if len(arms) == 1 else "mixed"
    torso = set()
    if _part_has(text, {"상체", "몸통", "몸을"}, rf"(?:상체|몸통|몸을){gap}(?:앞으로|숙|구부)"):
        torso.add("forward")
    if _part_has(text, {"상체", "몸통", "몸을"}, rf"(?:상체|몸통|몸을){gap}(?:뒤로|뒤쪽으로|젖)"):
        torso.add("backward")
    if _part_has(text, {"상체", "몸통", "몸을"}, rf"(?:상체|몸통|몸을){gap}(?:세우|세운|곧게)"):
        torso.add("upright")
    if len(torso) == 1:
        result["torso"] = next(iter(torso))
    return result


def query_observations(body_details: str, explicit: dict) -> dict:
    observed = observations_from_text(body_details)
    # An explicit unknown must stay unknown instead of being guessed from prose.
    observed.update(validated_observations(explicit))
    return {key: value for key, value in observed.items() if value != "unknown"}


def specific_observation_count(observed: dict) -> int:
    """Plain standing attributes alone must not unlock a semantic route."""
    generic = {"knees": "straight", "legs": "together",
               "arms": "lowered", "torso": "upright"}
    return sum(value not in ("unknown", generic.get(key)) for key, value in observed.items())


def compare_observations(query: dict, candidate: dict) -> dict:
    matched, conflicts = {}, {}
    reward = penalty = 0.0
    for key, value in query.items():
        other = candidate.get(key)
        if value == "unknown" or other in (None, "unknown"):
            continue
        if value == other:
            matched[key] = value
            reward += BODY_WEIGHTS[key]
        else:
            conflicts[key] = {"query": value, "candidate": other}
            penalty += BODY_WEIGHTS[key]
    return {"matched": matched, "conflicts": conflicts,
            "reward": reward, "penalty": penalty}
