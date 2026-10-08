"""v3.2 evaluator/repairer contract. No GT or angular scoring in routing."""

import copy
from . import schema

obj = schema.obj
STRING = {"type": "string"}
PID = {"type": ["integer", "null"]}
NAMES = {"type": "array", "items": {"type": "string", "enum": schema.COCO17}}
BOX = copy.deepcopy(schema.box)
NULLBOX = {"anyOf": [BOX, {"type": "null"}]}
TYPES = [
    "missing_person",
    "missing_joint",
    "obvious_pose_error",
    "false_or_duplicate_person",
]
EVALUATION_SCHEMA = obj(
    {
        "decision": {"type": "string", "enum": ["pass", "repair", "review_needed"]},
        "issues": {
            "type": "array",
            "items": obj(
                {
                    "issue_id": STRING,
                    "type": {"type": "string", "enum": TYPES},
                    "person_index": PID,
                    "joint_names": NAMES,
                    "region_bbox": NULLBOX,
                    "evidence": STRING,
                }
            ),
        },
        "uncertainties": {
            "type": "array",
            "items": obj(
                {
                    "person_index": PID,
                    "joint_names": NAMES,
                    "region_description": STRING,
                    "reason": STRING,
                }
            ),
        },
    }
)
REPAIR_SCHEMA = obj(
    {
        "edits": {
            "type": "array",
            "items": obj(
                dict(
                    schema.point["properties"],
                    issue_id=STRING,
                    person_index={"type": "integer"},
                    evidence=STRING,
                )
            ),
        },
        "added_people": {
            "type": "array",
            "items": obj(
                {
                    "issue_id": STRING,
                    "new_person_id": STRING,
                    "bbox": BOX,
                    "keypoints": {"type": "array", "items": schema.point},
                    "evidence": STRING,
                }
            ),
        },
        "removed_people": {
            "type": "array",
            "items": obj(
                {
                    "issue_id": STRING,
                    "person_index": {"type": "integer"},
                    "evidence": STRING,
                }
            ),
        },
        "unresolved": {
            "type": "array",
            "items": obj({"issue_id": STRING, "reason": STRING}),
        },
    }
)


SINGLE_SCHEMA = obj(
    {
        "evaluation": copy.deepcopy(EVALUATION_SCHEMA),
        "patch": copy.deepcopy(REPAIR_SCHEMA),
    }
)


def merge_single_response(base, response):
    """One response, same strict/atomic patch rules as the two-stage protocol."""
    keys(response, SINGLE_SCHEMA["properties"])
    evaluation, patch = response["evaluation"], response["patch"]
    validate_evaluation(evaluation, base)
    if evaluation["decision"] == "repair":
        return merge_repair(base, evaluation, patch)
    keys(patch, REPAIR_SCHEMA["properties"])
    if any(not isinstance(value, list) or value for value in patch.values()):
        raise ValueError("No-change decision must have empty patch arrays")
    return copy.deepcopy(base), {
        "changes": [],
        "unresolved": [],
        "uncertainties": evaluation["uncertainties"],
        "visual_reassessment": False,
        "unchanged_joint_count": sum(len(p["keypoints"]) for p in base["people"]),
    }


def keys(x, expected):
    if not isinstance(x, dict) or set(x) != set(expected):
        raise ValueError("Invalid object fields")


def array(x, maximum=340):
    if not isinstance(x, list) or len(x) > maximum:
        raise ValueError("Invalid array")


def brief(s):
    if not isinstance(s, str) or not s.strip() or len(s) > 400:
        raise ValueError("Missing/overlong evidence")


def names(ns):
    array(ns, 17)
    if any(n not in schema.COCO17 for n in ns) or len(ns) != len(set(ns)):
        raise ValueError("Invalid joint names")


def validate_evaluation(data, base):
    schema.validate_prediction(base)
    keys(data, EVALUATION_SCHEMA["properties"])
    array(data["issues"])
    array(data["uncertainties"])
    ids = set()
    pids = {x["person_index"] for x in base["people"]}
    for issue in data["issues"]:
        keys(issue, EVALUATION_SCHEMA["properties"]["issues"]["items"]["properties"])
        brief(issue["issue_id"])
        brief(issue["evidence"])
        names(issue["joint_names"])
        if issue["issue_id"] in ids or issue["type"] not in TYPES:
            raise ValueError("Duplicate/invalid issue")
        ids.add(issue["issue_id"])
        pid = issue["person_index"]
        if issue["type"] == "missing_person":
            if pid is not None or issue["joint_names"]:
                raise ValueError("Invalid missing-person reference")
            schema.valid_box(issue["region_bbox"])
        else:
            if (
                type(pid) is not int
                or pid not in pids
                or issue["region_bbox"] is not None
            ):
                raise ValueError("Invalid existing person reference")
            if (
                issue["type"] in ["missing_joint", "obvious_pose_error"]
                and not issue["joint_names"]
            ):
                raise ValueError("Joint issue requires target joints")
            if issue["type"] == "false_or_duplicate_person" and issue["joint_names"]:
                raise ValueError("Person removal is not a joint issue")
    for u in data["uncertainties"]:
        keys(u, EVALUATION_SCHEMA["properties"]["uncertainties"]["items"]["properties"])
        names(u["joint_names"])
        brief(u["region_description"])
        brief(u["reason"])
        if u["person_index"] is not None and (
            type(u["person_index"]) is not int or u["person_index"] not in pids
        ):
            raise ValueError("Invalid uncertainty reference")
    expected = (
        "repair" if ids else ("review_needed" if data["uncertainties"] else "pass")
    )
    if data["decision"] != expected:
        raise ValueError("Decision contradicts findings")
    return data


def merge_repair(base, evaluation, patch):
    validate_evaluation(evaluation, base)
    if evaluation["decision"] != "repair":
        raise ValueError("No repair authorized by evaluation")
    keys(patch, REPAIR_SCHEMA["properties"])
    for value in patch.values():
        array(value)
    issues = {x["issue_id"]: x for x in evaluation["issues"]}
    covered = set()
    edited = set()
    removed = set()
    added_issues = set()
    new_ids = set()
    audit = []
    result = copy.deepcopy(base)
    by_id = {x["person_index"]: x for x in result["people"]}
    original = {x["person_index"]: x for x in base["people"]}

    def issue_for(x, kind):
        keys(x, REPAIR_SCHEMA["properties"][kind]["items"]["properties"])
        iid = x["issue_id"]
        if iid not in issues:
            raise ValueError("Unknown issue reference")
        brief(x["reason"] if kind == "unresolved" else x["evidence"])
        covered.add(iid)
        return issues[iid]

    for x in patch["removed_people"]:
        i = issue_for(x, "removed_people")
        pid = x["person_index"]
        if (
            i["type"] != "false_or_duplicate_person"
            or type(pid) is not int
            or pid != i["person_index"]
            or pid in removed
        ):
            raise ValueError("Invalid removal")
        removed.add(pid)
        audit.append({"kind": "remove_person", **x})
    for x in patch["edits"]:
        i = issue_for(x, "edits")
        schema.valid_point(x)
        pid = x["person_index"]
        key = (pid, x["name"])
        if (
            type(pid) is not int
            or pid != i["person_index"]
            or i["type"] not in ["missing_joint", "obvious_pose_error"]
            or pid in removed
            or key in edited
        ):
            raise ValueError("Invalid/conflicting edit")
        allowed = set(i["joint_names"])
        # Only the same anatomical arm/leg may be adjusted with the flagged anchor.
        for side in ["left", "right"]:
            for limb in [("shoulder", "elbow", "wrist"), ("hip", "knee", "ankle")]:
                chain = {side + "_" + n for n in limb}
                if chain.intersection(i["joint_names"]):
                    allowed.update(chain)
        if x["name"] not in allowed:
            raise ValueError("Unrelated joint edit")
        point = next(k for k in by_id[pid]["keypoints"] if k["name"] == x["name"])
        before = copy.deepcopy(point)
        for k in ["x", "y", "state"]:
            point[k] = x[k]
        point.pop("score", None)
        edited.add(key)
        audit.append(
            {"kind": "joint", "before": before, "after": copy.deepcopy(point), **x}
        )
    for x in patch["added_people"]:
        i = issue_for(x, "added_people")
        schema.valid_box(x["bbox"])
        brief(x["new_person_id"])
        array(x["keypoints"], 17)
        if (
            i["type"] != "missing_person"
            or x["issue_id"] in added_issues
            or x["new_person_id"] in new_ids
        ):
            raise ValueError("Duplicate/invalid addition")
        if len(x["keypoints"]) != 17:
            raise ValueError("New person requires 17 entries")
        for k in x["keypoints"]:
            keys(k, schema.point["properties"])
            schema.valid_point(k)
        lookup = {k["name"]: k for k in x["keypoints"]}
        if set(lookup) != set(schema.COCO17):
            raise ValueError("Duplicate/missing added joint")
        pid = max(by_id, default=-1) + 1
        person = {
            "person_index": pid,
            "bbox": copy.deepcopy(x["bbox"]),
            "keypoints": [copy.deepcopy(lookup[n]) for n in schema.COCO17],
        }
        by_id[pid] = person
        result["people"].append(person)
        added_issues.add(x["issue_id"])
        new_ids.add(x["new_person_id"])
        audit.append(
            {
                "kind": "add_person",
                "final_person_index": pid,
                "issue_id": x["issue_id"],
                "evidence": x["evidence"],
            }
        )
    seen_unresolved = set()
    for x in patch["unresolved"]:
        issue_for(x, "unresolved")
        if x["issue_id"] in seen_unresolved:
            raise ValueError("Duplicate unresolved issue")
        seen_unresolved.add(x["issue_id"])
    if covered != set(issues):
        raise ValueError("Issue silently unhandled")
    result["people"] = [x for x in result["people"] if x["person_index"] not in removed]
    schema.validate_prediction(
        result
    )  # validate only; preserve stable IDs and untouched scores
    for pid, person in original.items():
        if pid in removed:
            continue
        for a, b in zip(person["keypoints"], by_id[pid]["keypoints"]):
            if (pid, a["name"]) not in edited and a != b:
                raise ValueError("Untouched joint mutated")
    return result, {
        "changes": audit,
        "unresolved": patch["unresolved"],
        "uncertainties": evaluation["uncertainties"],
        "visual_reassessment": False,
        "unchanged_joint_count": sum(
            (pid, k["name"]) not in edited
            for pid, person in original.items()
            if pid not in removed
            for k in person["keypoints"]
        ),
    }
