"""Small, asset-free contract checks for the handoff worktree."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.search_routing_demo import example
from src.experimental.rough_router import Observation, plan_search
from src.experimental.rough_semantic import documents_for
from src.experimental.rough_semantic_v2 import execute_structured_plan
from src.vlm.rough_slots import SCHEMA_VERSION, parse_slots


def sample_person():
    raw = {"rough_semantics": {"schema_version": SCHEMA_VERSION, "people": [{
        "person_index": 0,
        "framing": "full",
        "body_visibility": {
            part: "visible" for part in (
                "torso", "left_arm", "right_arm", "left_leg", "right_leg"
            )
        },
        "upper_action": [{
            "value": "arms crossed", "evidence_kind": "body_shape",
            "evidence_note": "visible wrists", "status": "tentative",
        }],
    }]}}
    parsed = parse_slots(raw, 1)
    assert not parsed.issues
    return parsed.people[0]


def test_plan_is_deterministic_and_only_plans():
    first = example()
    second = example()
    assert first == second
    plan = first["plan"]
    assert plan["route"] == "full_observed"
    assert not plan["candidates_changed"]
    assert not plan["refine_allowed"]
    assert all(c["execution"] == "not_executed_planning_only"
               for c in plan["channels"])
    assert {"G", "S", "F"} <= {c["kind"] for c in plan["channels"]}


def test_missing_pose_uses_semantics_without_claiming_geometry():
    plan = plan_search(sample_person(), Observation())
    assert any(c.kind == "S" for c in plan.channels)
    assert not any(c.kind == "G" for c in plan.channels)


def test_region_documents_keep_upper_and_lower_apart():
    row = {
        "pose_id": "example",
        "bvh_sha256": "sha256:example",
        "posecode": {"observed_atoms": [
            {"subject": "left_elbow", "predicate": "joint_flexion", "bucket": "bent"},
            {"subject": "right_knee", "predicate": "joint_flexion", "bucket": "extended"},
        ]},
    }
    docs = {d["region"]: d["text"] for d in documents_for(row)}
    assert "knee" not in docs["upper"]
    assert "elbow" not in docs["lower"]
    assert "knee" in docs["full"] and "elbow" in docs["full"]


def test_semantic_executor_keeps_refine_and_composition_off():
    class RecordingRuntime:
        version = "fake-test"
        def __init__(self):
            self.calls = []
        def search(self, query, **options):
            self.calls.append((query, options))
            return {"results": []}

    runtime = RecordingRuntime()
    plan = plan_search(sample_person(), Observation())
    outcome = execute_structured_plan(plan, runtime, slots=sample_person())
    assert runtime.calls
    assert outcome["executed_query_count"] == len(runtime.calls)
    assert not outcome["refine_allowed"]
    assert not outcome["composition_executed"]
    assert not outcome["geometry_executed"]


if __name__ == "__main__":
    import traceback
    tests = [value for name, value in list(globals().items())
             if name.startswith("test_") and callable(value)]
    failed = 0
    for check in tests:
        try:
            check()
            print("PASS", check.__name__)
        except Exception:
            failed += 1
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(bool(failed))
