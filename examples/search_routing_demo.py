#!/usr/bin/env python3
"""Asset-free demonstration of image-slot search planning, not retrieval."""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experimental.rough_router import Observation, plan_search
from src.vlm.rough_slots import SCHEMA_VERSION, parse_slots


def example():
    raw = {
        "rough_semantics": {
            "schema_version": SCHEMA_VERSION,
            "people": [{
                "person_index": 0,
                "framing": "full",
                "body_visibility": {
                    name: "visible" for name in (
                        "torso", "left_arm", "right_arm", "left_leg", "right_leg"
                    )
                },
                "upper_action": [{
                    "value": "arms crossed",
                    "evidence_kind": "body_shape",
                    "evidence_note": "both wrists overlap the torso",
                    "status": "tentative",
                }],
                "support_state": [{
                    "value": "standing",
                    "evidence_kind": "body_shape",
                    "evidence_note": "both legs extend below the hips",
                    "status": "tentative",
                }],
            }],
        }
    }
    slots = parse_slots(raw, 1)
    if slots.issues:
        raise ValueError(slots.issues)
    points = np.zeros((17, 2), dtype=float)
    points[5:] = [
        [-1, -2], [1, -2], [-2, -1], [2, -1], [-1, 0], [1, 0],
        [-0.5, 0], [0.5, 0], [-0.6, 1], [0.6, 1], [-0.6, 2], [0.6, 2],
    ]
    mask = np.ones(17, dtype=bool)
    mask[:5] = False
    plan = plan_search(slots.people[0], Observation(points, mask, state="valid"))
    return {
        "note": "SearchPlan only: no DB lookup, model call, composition, or refine",
        "plan_id": plan.plan_id,
        "plan": plan.to_dict(),
    }


if __name__ == "__main__":
    print(json.dumps(example(), ensure_ascii=False, indent=2))
