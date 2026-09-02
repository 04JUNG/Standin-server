"""Blender에서 실행하는 V3.2.1 palm-roll 순수 수학 control."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from mathutils import Matrix, Vector


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from converter import retarget as rt  # noqa: E402


RESULTS: list[dict] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append({"name": name, "passed": bool(condition), "detail": detail})
    print(f"[{'PASS' if condition else 'FAIL'}] {name} {detail}")


def matrix_delta(a: Matrix, b: Matrix) -> float:
    return max(abs(float(a[r][c] - b[r][c])) for r in range(4) for c in range(4))


FRAME = {
    "forward": Vector((1.0, 0.0, 0.0)),
    "normal": Vector((0.0, 0.0, 1.0)),
}
IDENTITY = Matrix.Identity(4)


for angle_deg in (0.0, 30.0, -30.0, 90.0, -90.0):
    pose = Matrix.Rotation(math.radians(angle_deg), 4, Vector((1.0, 0.0, 0.0)))
    result, diag = rt._solve_palm_roll(
        source_rest_rotation=IDENTITY,
        source_pose_rotation=pose,
        source_frame=FRAME,
        target_rest_rotation=IDENTITY,
        target_frame=FRAME,
        baseline_rotation=IDENTITY,
        mu=1.0,
    )
    expected_mode = "identity" if angle_deg == 0.0 else "corrected"
    check(
        f"roll {angle_deg:+.0f}deg mode",
        diag["mode"] == expected_mode,
        str(diag),
    )
    expected = IDENTITY if angle_deg == 0.0 else pose
    check(
        f"roll {angle_deg:+.0f}deg recovery",
        matrix_delta(result, expected) <= 1e-6,
        f"max_matrix_delta={matrix_delta(result, expected):.3e}",
    )


for angle_deg in (179.0, -179.0):
    pose = Matrix.Rotation(math.radians(angle_deg), 4, Vector((1.0, 0.0, 0.0)))
    result, diag = rt._solve_palm_roll(
        source_rest_rotation=IDENTITY,
        source_pose_rotation=pose,
        source_frame=FRAME,
        target_rest_rotation=IDENTITY,
        target_frame=FRAME,
        baseline_rotation=IDENTITY,
        mu=1.0,
    )
    check(
        f"near-180 {angle_deg:+.0f}deg fallback",
        diag["mode"] == "fallback" and matrix_delta(result, IDENTITY) == 0.0,
        str(diag),
    )


pose_60 = Matrix.Rotation(math.radians(60.0), 4, Vector((1.0, 0.0, 0.0)))
for mu in rt._QA_PALM_POLICY["mu_candidates"]:
    result, diag = rt._solve_palm_roll(
        source_rest_rotation=IDENTITY,
        source_pose_rotation=pose_60,
        source_frame=FRAME,
        target_rest_rotation=IDENTITY,
        target_frame=FRAME,
        baseline_rotation=IDENTITY,
        mu=mu,
    )
    expected = Matrix.Rotation(math.radians(60.0 * mu), 4, Vector((1.0, 0.0, 0.0)))
    check(
        f"mu ladder {mu:.2f}",
        matrix_delta(result, expected) <= 1e-6,
        f"applied={diag.get('applied_deg')} delta={matrix_delta(result, expected):.3e}",
    )


baseline = Matrix.Rotation(math.radians(37.0), 4, Vector((0.0, 1.0, 0.0)))
result, diag = rt._solve_palm_roll(
    source_rest_rotation=IDENTITY,
    source_pose_rotation=pose_60,
    source_frame=FRAME,
    target_rest_rotation=IDENTITY,
    target_frame=FRAME,
    baseline_rotation=baseline,
    mu=0.0,
)
check(
    "mu=0 exact frozen V3.2 matrix",
    matrix_delta(result, baseline) == 0.0,
    f"max_matrix_delta={matrix_delta(result, baseline):.3e}",
)


theta, reason = rt._signed_angle_about_axis(
    Vector((1.0, 0.0, 0.0)),
    Vector((0.0, 1.0, 0.0)),
    Vector((1.0, 0.0, 0.0)),
)
check("projection-zero fallback", theta is None, reason)


check(
    "common role priority index+pinky",
    rt._common_palm_roles(
        {"index": ("i", Vector()), "pinky": ("p", Vector()), "thumb": ("t", Vector())},
        {"index": ("i", Vector()), "pinky": ("p", Vector()), "thumb": ("t", Vector())},
    ) == ("index", "pinky"),
)
check(
    "common role fallback index+thumb",
    rt._common_palm_roles(
        {"index": ("i", Vector()), "thumb": ("t", Vector())},
        {"index": ("i", Vector()), "thumb": ("t", Vector())},
    ) == ("index", "thumb"),
)
check(
    "one-side role mismatch is unavailable",
    rt._common_palm_roles(
        {"index": ("i", Vector()), "pinky": ("p", Vector())},
        {"index": ("i", Vector()), "thumb": ("t", Vector())},
    ) is None,
)


payload = {
    "blender_version": __import__("bpy").app.version_string,
    "policy_sha256": rt._QA_PALM_POLICY_SHA256,
    "total": len(RESULTS),
    "passed": sum(row["passed"] for row in RESULTS),
    "failed": sum(not row["passed"] for row in RESULTS),
    "results": RESULTS,
}
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument("--json")
args = parser.parse_args(argv)
if args.json:
    output = Path(args.json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
print("PALM_MATH_RESULT=" + json.dumps(payload, ensure_ascii=False))
raise SystemExit(1 if payload["failed"] else 0)
