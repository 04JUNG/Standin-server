"""Deterministic math controls for CHAIN_TRANSPORT_V3_2_PELVIS_BOUNDARY.

Run with Blender.  This imports only the QA variant and does not touch production files.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import json
import math
import os

from mathutils import Matrix, Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-root", required=True)
    parser.add_argument("--json", required=True)
    return parser.parse_args(argv)


def angle(a, b):
    dot = max(-1.0, min(1.0, a.normalized().dot(b.normalized())))
    return math.degrees(math.acos(dot))


def main():
    args = parse_args()
    root = os.path.abspath(args.variant_root)
    sys.path.insert(0, root)
    import converter.retarget as rt

    if not os.path.abspath(rt.__file__).startswith(root + os.sep):
        raise SystemExit("[FAIL] QA variant 밖 retarget import")

    def solve(parent, dst, src):
        h, reason = rt._min_rotation(parent.to_3x3() @ dst, src)
        return (None, reason) if h is None else (h @ parent, "")

    cases = []

    dst = Vector((0.2, 0.94, -0.1)).normalized()
    src = Vector((-0.55, 0.7, 0.45)).normalized()
    identity = Matrix.Identity(4)
    q_v31, why = solve(identity, dst, src)
    q_identity, why2 = solve(identity, dst, src)
    err = rt._rotation_error_deg(q_v31, q_identity)
    cases.append({"name": "identity_parent_equals_v31", "error_deg": err,
                  "pass": not why and not why2 and err <= 1e-6})

    parent = (Matrix.Rotation(math.radians(52), 4, Vector((0.3, 0.8, 0.5)).normalized())
              @ Matrix.Rotation(math.radians(-27), 4, Vector((0.9, -0.1, 0.2)).normalized()))
    transported = parent.to_3x3() @ dst
    q_exact, why = solve(parent, dst, transported)
    exact_err = rt._rotation_error_deg(q_exact, parent)
    cases.append({"name": "already_parent_transport_is_exact", "error_deg": exact_err,
                  "pass": not why and exact_err <= 1e-6})

    q_parent, why = solve(parent, dst, src)
    direction_error = angle(q_parent.to_3x3() @ dst, src)
    correction = q_parent @ q_v31.inverted()
    correction_deg = rt._rotation_error_deg(Matrix.Identity(4), correction)
    twist_deg = rt._signed_twist_deg(correction, src)
    non_twist = abs(correction_deg - abs(twist_deg))
    cases.append({"name": "same_direction_pure_boundary_twist",
                  "direction_error_deg": direction_error,
                  "correction_deg": correction_deg,
                  "twist_deg": twist_deg,
                  "non_twist_residual_deg": non_twist,
                  "pass": not why and direction_error <= 0.05 and non_twist <= 1e-4})

    mirror = Matrix.Diagonal((-1.0, 1.0, 1.0, 1.0))
    mirrored_parent = mirror @ parent @ mirror
    mirrored_dst = mirror.to_3x3() @ dst
    mirrored_src = mirror.to_3x3() @ src
    q_mirror, why = solve(mirrored_parent, mirrored_dst, mirrored_src)
    expected_mirror = mirror @ q_parent @ mirror
    mirror_error = rt._rotation_error_deg(q_mirror, expected_mirror)
    cases.append({"name": "mirror_covariance", "error_deg": mirror_error,
                  "pass": not why and mirror_error <= 1e-5})

    opposite_axis = dst.cross(Vector((0, 0, 1)))
    if opposite_axis.length < 1e-6:
        opposite_axis = dst.cross(Vector((1, 0, 0)))
    opposite = (Matrix.Rotation(math.radians(176), 4, opposite_axis.normalized())
                .to_3x3() @ dst)
    degenerate, reason = solve(identity, dst, opposite)
    cases.append({"name": "over_175_is_degenerate", "reason": reason,
                  "pass": degenerate is None and "175" in reason})

    h_zero, zero_reason = rt._min_rotation(Vector((0, 0, 0)), dst)
    cases.append({"name": "zero_direction_is_degenerate", "reason": zero_reason,
                  "pass": h_zero is None})

    cmu = rt._ankle_transport_amount(141.72, "cmu_bvh")
    cases.append({"name": "v31_ankle_hard_guard_preserved", "amount": cmu,
                  "pass": cmu["selected_mu"] == 0.0
                          and cmu["reason"] == "frozen_v3_hard_guard"})

    soft = rt._ankle_transport_amount(96.0, "mixamo_noprefix")
    cases.append({"name": "v31_ankle_soft_cap_preserved", "amount": soft,
                  "pass": 0.0 < soft["selected_mu"] < 1.0
                          and soft["reason"] == "spherical_soft_cap"})

    payload = {
        "variant": rt._QA_VARIANT_NAME,
        "parent_variant": rt._QA_PARENT_VARIANT,
        "parent_retarget_sha256": rt._QA_PARENT_RETARGET_SHA256,
        "production_retarget_loaded": False,
        "cases": cases,
        "ok": all(row["pass"] for row in cases),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(("[OK]" if payload["ok"] else "[FAIL]") +
          f" V3.2 parent-coherent math {sum(x['pass'] for x in cases)}/{len(cases)}")
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
