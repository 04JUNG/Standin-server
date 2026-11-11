#!/usr/bin/env python3
"""Make a bounded derived skin plan from actual Blender FK inputs. Offline only."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    from converter.skin_profile_repair import VERSION, diffuse_arm_weights, correct_rest_surface
    from converter.deformation_witness import _skin_maps, _normals
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(a.inputs, allow_pickle=False) as packed:
        x = {k: packed[k] for k in packed.files}
    n, f, names = x["neutral"], x["faces"], list(x["names"])
    parents = json.loads(str(x["parents"]))
    weights, weight_report = diffuse_arm_weights(x["weights"], f, names)
    root = next(name for name in names if name.split(":")[-1] == "RightArm")
    active = {root}
    while True:
        expanded = active | {name for name in names if parents[name] in active}
        if expanded == active:
            break
        active = expanded
    indices = [i for i, name in enumerate(names) if name in active]
    pivot = x["body_bones"][names.index(root), :3, 3]
    normals = _normals(n, f)
    attempts = []
    for degree in (0., 3., 6., 9., 12.):
        rad = np.deg2rad(degree)
        delta = np.eye(4)
        delta[:3, :3] = [[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]]
        delta[:3, 3] = pivot-delta[:3, :3]@pivot
        body, hand = x["body_bones"].copy(), x["hand_bones"].copy()
        body[indices] = delta @ body[indices]
        hand[indices] = delta @ hand[indices]
        bm, bc, _ = _skin_maps(weights, body@np.linalg.inv(x["bones"]), len(n), "lbs")
        hm, hc, _ = _skin_maps(weights, hand@np.linalg.inv(body), len(n), "lbs")
        bd = np.einsum("fvij,fj->fvi", bc[f], normals)
        hd = np.einsum("fvij,fvj->fvi", hc[f], bd)
        try:
            corrected, report = correct_rest_surface(n, x["rest"], f,
                [("body", bm, bd), ("hands", hm@bm, hd)])
        except ValueError as error:
            attempts.append(dict(clearance_degrees=degree, accepted=False, reason=str(error)))
            (a.output_dir/"attempts.json").write_text(json.dumps(attempts, indent=2))
            continue
        attempts.append(dict(clearance_degrees=degree, accepted=True))
        np.savez(a.output_dir/"skin-plan.npz", version=VERSION,
            source_sha256=x["source_sha256"], asset_sha256=x["asset_sha256"],
            neutral=n, repaired_rest=x["rest"], faces=f, original_weights=x["weights"],
            weights=weights, corrected_rest=corrected, right_arm_clearance_degrees=degree,
            hand_left=x["hand_left"], hand_right=x["hand_right"])
        (a.output_dir/"plan-report.json").write_text(json.dumps(dict(
            attempts=attempts, weights=weight_report, corrective=report,
            final_receipt_issued=False, production_registered=False), indent=2))
        print("PLANNED", a.output_dir/"skin-plan.npz", "clearance", degree, flush=True)
        return
    raise ValueError("no_fully_validated_skin_plan")


if __name__ == "__main__":
    main()
