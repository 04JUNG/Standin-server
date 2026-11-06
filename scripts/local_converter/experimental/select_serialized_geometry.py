"""Select between two existing FBX paths using their reimported geometry."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.contact import contact_guard
from scripts.local_converter.experimental.deformation_guard import DeformationGuard


def select(stage, patch, capture, source, candidate):
    decision = dict(applied=False, selected=str(source))
    with np.load(stage) as raw, np.load(capture) as measured, np.load(patch) as expected:
        if any(hashlib.sha256(path.read_bytes()).hexdigest() != str(measured[key])
               for path,key in ((source,"sourceSHA"),(candidate,"candidateSHA"))):
            return dict(decision, reason="file_hash_mismatch")
        # The posed FBX may choose a different diagonal inside an authored
        # quad than its neutral mesh. Compare the two posed tessellations.
        if not np.array_equal(measured["faces"], measured["afterFaces"]):
            return dict(decision, reason="triangulation_changed")
        start, end = measured["before"], measured["after"]
        if start.shape != raw["rest"].shape or end.shape != start.shape:
            return dict(decision, reason="vertex_count_changed")
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            return dict(decision, reason="invalid_geometry")
        if np.linalg.norm(start-raw["corrected"],axis=1).max() >= 2e-5:
            return dict(decision, reason="baseline_mismatch")
        fixed = np.ones(len(start),dtype=bool)
        fixed[expected["ids"]] = False
        faces = measured["faces"]
        guard = DeformationGuard(raw["rest"],start,faces,fixed | raw["anchors"])
        _, groups, counts = np.unique(raw["rest"],axis=0,return_inverse=True,return_counts=True)
        guard.fixed |= guard.graph.fixed | (counts[groups.ravel()]>1)
        guard.upper = np.maximum(1.,guard.reference["stretch"])
        guard.edge_upper = np.maximum(1.,guard.reference["edge"])
        guard.crease_limit = np.maximum(1e-7, abs(guard.angle_difference(guard.reference["bend"],guard.rest_bend)))
        before = guard.reference["edge"]
        after = guard.measure(end)["edge"]
        decision.update(peakBefore=float(before.max()), peakAfter=float(after.max()),
                        severeBefore=int((before > 2).sum()), severeAfter=int((after > 2).sum()),
                        violations=guard.violations(end))
        if decision["violations"]:
            return dict(decision, reason="geometry_guard")
        if after.max()>before.max()+1e-7 or not (after.max()<=.99*before.max() or
                                                  decision["severeAfter"]<decision["severeBefore"]):
            return dict(decision, reason="insufficient_improvement")
        if not guard.trajectory_safe(start,end):
            return dict(decision, reason="unsafe_trajectory")
        decision["contact"] = contact_guard(start,end,faces)
        if not decision["contact"]["passCheck"]:
            return dict(decision, reason="unsafe_contact")
        return dict(decision,applied=True,selected=str(candidate),reason="geometry_passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("stage","patch","capture","source","candidate","out"):
        parser.add_argument(name,type=Path)
    args = vars(parser.parse_args())
    output = args.pop("out")
    try:
        result = select(**args)
    except (ValueError, OSError, KeyError, IndexError, np.linalg.LinAlgError) as error:
        result = dict(applied=False,selected=str(args["source"]),reason="unverified",detail=str(error))
    output.write_text(json.dumps(result, indent=2))
