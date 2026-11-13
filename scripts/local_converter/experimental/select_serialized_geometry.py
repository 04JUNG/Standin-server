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


def oriented_faces(faces):
    # Cyclic reindexing and changes in triangle iteration order preserve the
    # rendered topology. Reversing winding or flipping a quad diagonal does not.
    order=(faces.argmin(1)[:,None]+np.arange(3))%3
    triples=np.take_along_axis(faces,order,axis=1)
    keys=np.ascontiguousarray(triples,dtype="<i8").view([("a","<i8"),("b","<i8"),("c","<i8")])
    return np.sort(keys.ravel())


def measured_peak(rest, points, faces):
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    a,b=edges.T
    lengths=np.linalg.norm(rest[a]-rest[b],axis=1)
    if np.any(lengths<=0):
        raise ValueError("Degenerate rest edge")
    ratio=np.linalg.norm(points[a]-points[b],axis=1)/lengths
    highest=int(ratio.argmax())
    return float(ratio[highest]),edges[highest]


def select(stage, patch, capture, source, candidate, certify_sliding=False):
    decision = dict(applied=False, selected=str(source))
    with np.load(stage) as raw, np.load(capture) as measured, np.load(patch) as expected:
        if any(hashlib.sha256(path.read_bytes()).hexdigest() != str(measured[key])
               for path,key in ((source,"sourceSHA"),(candidate,"candidateSHA"))):
            return dict(decision, reason="file_hash_mismatch")
        # The posed FBX may choose a different diagonal inside an authored
        # quad than its neutral mesh. Compare the two posed tessellations.
        start, end = measured["before"], measured["after"]
        if start.shape != raw["rest"].shape or end.shape != start.shape:
            return dict(decision, reason="vertex_count_changed")
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            return dict(decision, reason="invalid_geometry")
        if not np.array_equal(measured["faces"],measured["afterFaces"]):
            base,trial=(oriented_faces(measured[key]) for key in ("faces","afterFaces"))
            if not np.array_equal(base,trial):
                before_peak,edge=measured_peak(raw["rest"],start,measured["faces"])
                after_peak,_=measured_peak(raw["rest"],end,measured["afterFaces"])
                a,b=edge
                target=np.linalg.norm(end[a]-end[b])/np.linalg.norm(raw["rest"][a]-raw["rest"][b])
                return dict(decision,reason="triangulation_changed",peakBefore=before_peak,peakAfter=after_peak,
                            originalPeakEdgeAfter=float(target),removedTriangles=len(np.setdiff1d(base,trial)),
                            addedTriangles=len(np.setdiff1d(trial,base)))
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
        measured = guard.measure(end)
        after = measured["edge"]
        smooth = abs(guard.rest_bend) < np.pi/6
        initial_angle = abs(guard.reference["bend"])
        final_angle = abs(measured["bend"])
        new_sharp = int((smooth & (initial_angle<np.pi/3) & (final_angle>=np.pi/3)).sum())
        new_folds = int((smooth & (initial_angle<np.pi/2) & (final_angle>=np.pi/2)).sum())
        decision.update(peakBefore=float(before.max()), peakAfter=float(after.max()),
                        severeBefore=int((before > 2).sum()), severeAfter=int((after > 2).sum()),
                        newSharp=new_sharp, newFolds=new_folds, violations=guard.violations(end))
        if decision["violations"] or new_sharp or new_folds:
            return dict(decision, reason="geometry_guard")
        if after.max()>before.max()+1e-7 or not (after.max()<=.99*before.max() or
                                                  decision["severeAfter"]<decision["severeBefore"]):
            return dict(decision, reason="insufficient_improvement")
        if not guard.trajectory_safe(start,end):
            return dict(decision, reason="unsafe_trajectory")
        decision["contact"] = contact_guard(start,end,faces,sliding=certify_sliding)
        if not decision["contact"]["passCheck"]:
            return dict(decision, reason="unsafe_contact")
        return dict(decision,applied=True,selected=str(candidate),reason="geometry_passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("stage","patch","capture","source","candidate","out"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--certify-sliding",action="store_true")
    args = vars(parser.parse_args())
    output = args.pop("out")
    try:
        result = select(**args)
    except (ValueError, OSError, KeyError, IndexError, np.linalg.LinAlgError) as error:
        result = dict(applied=False,selected=str(args["source"]),reason="unverified",detail=str(error))
    output.write_text(json.dumps(result, indent=2))
