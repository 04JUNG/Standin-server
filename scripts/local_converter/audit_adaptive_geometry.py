"""Independent before/after counts for one saved positional candidate."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from converter.local import surface
from scripts.local_converter.experimental.deformation_guard import DeformationGuard


def audit(source, candidate, output):
    raw = np.load(source)
    change = np.load(candidate)
    rest, posed, faces = (raw[key] for key in ("rest","corrected","triangles"))
    corrected = posed.copy()
    corrected[change["ids"]] += change["delta"]
    fixed = np.ones(len(rest), dtype=bool)
    fixed[change["ids"]] = False
    guard = DeformationGuard(rest,posed,faces,fixed)
    left,right = guard.graph.face_pairs.T
    _,normals = surface.triangle_geometry(rest,faces)
    smooth = (normals[left]*normals[right]).sum(1)>np.cos(np.pi/6)

    def flags(points):
        values = guard.measure(points)
        _,normal = surface.triangle_geometry(points,faces)
        cosine = (normal[left]*normal[right]).sum(1)
        return dict(overDouble=values["edge"]>2+1e-7,
                    compressed=values["edge"]<.2-1e-7,
                    sharp=smooth&(cosine<.5),folded=smooth&(cosine<0),
                    collapsed=values["area"]<.01-1e-7)

    before,after = flags(posed),flags(corrected)
    stats = {key:dict(before=int(before[key].sum()),after=int(after[key].sum()),
                      new=int((after[key]&~before[key]).sum())) for key in before}
    a,b = guard.measure(posed),guard.measure(corrected)
    original_error = np.abs(guard.angle_difference(a["bend"],guard.rest_bend))
    corrected_error = np.abs(guard.angle_difference(b["bend"],guard.rest_bend))
    stats["geometry"] = dict(
        maxMoveMm=float(surface.lengths(corrected-posed).max()*1000),
        outsideMoveMm=float(surface.lengths(corrected[fixed]-posed[fixed]).max()*1000),
        edgeLengthIncreasedCount=int((b["edge"]>a["edge"]+1e-7).sum()),
        alreadyStretchedWorsened=int(((a["edge"]>1)&(b["edge"]>a["edge"]+1e-7)).sum()),
        newlyStretched=int(((a["edge"]<=1)&(b["edge"]>1+1e-7)).sum()),
        maxCreaseDeviationIncreaseDegrees=(float(np.degrees(
            (corrected_error-original_error)[guard.crease]).max()) if guard.crease.any() else None),
        originalCreasesOutsideTenDegrees=int((guard.crease&(corrected_error>np.pi/18)).sum()))
    stats["sourceGeometrySha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    output.write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("candidate",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    audit(args.source,args.candidate,args.output)
