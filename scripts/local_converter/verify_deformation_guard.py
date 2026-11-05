"""Replay real one-pose candidates through the baseline-relative guard."""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from converter.local.deformation_guard import DeformationGuard
from converter.local import surface

base = Path(sys.argv[1])
candidates = Path(sys.argv[2])
out = Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
data = np.load(base)
rest, posed, faces = (data[k] for k in ("rest", "corrected", "triangles"))
rows = []
for source in sorted(candidates.glob("*.npz")):
    change = np.load(source)
    fixed = np.ones(len(rest), dtype=bool)
    fixed[change["ids"]] = False
    guard = DeformationGuard(rest, posed, faces, fixed)
    assert not guard.violations(posed)
    proposal = posed.copy()
    proposal[change["ids"]] += change["delta"]
    corrected, report = guard.accept(posed, proposal)
    assert not guard.violations(corrected)
    assert np.array_equal(corrected[fixed], posed[fixed])
    g = guard.graph
    _, normals = surface.triangle_geometry(rest, faces)
    left, right = g.face_pairs.T
    smooth = (normals[left]*normals[right]).sum(1) > np.cos(np.pi/6)
    metrics = surface.metrics(corrected, faces, g, guard.lengths, guard.area, smooth)
    row = dict(candidate=source.stem, acceptance=report, metrics=metrics,
               targetMm=float(np.linalg.norm(corrected[28683]-corrected[28685])*1000),
               seamStatus="unclassified boundaries pinned by candidate generator",
               selfIntersectionStatus="not evaluated")
    rows.append(row)
    np.savez_compressed(out/source.name, ids=change["ids"],
                        delta=corrected[change["ids"]]-posed[change["ids"]])
    (out/"report.json").write_text(json.dumps(rows, indent=2))
    print(row, flush=True)
