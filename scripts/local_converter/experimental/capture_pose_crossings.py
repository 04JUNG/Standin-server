"""Record whole-mesh sampled intersection witnesses for one proposed pose repair."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.compare_posed_intersections import overlaps
from scripts.local_converter.experimental.contact_features import crossing_edges


def capture(stage,candidate,output):
    with np.load(stage) as raw, np.load(candidate) as change:
        start,faces=raw["corrected"],raw["triangles"]
        delta=np.zeros_like(start)
        delta[change["ids"]]=change["delta"]
    extent=np.ptp(start,axis=0).max()
    baseline=overlaps(start,faces)
    introduced=set()
    rows=[]
    for fraction in (.25,.5,.75,1.):
        points=start+fraction*delta
        observed=overlaps(points,faces)
        added=observed-baseline
        proper=[pair for pair in added if crossing_edges(points[faces[pair[0]]],
                                                          points[faces[pair[1]]],extent) is not None]
        introduced.update(proper)
        rows.append(dict(fraction=fraction,newPairs=len(added),newProper=len(proper),
                         removedPairs=len(baseline-observed)))
    output.write_text(json.dumps(dict(stageSHA=hashlib.sha256(stage.read_bytes()).hexdigest(),
        candidateSHA=hashlib.sha256(candidate.read_bytes()).hexdigest(),samples=rows,
        pairs=sorted(introduced)),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("stage","candidate","output"): parser.add_argument(name,type=Path)
    capture(**vars(parser.parse_args(sys.argv[sys.argv.index("--")+1:])))
