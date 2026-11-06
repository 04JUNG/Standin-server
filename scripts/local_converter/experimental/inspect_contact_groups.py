"""Find connected sets of vertices participating in non-neighbor contacts."""
import sys
from pathlib import Path
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from compare_posed_intersections import overlaps


def groups(points,faces,pairs):
    parent=np.arange(len(points))
    def root(v):
        while parent[v]!=v:
            parent[v]=parent[parent[v]]
            v=parent[v]
        return v
    for left,right in pairs:
        vertices=faces[[left,right]].ravel()
        first=root(vertices[0])
        for v in vertices[1:]:
            parent[root(v)]=first
    return np.array([root(i) for i in range(len(points))])


def inspect(source,out):
    raw=np.load(source)
    rest,posed,faces=(raw[k] for k in ("rest","corrected","triangles"))
    edges=np.unique(np.sort(np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    a,b=edges.T
    ratio=np.linalg.norm(posed[a]-posed[b],axis=1)/np.linalg.norm(rest[a]-rest[b],axis=1)
    seed=edges[ratio.argmax()]
    pairs=overlaps(posed,faces)
    labels=groups(posed,faces,pairs)
    np.savez_compressed(out.with_suffix(".npz"),labels=labels)
    row=dict(totalPairs=len(pairs),targetEdge=seed.tolist(),
             sameGroup=bool(labels[seed[0]]==labels[seed[1]]),
             groupSizes=[int((labels==labels[v]).sum()) for v in seed])
    out.write_text(json.dumps(row,indent=2))
    print(row)


if __name__=="__main__":
    inspect(*map(Path,sys.argv[sys.argv.index("--")+1:]))
