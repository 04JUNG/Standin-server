"""Enumerate endpoint intersection candidates with Blender's triangle BVH."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from mathutils.bvhtree import BVHTree


def overlaps(points,faces):
    tree=BVHTree.FromPolygons(points.tolist(),faces.tolist(),all_triangles=True)
    pairs=np.asarray(tree.overlap(tree),int).reshape(-1,2)
    pairs=pairs[pairs[:,0]<pairs[:,1]]
    left,right=faces[pairs[:,0]],faces[pairs[:,1]]
    separate=~(left[:,:,None]==right[:,None,:]).any(axis=(1,2))
    return set(map(tuple,pairs[separate].tolist()))


def compare(source,output):
    raw=np.load(source)
    rest,before,after,faces=(raw[k] for k in ("rest","baked","corrected","triangles"))
    base=overlaps(before,faces)
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    a,b=edges.T
    lengths=np.linalg.norm(rest[a]-rest[b],axis=1)
    rows=[]
    for fraction in (0,.0625,.125,.25,.5,.75,1.):
        points=before+fraction*(after-before)
        observed=overlaps(points,faces) if fraction else base
        added=sorted(observed-base)
        rows.append(dict(fraction=fraction,maxStretch=float(
            (np.linalg.norm(points[a]-points[b],axis=1)/lengths).max()),
            existing=len(base),newPairs=len(added),removedPairs=len(base-observed),
            newPairIds=added))
    # Retain the proposed positions away from implicated faces, and recheck
    # all pairs since fixing a face can change intersections at its neighbors.
    points = after.copy()
    attempts = []
    for iteration in range(32):
        observed = overlaps(points,faces)
        introduced = observed-base
        attempts.append(dict(iteration=iteration,newPairs=len(introduced),
                             removedPairs=len(base-observed),
                             moved=int(np.any(points!=before,axis=1).sum())))
        if not introduced:
            break
        locked = np.unique(faces[np.array(sorted(introduced),int)].ravel())
        points[locked] = before[locked]
    if introduced:
        points = before.copy()
    moved = np.flatnonzero(np.any(points!=before,axis=1))
    np.savez_compressed(output.with_suffix(".npz"),ids=moved,delta=points[moved]-before[moved])
    clipped_ratio = np.linalg.norm(points[a]-points[b],axis=1)/lengths
    output.write_text(json.dumps(dict(strengths=rows,clipping=attempts,
        finalMaxStretch=float(clipped_ratio.max()),aboveDouble=int((clipped_ratio>2).sum()),
        unresolved=bool(introduced)),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    compare(args.source,args.output)
