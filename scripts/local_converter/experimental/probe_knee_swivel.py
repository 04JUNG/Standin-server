"""Measure pelvis relief with fixed hip and ankle positions and bounded knee drift."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.deformation_guard import DeformationGuard


def probe(stage,palette,output):
    with np.load(stage) as raw, np.load(palette) as capture:
        r,p,f=raw["rest"],raw["solver"],raw["triangles"]
        g=DeformationGuard(r,p,f,raw["anchors"])
        transforms,w=capture["maps"],capture["weights"]
        heads,parents,names=capture["heads"],capture["parents"],capture["names"]
    g.upper=np.maximum(1.,g.reference["stretch"])
    g.edge_upper=np.maximum(1.,g.reference["edge"])
    g.crease_limit=np.maximum(1e-7,abs(g.angle_difference(g.reference["bend"],g.rest_bend)))
    peak=g.reference["edge"].argmax()
    a,b=g.graph.a[peak],g.graph.b[peak]
    edge=p[b]-p[a]
    rest=np.column_stack((r,np.ones(len(r))))
    maps=np.einsum("vb,bij->vij",w,transforms[:,:3,:],optimize=True)
    predicted=np.einsum("vij,vj->vi",maps,rest)
    error=float(np.linalg.norm(predicted-p,axis=1).max())
    if error>2e-5: raise ValueError("Palette cannot reproduce the target pose")
    fields=[]
    for i in np.flatnonzero(w[[a,b]].any(axis=0)):
        child=np.flatnonzero(parents==i)
        if len(child)!=1: continue
        j=child.item()
        ankle=np.flatnonzero(parents==j)
        if len(ankle)!=1: continue
        k=ankle.item()
        h,knee,foot=heads[[i,j,k]]
        thigh=knee-h
        thigh=thigh/np.linalg.norm(thigh)
        swivel=foot-h
        swivel=swivel/np.linalg.norm(swivel)
        offsets=[w[:,idx,None]*(rest@transforms[idx,:3,:].T-h) for idx in (i,j)]
        fields.append(dict(name=str(names[i]),thigh=thigh,swivel=swivel,
                           radius=float(np.linalg.norm(np.cross(knee-h,swivel))),
                           upper=offsets[0],lower=offsets[1]))
    if len(fields)!=2: raise ValueError("Expected two independent contributing limb chains")
    rows=[]
    for cap_mm in (0.,5.,15.,30.,None):
        grid=[]
        bounds=[]
        for field in fields:
            upper=field["upper"][b]-field["upper"][a]
            lower=field["lower"][b]-field["lower"][a]
            radius=field["radius"]
            cap=np.pi if cap_mm is None or radius<1e-12 else 2*np.arcsin(min(1.,cap_mm/(2000*radius)))
            twist=np.linspace(-np.pi,np.pi,360,endpoint=False)
            swivel=np.linspace(-cap,cap,121) if cap>0 else np.array([0.])
            twisted=Rotation.from_rotvec(twist[:,None]*field["thigh"]).apply(upper)
            matrices=Rotation.from_rotvec(swivel[:,None]*field["swivel"]).as_matrix()
            delta=(matrices[:,None]@(twisted+lower)[None,:,:,None])[:,:,:,0]-(upper+lower)
            grid.append((delta.reshape(-1,3),twist,swivel))
            twist_radius=np.linalg.norm(np.cross(upper,field["thigh"]))
            bounds.append(2*np.sin(np.pi/(2*len(twist)))*twist_radius+
                          2*np.sin(cap/(2*max(1,len(swivel)-1)))*(np.linalg.norm(upper)+np.linalg.norm(lower)))
        distances,matches=cKDTree(grid[1][0]).query(-edge-grid[0][0],workers=1)
        best=int(distances.argmin())
        q=p.copy()
        choices=[]
        for field,(_,twist,swivel),index in zip(fields,grid,(best,matches[best])):
            si,ti=divmod(int(index),len(twist))
            twist_rot=Rotation.from_rotvec(twist[ti]*field["thigh"])
            swivel_rot=Rotation.from_rotvec(swivel[si]*field["swivel"])
            offsets=twist_rot.apply(field["upper"])+field["lower"]
            q+=swivel_rot.apply(offsets)-(field["upper"]+field["lower"])
            choices.append(dict(joint=field["name"],twistDeg=float(np.degrees(twist[ti])),
                swivelDeg=float(np.degrees(swivel[si])),
                kneeDriftMm=float(2000*field["radius"]*abs(np.sin(swivel[si]/2)))))
        m=g.measure(q)["edge"]
        rows.append(dict(capMm=cap_mm,choices=choices,target=float(m[peak]),peak=float(m.max()),
            lowerBound=float(max(0.,distances[best]-sum(bounds))/g.lengths[peak]),
            newSevere=int(((g.reference["edge"]<=2)&(m>2+1e-7)).sum()),
            maxMoveMm=float(1000*np.linalg.norm(q-p,axis=1).max()),violations=g.violations(q)))
    output.write_text(json.dumps(dict(stage="raw skinning",before=float(g.reference["edge"][peak]),
        reconstructionErrorM=error,rows=rows),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("stage","palette","output"): parser.add_argument(name,type=Path)
    probe(**vars(parser.parse_args()))
