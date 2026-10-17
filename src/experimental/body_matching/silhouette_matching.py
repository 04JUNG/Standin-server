"""Compare isolated upright silhouettes; experimental, never auto-applies a body.

A validated binary person mask is required. This module does NOT segment arbitrary
rough drawings. A renderer-background extractor exists only in the evaluation tool.
The profile handles detached arm outlines, not general articulation or perspective.
"""
from __future__ import annotations
import numpy as np

VERSION="body-silhouette-profile.v1"
LEVELS=np.linspace(.03,.96,100)
WEIGHTS=np.ones(100)
# Shoulder/upper-arm row widths are very pose dependent in T versus lowered arms.
WEIGHTS[(LEVELS>.16)&(LEVELS<.32)]=0

def extract_profile(mask, *, full_body_visible, upright, foreshortening, ownership_verified):
    m=np.asarray(mask)
    if m.ndim!=2 or m.dtype!=np.bool_ or min(m.shape)<32:raise ValueError("invalid_silhouette_mask")
    if not full_body_visible or not upright or foreshortening or not ownership_verified:
        raise ValueError("unsupported_silhouette_input")
    ys,xs=np.where(m)
    if len(ys)<100:raise ValueError("insufficient_silhouette")
    top,bottom=int(ys.min()),int(ys.max());height=bottom-top
    if height<64 or top==0 or bottom==m.shape[0]-1 or xs.min()==0 or xs.max()==m.shape[1]-1:
        raise ValueError("cropped_or_small_silhouette")
    center=float(np.median(xs[ys<top+.15*height]));values=[];valid=[]
    for t in LEVELS:
        row=m[int(top+t*height)]
        bounds=np.flatnonzero(np.diff(np.r_[False,row,False]))
        runs=list(zip(bounds[::2],bounds[1::2]))
        if t<.55:
            run=min(runs,key=lambda p:abs((p[0]+p[1])/2-center)) if runs else (center,center)
            width=(run[1]-run[0])/height
            available=bool(runs) and abs((run[0]+run[1])/2-center)<height*.15
        else:
            runs=[p for p in runs if abs((p[0]+p[1])/2-center)<height*.16]
            width=sum(b-a for a,b in runs)/height;available=bool(runs)
        values.append(float(width));valid.append(available)
    if sum(valid)<90:raise ValueError("incomplete_silhouette_profile")
    return dict(version=VERSION,widths=values,valid=valid)

def rank_profiles(query, references):
    """Minimum over registered views per body; no sex/age/name/ground-truth filtering."""
    def read(p):
        if p.get("version")!=VERSION:raise ValueError("profile_version_mismatch")
        v=np.asarray(p.get("widths"),dtype=float);m=np.asarray(p.get("valid"))
        if v.shape!=(100,) or m.shape!=(100,) or m.dtype!=np.bool_ or not np.isfinite(v).all() or (v<0).any() or (v>3).any():
            raise ValueError("invalid_silhouette_profile")
        return v,m
    q,qm=read(query);by_id={};identities=set()
    for r in references:
        identity=(r["body_id"],r["view"])
        if identity in identities:raise ValueError("duplicate_reference_view")
        identities.add(identity);v,vm=read(r["profile"]);weights=WEIGHTS*qm*vm
        if (weights>0).sum()<60:continue
        score=float(np.average(abs(q-v),weights=weights))
        if r["body_id"] not in by_id or score<by_id[r["body_id"]]["distance"]:
            by_id[r["body_id"]]=dict(body_id=r["body_id"],view=r["view"],distance=score)
    ranked=sorted(by_id.values(),key=lambda r:(r["distance"],r["body_id"]))
    ties=[r["body_id"] for r in ranked if abs(r["distance"]-ranked[0]["distance"])<=1e-8]
    return dict(schema_version=VERSION,candidates=ranked,suggested_body_id=ranked[0]["body_id"] if ranked else None,
        tied_body_ids=ties if len(ties)>1 else [],auto_body_id=None,applied_body_id=None,
        acceptance_probability=None,application_policy=dict(allowed=False,reason="silhouette_input_and_articulation_not_calibrated_for_roughs"))
