"""Critical relational pose evidence and preview-only camera comparison.

All coordinates come from existing library BVHs, never from VLM output. The
experimental thresholds describe broad relations, not pose correctness scores.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from .semantic_body_index import records_digest

SHAPE_VERSION = "semantic-shapes-v1"
SHAPE_THRESHOLDS = {
    "v_min_up_torso": 0.3, "v_min_out_torso": 0.25,
    "v_open_min_deg": 40.0, "v_open_max_deg": 150.0,
    "fold_min_chest_height": 0.4, "fold_max_chest_height": 1.2,
    "fold_cross_margin_width": -0.05,
}
CAMERAS = tuple({"view": view, "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": 0}
                for view, yaw in (("front", 0), ("three_quarter", 45), ("side", 90), ("back", 180))
                for pitch in (0, 30, 60, 80, -30, -60))


def critical_relations(query) -> set[str]:
    text = re.sub(r"\s+", "", (query.body_details or "") + "." + (query.text or ""))
    needed = set()
    # Negated/uncertain phrases do not become hard requirements.
    negative = r"않|아니|불명|모르|가려|불확실|추정"
    fold_denied = re.search(r"팔짱[^.,;]{0,18}(?:" + negative + r")", text)
    folded = re.search(r"팔짱|가슴앞.{0,12}양팔.{0,8}교차", text)
    if not fold_denied and ("fold_arms" in query.actions or (folded and not re.search(
            negative, re.split(r"[.,;]", text[folded.end():])[0][:18]))):
        needed.add("fold_arms")
    for match in re.finditer(r"(?:양팔|두팔|팔)[^.,;]{0,12}(?:[Vv]자|[Vv]형|브이자)|(?:[Vv]자|[Vv]형|브이자)[^.,;]{0,12}(?:양팔|두팔|팔)", text):
        if not re.search(negative, re.split(r"[.,;]", text[match.end():])[0][:18]):
            needed.add("v_arms")
    return needed


def camera_hint(query) -> str | None:
    text = query.body_details + " " + query.text + " " + getattr(query, "context_evidence", "")
    if re.search(r"위에서\s*내려다|부감|하이앵글", text) and not re.search(r"(?:부감|하이앵글).{0,8}(?:아니|불명|불확실)", text):
        return "high_angle"
    if re.search(r"아래에서\s*올려다|앙각|로우앵글", text) and not re.search(r"(?:앙각|로우앵글).{0,8}(?:아니|불명|불확실)", text):
        return "low_angle"
    return None


def rotate_for_camera(joints, camera):
    """Y-up orthographic camera rotation; not a saved pose modification."""
    yaw, pitch, roll = np.deg2rad([camera[k] for k in ("yaw_deg", "pitch_deg", "roll_deg")])
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    rx = np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])
    rz = np.array([[cr,-sr,0],[sr,cr,0],[0,0,1]])
    return np.asarray(joints, dtype=float) @ (rz @ rx @ ry).T


def _frame(points):
    shoulder = (points[5] + points[6]) / 2
    hip = (points[11] + points[12]) / 2
    up = shoulder - hip
    length = float(np.linalg.norm(up))
    if length < 1e-8:
        return None
    up /= length
    lateral = points[6] - points[5]
    lateral -= np.dot(lateral, up) * up
    width = float(np.linalg.norm(lateral))
    if width < 1e-8:
        return None
    return shoulder, hip, up, lateral / width, length, width


def relational_geometry(points, scores) -> dict:
    p=np.asarray(points,dtype=float); s=np.asarray(scores,dtype=float)
    if p.shape!=(17,3) or s.shape!=(17,) or not np.isfinite(p).all() or not np.isfinite(s).all():
        raise ValueError("invalid library COCO17")
    if not (s[[5,6,7,8,9,10,11,12]] >= .3).all():
        return {"fold_geometry":None,"v_arms":None}
    frame=_frame(p)
    if frame is None:
        return {"fold_geometry":None,"v_arms":None}
    sh,hip,up,right,length,width=frame
    heights=[float(np.dot(p[i]-hip,up)/length) for i in (9,10)]
    crossed=(np.dot(p[9]-sh,right)/width >= SHAPE_THRESHOLDS["fold_cross_margin_width"]
             and np.dot(p[10]-sh,right)/width <= -SHAPE_THRESHOLDS["fold_cross_margin_width"])
    chest=all(SHAPE_THRESHOLDS["fold_min_chest_height"] <= h <= SHAPE_THRESHOLDS["fold_max_chest_height"] for h in heights)
    arms=[p[9]-p[5],p[10]-p[6]]
    norms=[float(np.linalg.norm(a)) for a in arms]
    if min(norms)<1e-8:
        return {"fold_geometry":bool(crossed and chest),"v_arms":None}
    opening=float(np.degrees(np.arccos(np.clip(np.dot(*arms)/np.prod(norms),-1,1))))
    upwards=[float(np.dot(a,up)/length) for a in arms]
    outward=[float(-np.dot(arms[0],right)/length),float(np.dot(arms[1],right)/length)]
    v=(min(upwards)>=SHAPE_THRESHOLDS["v_min_up_torso"]
       and min(outward)>=SHAPE_THRESHOLDS["v_min_out_torso"]
       and SHAPE_THRESHOLDS["v_open_min_deg"] <= opening <= SHAPE_THRESHOLDS["v_open_max_deg"])
    return {"fold_geometry":bool(crossed and chest),"v_arms":bool(v),
            "hand_chest_heights":[round(h,5) for h in heights],
            "v_upward":[round(h,5) for h in upwards],"v_outward":[round(h,5) for h in outward],
            "arm_opening_deg":round(opening,5)}


def projected_relations(joints, scores, camera) -> dict:
    p=rotate_for_camera(joints,camera)[:,:2]
    if not (np.asarray(scores)[[5,6,9,10,11,12]] >= .3).all():
        return {"v_visible":False,"v_score":0.,"fold_visible":False,"fold_score":0.}
    sh=(p[5]+p[6])/2; hip=(p[11]+p[12])/2
    length=float(np.linalg.norm(sh-hip)); width=float(np.linalg.norm(p[6]-p[5]))
    if min(length,width)<1e-8:
        return {"v_visible":False,"v_score":0.,"fold_visible":False,"fold_score":0.}
    sign=1 if p[6,0]>=p[5,0] else -1
    up=[float((p[h,1]-p[a,1])/length) for a,h in ((5,9),(6,10))]
    out=[float(-sign*(p[9,0]-p[5,0])/length),float(sign*(p[10,0]-p[6,0])/length)]
    v=bool(min(up)>=.2 and min(out)>=.2 and sh[1]-hip[1]>.1*length)
    # Reward visibility of BOTH arms, not one arm's larger extent.
    vs=float(min(1.,min(up),min(out))) if v else 0.
    cross=[float(sign*(p[9,0]-sh[0])/width),float(-sign*(p[10,0]-sh[0])/width)]
    folded=bool(min(cross)>=-.05 and sh[1]-hip[1]>.1*length)
    fs=float(min(1.,max(0.,min(cross)+.05))) if folded else 0.
    return {"v_visible":v,"v_score":round(vs,5),"fold_visible":folded,"fold_score":round(fs,5)}


def choose_camera(profile, required, hint=None):
    views=profile["views"]
    key="v" if "v_arms" in required else "fold"
    eligible=[v for v in views if all(v[("v" if r=="v_arms" else "fold")+"_visible"] for r in required)]
    if hint=="high_angle":
        hinted=[v for v in eligible if v["camera"]["pitch_deg"]>0]
        eligible=hinted or eligible
    elif hint=="low_angle":
        hinted=[v for v in eligible if v["camera"]["pitch_deg"]<0]
        eligible=hinted or eligible
    if not eligible:
        return None
    # Avoid maximizing a torso-normalized ratio through extreme foreshortening.
    # Preserve a modest view when the relation is already visible. This is NOT
    # an image-fitting score, and cannot resolve full-body/view ambiguities.
    return min(eligible,key=lambda v:(abs(v["camera"]["pitch_deg"]),-v[key+"_score"],v["camera"]["yaw_deg"]))


def load_shape_index(path, records):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version")!=SHAPE_VERSION or data.get("thresholds")!=SHAPE_THRESHOLDS or data.get("cameras")!=list(CAMERAS):
        raise ValueError("semantic shape index version/config mismatch")
    if data.get("tags_sha256")!=records_digest(records) or data.get("frame")!=0:
        raise ValueError("semantic shape corpus/frame mismatch")
    poses=data.get("poses")
    if not isinstance(poses,dict) or not poses or not set(poses).issubset({r["pose_id"] for r in records}):
        raise ValueError("invalid semantic shape pose IDs")
    for value in poses.values():
        relations=value.get("relations",{})
        if set(relations)!={"fold_arms","v_arms"} or any(v is not None and type(v) is not bool for v in relations.values()):
            raise ValueError("invalid semantic relation values")
        views=value.get("views")
        if not isinstance(views,list) or len(views)!=len(CAMERAS):
            raise ValueError("invalid camera entries")
        for view,camera in zip(views,CAMERAS):
            if view.get("camera")!=camera:
                raise ValueError("invalid camera configuration")
            for key in ("v_visible","fold_visible"):
                if type(view.get(key)) is not bool: raise ValueError("invalid camera flag")
            for key in ("v_score","fold_score"):
                x=view.get(key)
                if isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or not 0<=x<=1:
                    raise ValueError("invalid camera score")
        digest=value.get("source_sha256","")
        if not re.fullmatch(r"[a-f0-9]{64}",digest): raise ValueError("invalid source hash")
    return poses
