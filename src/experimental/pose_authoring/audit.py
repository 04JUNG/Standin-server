"""Independent file-level audit: elbow hinge direction, forearm roll, wrist angles, overlaps.

Measures from the BVH alone (not the authoring spec)."""
import sys, json
from pathlib import Path
import numpy as np
from .author import Author
from .presets import stand, merge
from . import anatomy as A
from src.bvh import parse_bvh

def U(v):
    return v / (np.linalg.norm(v) + 1e-9)

def _rest_refs():
    au = Author(); au.build(merge(stand(), {"hands": {"L": "fist", "R": "fist"}}))
    rot, pos = au.sk.world(); ix = au.sk.idx
    refs = {}
    for s in ("Left", "Right"):
        Rh = rot[ix(s + "Hand")]
        mid = pos[ix(s + "HandMiddle1")] - pos[ix(s + "Hand")]
        tip = pos[ix(s + "HandMiddle3_End")] - pos[ix(s + "HandMiddle1")]
        n = U(tip - U(mid) * (tip @ U(mid)))
        refs[s + "palm"] = Rh.T @ n
    au2 = Author(); au2.build(stand()); rot2, pos2 = au2.sk.world()
    for s in ("Left", "Right"):
        refs[s + "uarm_rest"] = rot2[ix(s + "Arm")]
    return refs

REFS = _rest_refs()

def measure_file(path):
    au = Author(); joints, frames = parse_bvh(str(path)); au.sk.load_frame(frames[0])
    rot, pos = au.sk.world(); ix = au.sk.idx
    out = {}
    for s, k in (("Left", "L"), ("Right", "R")):
        sh, el, wr = pos[ix(s + "Arm")], pos[ix(s + "ForeArm")], pos[ix(s + "Hand")]
        u, f = U(el - sh), U(wr - el)
        ant = rot[ix(s + "Arm")] @ REFS[s + "uarm_rest"].T @ np.array([0, 0, 1.0])
        ant = U(ant - u * (ant @ u))
        lat = np.cross(u, ant)
        flex = np.degrees(np.arctan2(f @ ant, f @ u))
        off = np.degrees(np.arcsin(np.clip(f @ lat, -1, 1)))
        h = U(pos[ix(s + "HandMiddle1")] - wr)
        t = U(pos[ix(s + "HandIndex1")] - pos[ix(s + "HandPinky1")])
        n = rot[ix(s + "Hand")] @ REFS[s + "palm"]
        wflex = np.degrees(np.arcsin(np.clip(-(f @ n), -1, 1)))
        wrad = np.degrees(np.arcsin(np.clip(-(f @ t), -1, 1)))
        # forearm roll: thumb direction vs biceps side, both projected off the forearm axis
        # forearm roll vs the side facing the upper arm (biceps side); only meaningful when the elbow is bent
        bend = np.degrees(np.arccos(np.clip(u @ f, -1, 1)))
        a = -u - f * (-u @ f)
        if np.linalg.norm(a) > 0.2:
            tp, ap = U(t - f * (t @ f)), U(a)
            roll = np.degrees(np.arctan2(np.cross(ap, tp) @ f, ap @ tp)) * (1 if k == "L" else -1)
        else:
            roll = 0.0
        out[k] = {"elbow_flex": round(float(flex), 1), "elbow_off_plane": round(float(off), 1),
                  "wrist_flex": round(float(wflex), 1), "wrist_rad": round(float(wrad), 1),
                  "forearm_roll": round(float(roll), 1), "bend": round(float(bend), 1)}
    lm = A.landmarks(au.sk.joints, pos); lm["head_rot"] = rot[ix("Head")]
    prims = A.primitives(lm)
    _, allpairs = A.self_collisions(prims)
    return out, prims, allpairs

if __name__ == "__main__":
    print(json.dumps(measure_file(sys.argv[1])[0], indent=1))
