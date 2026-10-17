"""Rig-agnostic anatomy gate: joint range of motion, self-collision, support.

Works on joint positions only, so it can be calibrated against real motion
capture frames from any rig in the library. Thresholds are conservative
authoring guards for a 175 cm adult; they are not a clinical model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.bvh import parse_bvh, fk

# ------------------------------------------------------------------ limits
LIMITS = {
    # thorax relative to pelvis (deg)
    "spine_flex": 75, "spine_ext": 30, "spine_lat": 35, "spine_twist": 45,
    # shoulder (humerus in thorax frame)
    "sh_ext": 60,            # humerus behind the coronal plane
    "sh_int": 100, "sh_ext_rot": 110,   # humeral axial rotation
    "elbow_flex": 150,
    # hip (thigh in pelvis frame)
    "hip_flex": 130, "hip_ext": 35, "hip_abd": 50, "hip_add": 32, "hip_rot_ext": 80, "hip_rot_int": 50,
    "knee_flex": 155,
    "ankle_dorsi": 45, "ankle_plantar": 55,
    # neck (head relative to chest, rig channels only)
    "neck_yaw": 75, "neck_down": 55, "neck_up": 45, "neck_roll": 40,
    # authored forearm / wrist (spec values)
    "pron": 85, "wrist_flex": 70, "wrist_ext": 65, "wrist_ext_load": 80, "wrist_rad": 22, "wrist_uln": 32,
}
PENETRATION_TOL = 1.5    # cm of soft-tissue overlap tolerated at contacts
FLOOR_TOL = 1.0          # cm below the floor tolerated (sole compression)
BALANCE_TOL = 1.5        # cm COM may sit outside the contact hull

SUFFIX = {
    "hips": ["Hips"],
    "spine": ["Spine", "Chest", "LowerBack"],
    "spine_mid": ["Spine1", "Chest2", "Spine"],
    "chest": ["Spine2", "Chest4", "Chest3", "Spine1"],
    "neck": ["Neck"],
    "head": ["Head"],
    "head_end": ["Head_End", "HeadTop_End"],
}
SIDE = {
    "shoulder": ["{s}Arm", "{s}Shoulder"],
    "elbow": ["{s}ForeArm", "{s}Elbow"],
    "wrist": ["{s}Hand", "{s}Wrist"],
    "hip": ["{s}UpLeg", "{s}Hip"],
    "knee": ["{s}Leg", "{s}Knee"],
    "ankle": ["{s}Foot", "{s}Ankle"],
    "toe": ["{s}ToeBase", "{s}Toe"],
}


def _clean(name):
    return name.split(":")[-1]


def landmarks(joints, pos):
    names = {_clean(j[0]): i for i, j in enumerate(joints)}
    lm = {}

    def pick(cands):
        for c in cands:
            if c in names:
                return np.asarray(pos[names[c]], float)
        return None

    for key, cands in SUFFIX.items():
        lm[key] = pick(cands)
    for side in ("Left", "Right"):
        k = side[0]
        for key, cands in SIDE.items():
            # 100STYLE uses LeftShoulder for the humerus root; the
            # Mixamo-style rigs use LeftShoulder for the clavicle.
            if key == "shoulder" and (side + "Arm") not in names and (side + "Elbow") in names:
                cands = ["{s}Shoulder"]
            lm[k + key] = pick([c.format(s=side) for c in cands])
        tips = []
        for finger in ("Index", "Middle", "Ring", "Pinky", "Thumb"):
            chain = [names.get(f"{side}Hand{finger}{n}") for n in (1, 2, 3)]
            end = names.get(f"{side}Hand{finger}3_End")
            if None not in chain and end is not None:
                lm[f"{k}{finger}"] = [np.asarray(pos[i], float) for i in chain + [end]]
                tips.append(lm[f"{k}{finger}"][-1])
        lm[k + "has_fingers"] = bool(tips)
        if lm[k + "toe"] is not None:
            toe_end = names.get(f"{side}ToeBase_End") or names.get(f"{side}Toe_End")
            lm[k + "toe_end"] = np.asarray(pos[toe_end], float) if toe_end is not None else None
    if lm["head_end"] is None and lm["head"] is not None and lm["neck"] is not None:
        lm["head_end"] = lm["head"] + _u(lm["head"] - lm["neck"]) * 9.0
    return lm


def _u(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0


def _ang(a, b):
    return float(np.rad2deg(np.arccos(np.clip(_u(a) @ _u(b), -1, 1))))


def _frame(x, y_hint):
    x = _u(x)
    z = _u(np.cross(x, y_hint))
    y = np.cross(z, x)
    return np.column_stack((x, y, z))


def _rot_between(a, b):
    a, b = _u(a), _u(b)
    v = np.cross(a, b)
    s, c = np.linalg.norm(v), float(np.clip(a @ b, -1, 1))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = _u(np.cross(a, np.eye(3)[int(np.argmin(abs(a)))]))
        return 2 * np.outer(axis, axis) - np.eye(3)
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    th = np.arctan2(s, c)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _twist(u, d, rest_dir, rest_ref):
    """Signed angle (deg) of d about u, from rest_ref swung minimally rest_dir->u."""
    ref = _rot_between(rest_dir, u) @ rest_ref
    ref = _u(ref - u * (u @ ref))
    d = _u(d - u * (u @ d))
    return float(np.rad2deg(np.arctan2(u @ np.cross(ref, d), ref @ d)))


# ------------------------------------------------------------------ ROM
def body_frames(lm):
    hipmid = (lm["Lhip"] + lm["Rhip"]) / 2
    up_p = (lm["spine_mid"] if lm["spine_mid"] is not None else lm["chest"]) - hipmid
    P = _frame(lm["Lhip"] - lm["Rhip"], up_p)
    shmid = (lm["Lshoulder"] + lm["Rshoulder"]) / 2
    base = lm["spine_mid"] if lm["spine_mid"] is not None else hipmid
    T = _frame(lm["Lshoulder"] - lm["Rshoulder"], shmid - base)
    return P, T


def rom(lm, rest=None, rig=None):
    """Return metrics and violations. rest: per-rig neutral ankle angles.

    rig: optional world rotations {"hips","chest","head"} from the authoring
    rig (identity rest). When given, pelvis/thorax/head frames come from the
    real joint rotations, so pelvic tilt and neck rotation are measured too.
    """
    P, T = body_frames(lm)
    m, bad = {}, []
    if rig is not None:
        P, T = rig["hips"], rig["chest"]
        from scipy.spatial.transform import Rotation as _R
        yaw, pitch, nroll = _R.from_matrix(T.T @ rig["head"]).as_euler("YXZ", degrees=True)
        m["neck_yaw"], m["neck_pitch"], m["neck_roll"] = float(yaw), float(pitch), float(-nroll)
        if abs(yaw) > LIMITS["neck_yaw"]:
            bad.append(f"고개 과회전 {yaw:.0f}°")
        if pitch > LIMITS["neck_down"]:
            bad.append(f"고개 과숙임 {pitch:.0f}°")
        if -pitch > LIMITS["neck_up"]:
            bad.append(f"고개 과젖힘 {-pitch:.0f}°")
        if abs(nroll) > LIMITS["neck_roll"]:
            bad.append(f"고개 과기울임 {-nroll:.0f}°")
    # ---- spine
    rel = P.T @ T
    ty, tx = rel[:, 1], rel[:, 0]
    m["spine_flex"] = float(np.rad2deg(np.arctan2(ty[2], ty[1])))
    m["spine_lat"] = float(np.rad2deg(np.arcsin(np.clip(-ty[0], -1, 1))))
    m["spine_twist"] = float(np.rad2deg(np.arctan2(-tx[2], tx[0])))
    if m["spine_flex"] > LIMITS["spine_flex"]:
        bad.append(f"허리 과굴곡 {m['spine_flex']:.0f}°")
    if -m["spine_flex"] > LIMITS["spine_ext"]:
        bad.append(f"허리 과신전 {-m['spine_flex']:.0f}°")
    if abs(m["spine_lat"]) > LIMITS["spine_lat"]:
        bad.append(f"허리 옆굽힘 {m['spine_lat']:.0f}°")
    if abs(m["spine_twist"]) > LIMITS["spine_twist"]:
        bad.append(f"허리 비틀림 {m['spine_twist']:.0f}°")
    for k, side in (("L", 1), ("R", -1)):
        mir = np.diag([side, 1.0, 1.0])
        # ---- shoulder / elbow
        u = mir @ T.T @ _u(lm[k + "elbow"] - lm[k + "shoulder"])
        f = mir @ T.T @ _u(lm[k + "wrist"] - lm[k + "elbow"])
        flex = _ang(u, f)
        ext = float(np.rad2deg(np.arcsin(np.clip(-u[2], -1, 1))))
        m[k + "elbow_flex"], m[k + "sh_ext"] = flex, ext
        if flex > LIMITS["elbow_flex"]:
            bad.append(f"{k} 팔꿈치 과굴곡 {flex:.0f}°")
        if ext > LIMITS["sh_ext"]:
            bad.append(f"{k} 팔이 뒤로 과도하게 넘어감 {ext:.0f}°")
        if flex > 25:
            down = np.array([0, -1.0, 0])
            fwd = np.array([0, 0, 1.0])
            direct = _twist(u, f, down, fwd)
            via_abd = _twist(u, f, np.array([1.0, 0, 0]), fwd)
            via_flex = _twist(u, f, fwd, np.array([0, 1.0, 0]))
            tw = min((direct, via_abd, via_flex), key=abs)
            m[k + "sh_rot"] = tw
            if tw > LIMITS["sh_int"]:
                bad.append(f"{k} 어깨 안쪽 비틀림 과다 {tw:.0f}°")
            if -tw > LIMITS["sh_ext_rot"]:
                bad.append(f"{k} 어깨 바깥 비틀림 과다(팔꿈치 역방향) {-tw:.0f}°")
        # ---- hip / knee
        u = mir @ P.T @ _u(lm[k + "knee"] - lm[k + "hip"])
        f = mir @ P.T @ _u(lm[k + "ankle"] - lm[k + "knee"])
        kflex = _ang(u, f)
        hflex = float(np.rad2deg(np.arctan2(u[2], -u[1])))
        habd = float(np.rad2deg(np.arcsin(np.clip(u[0], -1, 1))))
        m[k + "knee_flex"], m[k + "hip_flex"], m[k + "hip_abd"] = kflex, hflex, habd
        if kflex > LIMITS["knee_flex"]:
            bad.append(f"{k} 무릎 과굴곡 {kflex:.0f}°")
        if hflex > LIMITS["hip_flex"]:
            bad.append(f"{k} 고관절 과굴곡(허리 접힘) {hflex:.0f}°")
        if -hflex > LIMITS["hip_ext"] and u[1] < 0.2:
            bad.append(f"{k} 다리가 뒤로 과도하게 넘어감 {-hflex:.0f}°")
        if habd > LIMITS["hip_abd"]:
            bad.append(f"{k} 다리 과외전 {habd:.0f}°")
        if -habd > LIMITS["hip_add"]:
            bad.append(f"{k} 다리 과내전 {-habd:.0f}°")
        if kflex > 25:
            back = np.array([0, 0, -1.0])
            direct = _twist(u, f, np.array([0, -1.0, 0]), back)
            via_flex = _twist(u, f, np.array([0, 0, 1.0]), np.array([0, -1.0, 0]))
            tw = min((direct, via_flex), key=abs)
            m[k + "hip_rot"] = tw
            if -tw > LIMITS["hip_rot_ext"] or tw > LIMITS["hip_rot_int"]:
                bad.append(f"{k} 고관절 비틀림 과다(무릎 방향 이상) {tw:.0f}°")
        # ---- ankle
        if lm.get(k + "toe") is not None:
            g = lm[k + "toe"] - lm[k + "ankle"]
            shin = lm[k + "ankle"] - lm[k + "knee"]
            neutral = (rest or {}).get(k + "ankle", 59.0)
            a = _ang(shin, g)
            dorsi = neutral - a
            m[k + "ankle_dorsi"] = dorsi
            if dorsi > LIMITS["ankle_dorsi"]:
                bad.append(f"{k} 발목 과배굴 {dorsi:.0f}°")
            if -dorsi > LIMITS["ankle_plantar"]:
                bad.append(f"{k} 발목 과저굴 {-dorsi:.0f}°")
    return {k: round(v, 2) for k, v in m.items()}, bad


# ------------------------------------------------------------------ body model
@dataclass
class Prim:
    part: str
    a: np.ndarray
    b: np.ndarray
    ra: float
    rb: float
    group: str = ""   # limb chain id for adjacency exclusion
    tags: tuple = field(default_factory=tuple)


def primitives(lm):
    P, T = body_frames(lm)
    hipmid = (lm["Lhip"] + lm["Rhip"]) / 2
    shmid = (lm["Lshoulder"] + lm["Rshoulder"]) / 2
    mid = lm["spine_mid"] if lm["spine_mid"] is not None else (hipmid + shmid) / 2
    chest = lm["chest"]
    neck = lm["neck"]
    head = lm["head"]
    up_h = _u(lm["head_end"] - head)
    if lm.get("head_rot") is not None:
        up_h = lm["head_rot"][:, 1]
        fwd_h = lm["head_rot"][:, 2]
    else:
        fwd_h = _u(np.cross(T[:, 0], up_h))
    prims = [
        Prim("pelvis", lm["Lhip"] + P[:, 0] * 1.5, lm["Rhip"] - P[:, 0] * 1.5, 9.5, 9.5, "trunk"),
        # gluteal mass: seat contact sits behind and below the hip joints
        Prim("glute", lm["Lhip"] + P[:, 0] * 0.5 - P[:, 2] * 5.5 - P[:, 1] * 1.0,
             lm["Rhip"] - P[:, 0] * 0.5 - P[:, 2] * 5.5 - P[:, 1] * 1.0, 9.5, 9.5, "trunk"),
        Prim("belly", hipmid + P[:, 1] * 2, mid, 11.5, 11.5, "trunk"),
        Prim("chest", mid, chest + (neck - chest) * 0.35, 12.0, 12.0, "trunk"),
        Prim("chest_w", chest + (lm["Lshoulder"] - chest) * 0.55 - T[:, 1] * 4,
             chest + (lm["Rshoulder"] - chest) * 0.55 - T[:, 1] * 4, 7.5, 7.5, "trunk"),
        Prim("neck", neck, head, 5.2, 5.0, "neck"),
        Prim("head", head + up_h * 7.5 + fwd_h * 1.5, head + up_h * 7.5 + fwd_h * 1.5, 10.0, 10.0, "head"),
        Prim("jaw", head + up_h * 2.5 + fwd_h * 5.0, head + up_h * 2.5 + fwd_h * 5.0, 6.0, 6.0, "head"),
        Prim("nose", head + up_h * 6.5 + fwd_h * 10.5, head + up_h * 6.5 + fwd_h * 10.5, 1.8, 1.8, "head"),
    ]
    for k in ("L", "R"):
        prims += [
            Prim(k + "uarm", lm[k + "shoulder"], lm[k + "elbow"], 4.6, 3.9, k + "arm"),
            Prim(k + "farm", lm[k + "elbow"], lm[k + "wrist"], 3.7, 2.7, k + "arm"),
        ]
        if lm[k + "has_fingers"]:
            base = (lm[k + "Index"][0] + lm[k + "Pinky"][0]) / 2
            prims.append(Prim(k + "palm", lm[k + "wrist"], base, 2.6, 2.3, k + "arm", ("hand",)))
            for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
                pts = lm[k + finger]
                r = 1.0 if finger == "Thumb" else 0.85
                start = lm[k + "wrist"] + (pts[0] - lm[k + "wrist"]) * (0.5 if finger == "Thumb" else 1.0)
                chain = [start] + pts[1:]
                for n in range(len(chain) - 1):
                    prims.append(Prim(f"{k}{finger}{n}", chain[n], chain[n + 1], r, r * 0.9,
                                      k + "arm", ("hand", "finger")))
        else:
            fdir = _u(lm[k + "wrist"] - lm[k + "elbow"])
            prims.append(Prim(k + "palm", lm[k + "wrist"], lm[k + "wrist"] + fdir * 15, 2.6, 2.0,
                              k + "arm", ("hand",)))
        prims += [
            Prim(k + "thigh", lm[k + "hip"], lm[k + "knee"], 7.8, 5.2, k + "leg"),
            Prim(k + "shin", lm[k + "knee"], lm[k + "ankle"], 5.0, 3.4, k + "leg"),
        ]
        if lm.get(k + "toe") is not None:
            toe, ank = lm[k + "toe"], lm[k + "ankle"]
            tip = lm.get(k + "toe_end")
            tip = toe + (tip - toe) * 0.45 if tip is not None else toe + _u(toe - ank) * 6
            # heel: behind/below ankle, perpendicular to the foot
            g = _u(toe - ank)
            lat = _u(np.cross(g, lm[k + "knee"] - ank))
            sole = _u(np.cross(lat, g))  # points from dorsum toward the sole
            if sole @ (lm[k + "knee"] - ank) > 0:
                sole = -sole
            heel = ank + sole * 8.9 + g * 0.75
            prims += [
                Prim(k + "foot", heel, toe, 2.4, 1.5, k + "leg", ("foot",)),
                Prim(k + "dorsum", ank, toe, 3.2, 1.5, k + "leg", ("foot",)),
                Prim(k + "toes", toe, tip, 1.4, 1.1, k + "leg", ("foot",)),
            ]
    return prims


def _sample(p: Prim, n):
    t = np.linspace(0, 1, n)[:, None]
    return p.a + (p.b - p.a) * t, (p.ra + (p.rb - p.ra) * t[:, 0])


def _pair_depth(p, q, n=14, skip_p=0.0, skip_q=0.0):
    pa, ra = _sample(p, n)
    qa, rq = _sample(q, n)
    tp = np.linspace(0, 1, n)
    keep_p = tp >= skip_p
    keep_q = tp >= skip_q
    pa, ra = pa[keep_p], ra[keep_p]
    qa, rq = qa[keep_q], rq[keep_q]
    d = np.linalg.norm(pa[:, None, :] - qa[None, :, :], axis=-1) - ra[:, None] - rq[None, :]
    i = np.unravel_index(np.argmin(d), d.shape)
    return float(-d[i]), (pa[i[0]], qa[i[1]])


# Pairs never tested (anatomically joined or always overlapping)
_TRUNK = {"pelvis", "glute", "belly", "chest", "chest_w", "neck"}


def _skip(p: Prim, q: Prim):
    """Return (skip, fraction of p to ignore, fraction of q to ignore)."""
    if p.group and p.group == q.group:
        names = {p.part[1:], q.part[1:]}
        # same limb: only check hand vs upper arm and foot vs thigh
        if names & {"uarm"} and ("hand" in p.tags or "hand" in q.tags):
            return False, 0.0, 0.0
        if names & {"thigh"} and ("foot" in p.tags or "foot" in q.tags):
            return False, 0.0, 0.0
        return True, 0, 0
    parts = {p.part, q.part}
    if p.group == "trunk" and q.group == "trunk":
        return True, 0, 0
    if {"neck"} & parts and ({"head", "jaw"} & parts or parts & _TRUNK):
        return True, 0, 0
    if p.group == "head" and q.group == "head":
        return True, 0, 0
    for a, b in ((p, q), (q, p)):
        if a.part[1:] == "uarm" and b.part in ("chest", "chest_w", "neck"):
            return (False, 0.55, 0.0) if a is p else (False, 0.0, 0.55)
        if a.part[1:] == "uarm" and b.part in ("belly", "pelvis", "glute"):
            return (False, 0.3, 0.0) if a is p else (False, 0.0, 0.3)
        if a.part[1:] == "thigh" and b.part in ("pelvis", "belly", "glute"):
            return (False, 0.6, 0.0) if a is p else (False, 0.0, 0.6)
    if {p.part[1:], q.part[1:]} == {"thigh"}:
        return False, 0.35, 0.35
    return False, 0.0, 0.0


# Muscle-on-muscle contacts compress more than skin on bone.
SOFT_PAIRS = {"Luarm~chest_w": 3.5, "Ruarm~chest_w": 3.5, "Luarm~chest": 2.0, "Ruarm~chest": 2.0}


def self_collisions(prims, tol=PENETRATION_TOL, detail=False):
    worst = {}
    for i, p in enumerate(prims):
        for q in prims[i + 1:]:
            skip, sp, sq = _skip(p, q)
            if skip:
                continue
            depth, where = _pair_depth(p, q, skip_p=sp, skip_q=sq)
            key = _pair_key(p, q)
            if depth > worst.get(key, (-1e9,))[0]:
                worst[key] = (depth, where, p, q)
    hits = {k: round(v[0], 2) for k, v in worst.items() if v[0] > SOFT_PAIRS.get(k, tol)}
    if detail:
        return hits, worst
    return hits, {k: round(v[0], 2) for k, v in worst.items()}


def _coarse(part):
    for key in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        if key in part:
            return part[0] + "hand"
    if part[1:] == "palm":
        return part[0] + "hand"
    if part[1:] in ("foot", "dorsum", "toes"):
        return part[0] + "foot"
    return part


def _pair_key(p, q):
    a, b = sorted((_coarse(p.part), _coarse(q.part)))
    return f"{a}~{b}"


def cross_collisions(prims_a, prims_b, tol=PENETRATION_TOL, detail=False):
    worst = {}
    for p in prims_a:
        for q in prims_b:
            depth, where = _pair_depth(p, q, n=10)
            key = "A." + _coarse(p.part) + "~B." + _coarse(q.part)
            if depth > worst.get(key, (-1e9,))[0]:
                worst[key] = (depth, where, p, q)
    hits = {k: round(v[0], 2) for k, v in worst.items() if v[0] > tol}
    closest = max(v[0] for v in worst.values()) if worst else None
    if detail:
        return hits, worst
    return hits, closest


# ------------------------------------------------------------------ support
SEG_MASS = [  # (fraction, endpoints by landmark names)
    (0.081, ("neck", "head_end")), (0.497, ("hipmid", "shmid")),
    (0.028, ("Lshoulder", "Lelbow")), (0.028, ("Rshoulder", "Relbow")),
    (0.016, ("Lelbow", "Lwrist")), (0.016, ("Relbow", "Rwrist")),
    (0.006, ("Lwrist", "Lwrist")), (0.006, ("Rwrist", "Rwrist")),
    (0.100, ("Lhip", "Lknee")), (0.100, ("Rhip", "Rknee")),
    (0.0465, ("Lknee", "Lankle")), (0.0465, ("Rknee", "Rankle")),
    (0.0145, ("Lankle", "Ltoe")), (0.0145, ("Rankle", "Rtoe")),
]


def center_of_mass(lm):
    pts = dict(lm)
    pts["hipmid"] = (lm["Lhip"] + lm["Rhip"]) / 2
    pts["shmid"] = (lm["Lshoulder"] + lm["Rshoulder"]) / 2
    total, acc = 0.0, np.zeros(3)
    for w, (a, b) in SEG_MASS:
        if pts.get(a) is None or pts.get(b) is None:
            continue
        acc += w * (pts[a] + pts[b]) / 2
        total += w
    return acc / total


def lowest(prims):
    return min(min(p.a[1] - p.ra, p.b[1] - p.rb) for p in prims)


def contact_points(prims, height=0.0, band=3.5):
    pts = []
    for p in prims:
        a, r = _sample(p, 9)
        bottom = a[:, 1] - r
        for xyz, y in zip(a, bottom):
            if abs(y - height) <= band:
                pts.append([xyz[0], xyz[2]])
    return np.array(pts)


def _hull(points):
    pts = sorted(map(tuple, points))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def hull_distance(point, hull):
    """Signed distance from 2D point to convex polygon (negative inside)."""
    if len(hull) == 0:
        return 1e9
    if len(hull) == 1:
        return float(np.linalg.norm(point - hull[0]))
    best = 1e9
    inside = len(hull) >= 3
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        ab = b - a
        t = np.clip((point - a) @ ab / max(ab @ ab, 1e-9), 0, 1)
        best = min(best, float(np.linalg.norm(point - (a + t * ab))))
        if inside and (ab[0] * (point[1] - a[1]) - ab[1] * (point[0] - a[0])) < 0:
            inside = False
    return -best if inside else best


def support(lm, prims, extra=None):
    """COM over the floor-contact hull plus declared external supports."""
    pts = contact_points(prims, 0.0)
    if extra:
        ext = np.asarray(extra, float).reshape(-1, 2)
        pts = ext if len(pts) == 0 else np.vstack([pts, ext])
    com = center_of_mass(lm)
    hull = _hull(pts) if len(pts) else np.zeros((0, 2))
    return {
        "com_xz": [round(float(com[0]), 2), round(float(com[2]), 2)],
        "com_y": round(float(com[1]), 2),
        "hull_distance_cm": round(hull_distance(com[[0, 2]], hull), 2),
        "contacts": int(len(pts)),
    }


# ------------------------------------------------------------------ helpers
def load(path):
    joints, frames = parse_bvh(str(path))
    pos = fk(joints, frames[0])
    return joints, landmarks(joints, pos)


def rest_ankles(path):
    joints, frames = parse_bvh(str(path))
    zero = np.zeros_like(frames[0])
    pos = fk(joints, zero)
    lm = landmarks(joints, pos)
    out = {}
    for k in ("L", "R"):
        if lm.get(k + "toe") is not None:
            out[k + "ankle"] = _ang(lm[k + "ankle"] - lm[k + "knee"], lm[k + "toe"] - lm[k + "ankle"])
    return out


def stature_ratio(lm, reference=None):
    """Pose-independent size from segment lengths, relative to the authoring rig."""
    def seg(lm):
        hipmid = (lm["Lhip"] + lm["Rhip"]) / 2
        shmid = (lm["Lshoulder"] + lm["Rshoulder"]) / 2
        return (np.linalg.norm(lm["Lhip"] - lm["Lknee"]) + np.linalg.norm(lm["Lknee"] - lm["Lankle"])
                + np.linalg.norm(shmid - hipmid) + np.linalg.norm(lm["Lshoulder"] - lm["Lelbow"])
                + np.linalg.norm(lm["Lelbow"] - lm["Lwrist"]))
    ref = reference if reference is not None else 189.0
    return seg(lm) / ref


def scaled(lm, factor):
    out = {}
    for k, v in lm.items():
        if isinstance(v, np.ndarray):
            out[k] = v / factor
        elif isinstance(v, list):
            out[k] = [p / factor for p in v]
        else:
            out[k] = v
    return out


# ------------------------------------------------------------------ props
def box_signed_distance(points, lo, hi):
    """Signed distance (negative inside) and outward normal for an AABB."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c, h = (lo + hi) / 2, (hi - lo) / 2
    q = np.abs(points - c) - h
    outside = np.linalg.norm(np.maximum(q, 0), axis=-1)
    inside = np.minimum(q.max(-1), 0)
    sd = outside + inside
    # normal: dominant axis of q (outside) or nearest face (inside)
    k = np.argmax(q, axis=-1)
    normal = np.zeros_like(points)
    sign = np.sign(points - c)
    normal[np.arange(len(points)), k] = sign[np.arange(len(points)), k]
    return sd, normal


def prop_collisions(prims, props, tol=PENETRATION_TOL, detail=False):
    worst = {}
    for prop in props:
        for p in prims:
            pts, r = _sample(p, 12)
            sd, nrm = box_signed_distance(pts, prop["min"], prop["max"])
            pen = r - sd
            i = int(np.argmax(pen))
            key = _coarse(p.part) + "~" + prop["kind"]
            if pen[i] > worst.get(key, (-1e9,))[0]:
                worst[key] = (float(pen[i]), (pts[i], pts[i] - nrm[i]), p, None, nrm[i])
    hits = {k: round(v[0], 2) for k, v in worst.items() if v[0] > tol}
    return (hits, worst) if detail else (hits, {k: round(v[0], 2) for k, v in worst.items()})


def prop_contacts(prims, props, band=2.0):
    """XZ points where the body rests on top of a prop (supporting surfaces)."""
    pts = []
    for prop in props:
        if not prop.get("supports", True):
            continue
        lo, hi = np.asarray(prop["min"], float), np.asarray(prop["max"], float)
        for p in prims:
            a, r = _sample(p, 9)
            for xyz, rr in zip(a, r):
                inside_xz = lo[0] - 1 <= xyz[0] <= hi[0] + 1 and lo[2] - 1 <= xyz[2] <= hi[2] + 1
                if inside_xz and abs((xyz[1] - rr) - hi[1]) <= band:
                    pts.append([xyz[0], xyz[2]])
                # vertical faces (walls) take load through the contact point
                if prop.get("wall") and lo[1] - 1 <= xyz[1] <= hi[1] + 1:
                    sd, _ = box_signed_distance(xyz[None], lo, hi)
                    if sd[0] - rr <= band:
                        pts.append([xyz[0], xyz[2]])
    return pts
