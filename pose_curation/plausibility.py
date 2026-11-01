"""Anatomical plausibility of a static BVH pose: two-tier ROM limits + correction.

Skeleton-agnostic. Each segment's anatomical frame is measured once from the
rest pose's joint positions and then carried by that joint's world rotation,
so Mixamo, CMU, 100STYLE and other hierarchies share one set of angles.

Frames use columns (X, Y, Z) with Y = toward the proximal joint and
Z = anterior (body forward at rest). Torso frames use X = character left.

Two tiers per measure (degrees):
  normal  = clinical active ROM (AAOS; Soucie et al. 2011; WA DSHS 13-585A)
  extreme = normal plus a 15-30 degree margin for athletic/comic poses
Outside normal -> "hold"; outside extreme -> "reject". Shoulder and hip use a
direction-dependent swing limit (reach cone, Wilhelms & Van Gelder 2001) plus a
twist interval (swing-twist, Baerlocher & Boulic 2001). Correction clamps into
the normal tier while descendants keep their local rotations.
Fingers are not checked here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from src.collision import (arm_leg_penetration, arm_torso_penetration,
                           leg_leg_penetration, leg_torso_penetration)

VERSION = "plausibility-2026-11-02.5"
SIDES = {"left": ("Left", 1.0), "right": ("Right", -1.0)}
TOLERANCE = 2.0          # degrees; measurement noise allowance before flagging
COLLISION_HOLD = 0.025   # torso lengths (pose_curation.qa.policy.CAPSULE_DEPTH)
COLLISION_REJECT = 0.08

# name: (label, normal (lo, hi), extreme (lo, hi), source)
LIMITS = {
    "trunk_flexion": ("몸통 굽힘(+)/젖힘(-)", (-30, 85), (-55, 115), "AAOS 80/25, DSHS 90/25"),
    "trunk_lateral": ("몸통 옆굽힘", (-35, 35), (-55, 55), "AAOS 35"),
    "trunk_rotation": ("몸통 비틀기", (-45, 45), (-70, 70), "AAOS 45"),
    "neck_flexion": ("목 숙임(+)/젖힘(-)", (-60, 50), (-80, 70), "DSHS 50/60"),
    "neck_lateral": ("목 옆굽힘", (-45, 45), (-60, 60), "AAOS/DSHS 45"),
    "neck_rotation": ("목 돌림", (-80, 80), (-95, 95), "DSHS 80"),
    "shoulder_posterior": ("팔 뒤로 뻗기", None, None, "AAOS ext 60, horiz. abd ~45"),
    "shoulder_girdle": ("팔을 높이 올렸는데 어깨가 안 올라감", None, None,
                        "scapulohumeral rhythm (Inman 1944): ~1 deg clavicle per 4 deg above 90"),
    "shoulder_medial": ("팔 몸 가로지르기", None, None, "AAOS add 40, horiz. add ~135"),
    "humeral_rotation": ("위팔 비틀림", (-90, 90), (-120, 120), "AAOS IR 70/ER 90"),
    "elbow_flexion": ("팔꿈치 굽힘(+)/과신전(-)", (-5, 150), (-15, 162), "Soucie 2011 145-150, hyperext 0.8-4.7"),
    "elbow_offaxis": ("팔꿈치 옆꺾임", (-15, 15), (-30, 30), "hinge; carrying angle 10-15"),
    "forearm_rotation": ("아래팔 회내/회외", (-90, 90), (-120, 120), "Soucie 2011 77-91"),
    "wrist_extension": ("손목 젖힘(+)/굽힘(-)", (-80, 70), (-95, 90), "AAOS 80/70"),
    "wrist_deviation": ("손목 좌우 꺾임", (-30, 35), (-40, 50), "DSHS radial 20/ulnar 30"),
    "hip_cone": ("다리 들기 범위", None, None, "AAOS flex 120, abd 40-45"),
    "hip_posterior": ("다리 뒤로 뻗기", None, None, "AAOS ext 20-30"),
    "hip_medial": ("다리 안쪽 모으기", None, None, "AAOS add 20-30"),
    "hip_rotation": ("고관절 비틀림", (-50, 50), (-75, 75), "Roaas 1982 IR 20-50/ER 10-55"),
    "knee_flexion": ("무릎 굽힘(+)/과신전(-)", (-5, 150), (-15, 165), "Soucie 2011 138-142, Roaas -10"),
    "knee_offaxis": ("무릎 옆꺾임", (-12, 12), (-25, 25), "hinge; varus/valgus"),
    "knee_rotation": ("정강이 비틀림", (-35, 35), (-55, 55), "tibial rotation ~30 flexed"),
    "ankle_plantar": ("발목 내림(+)/올림(-)", (-20, 55), (-35, 85),
                      "Soucie 2011 plantar 55-62; dorsi 20 knee straight, ~40 knee bent (Bennell 1998 lunge)"),
    "ankle_eversion": ("발목 바깥(+)/안쪽(-) 꺾임", (-35, 20), (-50, 35), "DSHS inv 30/ev 20"),
    "ankle_toe_out": ("발끝 돌림", (-30, 30), (-50, 50), "foot abd/add"),
}

# Shoulder swing: limit (normal, extreme) as a function of elevation from down.
SHOULDER_POSTERIOR = ((0, 60, 80), (90, 45, 65), (135, 20, 40), (180, 10, 25))
SHOULDER_MEDIAL_FRONT = ((0, 40, 55), (90, 50, 70), (180, 35, 55))
SHOULDER_MEDIAL_BACK = ((0, 20, 40), (180, 20, 40))
# Scapulohumeral rhythm: expected clavicle elevation above 90 deg of arm elevation.
GIRDLE_RATIO = 0.25
GIRDLE_SHORTFALL = 8.0  # degrees of missing clavicle elevation before "hold"
GIRDLE_MAX_LIFT = 30.0
# Clavicle angles are reported relative to rest but never range-checked: the
# rest clavicle is not anatomically neutral across sources (ACCAD rests ~25 deg
# above its usual pose, 100STYLE ~10, SD ~-18), so only the rhythm rule uses it.
# Hip swing: max elevation at lateral (abduction) and forward (flexion).
HIP_ABDUCTION = (50, 100)
HIP_FLEXION = (125, 170)          # knee bent; hamstrings cap a straight leg lower
HIP_FLEXION_STRAIGHT = (100, 155)  # straight-leg raise
# Twist about a raised limb is path dependent (Codman's paradox); above this
# elevation the twist reference is unreliable, so it is neither checked nor clamped.
TWIST_MAX_ELEVATION = 120
HIP_POSTERIOR = (25, 55)
HIP_MEDIAL = (30, 50)
# de Leva (1996) male segment mass fractions and CoM position from proximal end.
DE_LEVA = {"head": (.0694, 1.0), "trunk": (.4346, .4862), "upperarm": (.0271, .5772),
           "forearm": (.0162, .4574), "hand": (.0061, .5), "thigh": (.1416, .4095),
           "shank": (.0433, .4395), "foot": (.0137, .5)}


# ---------------------------------------------------------------- rotations
def _axis_rot(axis, deg):
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    if axis == "X": return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y": return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_matrix(order, angles):
    R = np.eye(3)
    for axis, angle in zip(order, angles):
        R = R @ _axis_rot(axis, angle)
    return R


def euler_angles(R, order):
    """R = R_a(α) R_b(β) R_c(γ) for a Tait-Bryan order such as 'ZXY' → degrees."""
    i, j, k = ("XYZ".index(a) for a in order)
    sign = 1.0 if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1.0
    beta = math.asin(max(-1.0, min(1.0, sign * R[i, k])))
    if abs(R[i, k]) < 1 - 1e-9:
        alpha = math.atan2(-sign * R[j, k], R[k, k])
        gamma = math.atan2(-sign * R[i, j], R[i, i])
    else:  # gimbal: fold everything into alpha
        alpha = math.atan2(sign * R[k, j], R[j, j]); gamma = 0.0
    return np.degrees([alpha, beta, gamma])


def rotation_between(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if c < -1 + 1e-9:
        axis = np.cross(a, [1, 0, 0] if abs(a[0]) < .9 else [0, 1, 0])
        return axis_angle(axis, 180.0)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def axis_angle(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    r = math.radians(deg)
    return np.eye(3) + math.sin(r) * K + (1 - math.cos(r)) * K @ K


def fractional(R, t):
    """R**t through axis-angle (used to spread a spine correction)."""
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
    if angle < 1e-6:
        return np.eye(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.linalg.norm(axis) < 1e-9:  # 180 degrees
        w, v = np.linalg.eigh((R + np.eye(3)) / 2); axis = v[:, -1]
    return axis_angle(axis, angle * t)


def swing_twist(M):
    """Twist (deg) of frame M about its own Y after the minimal swing from +Y."""
    Q = rotation_between(np.array([0.0, 1.0, 0.0]), M[:, 1])
    T = Q.T @ M
    return math.degrees(math.atan2(T[0, 2], T[0, 0]))


# ---------------------------------------------------------------- skeleton
class Skeleton:
    """Joint roles resolved by name suffix; rest frames measured from offsets."""

    def __init__(self, joints):
        self.joints = joints
        self.names = [j[0].split(":")[-1] for j in joints]
        self.parent = [j[1] for j in joints]
        self.children = [[] for _ in joints]
        for i, p in enumerate(self.parent):
            if p >= 0: self.children[p].append(i)
        find = lambda *names: next((self.names.index(n) for n in names if n in self.names), -1)
        r = {"pelvis": find("Hips", "Hip", "Root")}
        r["neck"] = find("Neck")
        r["head"] = find("Head")
        r["chest"] = self.parent[r["neck"]] if r["neck"] >= 0 else -1
        for side, (S, _) in SIDES.items():
            short = S[0]
            if S + "Collar" in self.names:
                arm = (S + "Collar", S + "Shoulder", S + "Elbow", S + "Wrist")
            else:
                arm = (S + "Shoulder", S + "Arm", S + "ForeArm", S + "Hand")
            r[side + "_clavicle"], r[side + "_upperarm"], r[side + "_forearm"], r[side + "_hand"] = (find(n) for n in arm)
            r[side + "_thigh"] = find(S + "UpLeg", S + "Hip")
            r[side + "_shin"] = find(S + "Leg", S + "Knee")
            r[side + "_foot"] = find(S + "Foot", S + "Ankle")
            r[side + "_toe"] = find(S + "ToeBase", S + "Toe")
            sfx = "_" + side[0]
            r[side + "_handtip"] = find(S + "HandMiddle1", "middle_01" + sfx, S + "FingerBase",
                                        S + "HandIndex1", "index_01" + sfx)
            if r[side + "_handtip"] < 0 and r[side + "_hand"] >= 0 and self.children[r[side + "_hand"]]:
                r[side + "_handtip"] = self.children[r[side + "_hand"]][0]
        self.role = r
        required = ["pelvis", "chest", "neck", "head"] + [
            f"{s}_{p}" for s in SIDES for p in ("upperarm", "forearm", "hand", "thigh", "shin", "foot")]
        self.missing = [k for k in required if r.get(k, -1) < 0]
        if self.missing:
            return
        self.spine = self._chain(r["pelvis"], r["chest"])          # excludes pelvis
        self.neck_chain = self._chain(r["chest"], r["head"])
        zero = np.zeros(sum(len(j[3]) for j in joints))
        # Rest pose = all channels zero with offsets (rotation identity).
        self.rest_pos, _, _ = fk_world(joints, zero, use_offsets=True)
        self.rest_frames = self._rest_frames()

    def _chain(self, top, bottom):
        chain = [bottom]
        while self.parent[chain[-1]] not in (top, -1):
            chain.append(self.parent[chain[-1]])
        return chain[::-1]

    def _rest_frames(self):
        p, r = self.rest_pos, self.role
        left = p[r["left_thigh"]] - p[r["right_thigh"]]; left /= np.linalg.norm(left)
        up = p[r["neck"]] - p[r["pelvis"]]; up -= left * (up @ left); up /= np.linalg.norm(up)
        fwd = np.cross(left, up)
        self.rest_body = np.column_stack([left, up, fwd])
        self.rest_clavicle = {}
        for side in SIDES:
            if r.get(side + "_clavicle", -1) >= 0 and r[side + "_clavicle"] != r["chest"]:
                v = self.rest_body.T @ (p[r[side + "_upperarm"]] - p[r[side + "_clavicle"]])
                self.rest_clavicle[side] = v / np.linalg.norm(v)
        frames = {k: self.rest_body for k in ("pelvis", "chest", "head")}
        for side in SIDES:
            for seg, child in (("upperarm", "forearm"), ("forearm", "hand"),
                               ("thigh", "shin"), ("shin", "foot")):
                y = p[r[f"{side}_{seg}"]] - p[r[f"{side}_{child}"]]
                y /= np.linalg.norm(y)
                if seg in ("forearm", "shin"):
                    # Share the hinge axis with the proximal segment so a bent
                    # rest pose reads as pure flexion, not as off-axis bending.
                    x = frames[f"{side}_{'upperarm' if seg == 'forearm' else 'thigh'}"][:, 0]
                    x = x - y * (x @ y); x /= np.linalg.norm(x)
                    frames[f"{side}_{seg}"] = np.column_stack([x, y, np.cross(x, y)])
                    continue
                ref = fwd if abs(fwd @ y) < .95 else up
                z = ref - y * (ref @ y); z /= np.linalg.norm(z)
                frames[f"{side}_{seg}"] = np.column_stack([np.cross(y, z), y, z])
            # Distal segments share the parent's rest frame so neutral = identity.
            frames[f"{side}_hand"] = frames[f"{side}_forearm"]
            frames[f"{side}_foot"] = frames[f"{side}_shin"]
        return frames


def fk_world(joints, frame, use_offsets=False):
    """World positions, world rotations and local rotations of one frame.

    Position channels on non-root joints are absolute local translations in this
    library (they equal the offsets up to retarget scale), so they replace the
    offset rather than adding to it.
    """
    n = len(joints)
    pos = np.zeros((n, 3)); rot = np.zeros((n, 3, 3)); local = np.zeros((n, 3, 3))
    ci = 0
    for idx, (name, parent, off, chans, is_end) in enumerate(joints):
        t = np.array(off, float); order = ""; rots = {}; tr = {}
        for ch in chans:
            if ch.endswith("position"): tr[ch[0]] = frame[ci]
            else: order += ch[0]; rots[ch[0]] = frame[ci]
            ci += 1
        if tr and not use_offsets:
            t = np.array([tr.get("X", 0.0), tr.get("Y", 0.0), tr.get("Z", 0.0)])
        R = euler_matrix(order, [rots[a] for a in order]) if order else np.eye(3)
        local[idx] = R
        if parent < 0:
            rot[idx] = R; pos[idx] = t
        else:
            rot[idx] = rot[parent] @ R; pos[idx] = pos[parent] + rot[parent] @ t
    return pos, rot, local


def recompute(joints, frame, local):
    """Write local rotations back into the frame's rotation channels."""
    out = np.array(frame, float).copy(); ci = 0
    for idx, (_, _, _, chans, _) in enumerate(joints):
        order = "".join(c[0] for c in chans if c.endswith("rotation"))
        if order:
            values = iter(euler_angles(local[idx], order))
            for k, ch in enumerate(chans):
                if ch.endswith("rotation"): out[ci + k] = next(values)
        ci += len(chans)
    return out


# ---------------------------------------------------------------- measures
def _interp(table, e, col):
    xs = [row[0] for row in table]; ys = [row[col] for row in table]
    return float(np.interp(e, xs, ys))


@dataclass
class Pose:
    sk: Skeleton
    frame: np.ndarray
    pos: np.ndarray = field(init=False)
    rot: np.ndarray = field(init=False)
    local: np.ndarray = field(init=False)

    def __post_init__(self):
        self.pos, self.rot, self.local = fk_world(self.sk.joints, self.frame)

    def seg(self, key):
        return self.rot[self.sk.role[key]] @ self.sk.rest_frames[key]


def _direction(P, S, s):
    """Segment's distal direction in the proximal frame: (lateral_out, up, fwd)."""
    d = P.T @ -S[:, 1]
    return np.array([s * d[0], d[1], d[2]])


def shoulder_limits(d, tier):
    col = 1 if tier == "normal" else 2
    e = math.degrees(math.acos(max(-1.0, min(1.0, -d[1]))))
    medial = SHOULDER_MEDIAL_FRONT if d[2] >= 0 else SHOULDER_MEDIAL_BACK
    return e, _interp(SHOULDER_POSTERIOR, e, col), _interp(medial, e, col)


def knee_bend(knee_flexion):
    """0 for a straight knee, 1 at 90 degrees or more (biarticular muscles)."""
    return float(np.clip((knee_flexion or 0.0) / 90.0, 0.0, 1.0))


def limits_for(code, m, side):
    """(normal, extreme) for a simple measure, conditioned on the neighbour joint."""
    _, normal, extreme, _ = LIMITS[code]
    if code == "ankle_plantar":  # gastrocnemius: dorsiflexion grows as the knee bends
        k = knee_bend(m.get(f"{side}_knee_flexion"))
        normal, extreme = (normal[0] - 20 * k, normal[1]), (extreme[0] - 20 * k, extreme[1])
    return normal, extreme


def hip_emax(d, tier, knee_flexion=90.0):
    col = 0 if tier == "normal" else 1
    theta = math.atan2(d[2], d[0])  # 0 lateral, +90 forward
    k = knee_bend(knee_flexion)
    abd = HIP_ABDUCTION[col]
    flex = HIP_FLEXION_STRAIGHT[col] + k * (HIP_FLEXION[col] - HIP_FLEXION_STRAIGHT[col])
    if theta >= math.pi / 2: return flex
    if theta <= 0: return abd
    return abd * math.cos(theta) ** 2 + flex * math.sin(theta) ** 2


def clavicle_elevation(pose, C, side):
    rest = pose.sk.rest_clavicle.get(side)
    if rest is None: return None
    r = pose.sk.role
    v = C.T @ (pose.pos[r[side + "_upperarm"]] - pose.pos[r[side + "_clavicle"]])
    return _deg_asin(v[1] / np.linalg.norm(v)) - _deg_asin(rest[1])


def girdle_shortfall(m, side):
    clav = m.get(side + "_clavicle_elevation")
    if clav is None: return None
    expected = GIRDLE_RATIO * max(0.0, m[side + "_shoulder_elevation"] - 90.0)
    if expected <= GIRDLE_SHORTFALL: return None  # arm not high enough to need the girdle
    return expected - clav


def _deg_asin(x):
    return math.degrees(math.asin(max(-1.0, min(1.0, x))))


def measure(pose: Pose) -> dict:
    sk = pose.sk
    m = {}
    P, C, H = pose.seg("pelvis"), pose.seg("chest"), pose.seg("head")
    a, b, c = euler_angles(P.T @ C, "XZY")
    m["trunk_flexion"], m["trunk_lateral"], m["trunk_rotation"] = a, b, c
    a, b, c = euler_angles(C.T @ H, "XZY")
    m["neck_flexion"], m["neck_lateral"], m["neck_rotation"] = a, b, c
    for side, (_, s) in SIDES.items():
        U, F, Hd = pose.seg(side + "_upperarm"), pose.seg(side + "_forearm"), pose.seg(side + "_hand")
        d = _direction(C, U, s)
        e, post_n, med_n = shoulder_limits(d, "normal")
        m[side + "_shoulder_elevation"] = e
        m[side + "_shoulder_posterior"] = _deg_asin(-d[2])
        m[side + "_shoulder_medial"] = _deg_asin(-d[0])
        m[side + "_humeral_rotation"] = s * swing_twist(C.T @ U) if e < TWIST_MAX_ELEVATION else None
        m[side + "_clavicle_elevation"] = clavicle_elevation(pose, C, side)
        a, b, c_e = euler_angles(U.T @ F, "XZY")
        m[side + "_elbow_flexion"], m[side + "_elbow_offaxis"] = -a, b
        a, b, c_w = euler_angles(F.T @ Hd, "XZY")
        m[side + "_wrist_extension"], m[side + "_wrist_deviation"] = s * b, a
        m[side + "_forearm_rotation"] = s * (c_e + c_w)
        T, K, Ft = pose.seg(side + "_thigh"), pose.seg(side + "_shin"), pose.seg(side + "_foot")
        d = _direction(P, T, s)
        m[side + "_hip_elevation"] = math.degrees(math.acos(max(-1.0, min(1.0, -d[1]))))
        m[side + "_hip_azimuth"] = math.degrees(math.atan2(d[2], d[0]))
        m[side + "_hip_posterior"] = _deg_asin(-d[2])
        m[side + "_hip_medial"] = _deg_asin(-d[0])
        m[side + "_hip_rotation"] = (s * swing_twist(P.T @ T)
                                     if m[side + "_hip_elevation"] < TWIST_MAX_ELEVATION else None)
        a, b, c = euler_angles(T.T @ K, "XZY")
        m[side + "_knee_flexion"], m[side + "_knee_offaxis"], m[side + "_knee_rotation"] = a, b, s * c
        a, b, c = euler_angles(K.T @ Ft, "XZY")
        m[side + "_ankle_plantar"], m[side + "_ankle_eversion"], m[side + "_ankle_toe_out"] = a, s * b, s * c
    return {k: (None if v is None else round(float(v), 2)) for k, v in m.items()}


# ---------------------------------------------------------------- evaluation
def _band(value, normal, extreme):
    if value is None: return None
    if value < extreme[0] - TOLERANCE or value > extreme[1] + TOLERANCE: return "reject"
    if value < normal[0] - TOLERANCE or value > normal[1] + TOLERANCE: return "hold"
    return None


def _finding(code, side, value, severity, limit):
    label = LIMITS[code][0]
    where = {"left": "왼쪽 ", "right": "오른쪽 "}.get(side, "")
    return {"code": code, "side": side, "label": where + label, "value": round(value, 1),
            "limit": limit, "severity": severity, "source": LIMITS[code][3]}


def rom_findings(m) -> list[dict]:
    out = []
    simple = [k for k, v in LIMITS.items() if v[1] is not None]
    for code in simple:
        sides = [None] if code.startswith(("trunk", "neck")) else list(SIDES)
        for side in sides:
            key = code if side is None else f"{side}_{code}"
            if key not in m: continue
            normal, extreme = limits_for(code, m, side)
            sev = _band(m[key], normal, extreme)
            if sev:
                lim = [round(x, 1) for x in (normal if sev == "hold" else extreme)]
                out.append(_finding(code, side, m[key], sev, list(lim)))
    for side in SIDES:
        short = girdle_shortfall(m, side)
        if short is not None and short > GIRDLE_SHORTFALL:
            out.append(_finding("shoulder_girdle", side, short, "hold", GIRDLE_SHORTFALL))
        e = m[side + "_shoulder_elevation"]
        fwd_ok = m[side + "_shoulder_posterior"] <= 0
        for code, table in (("shoulder_posterior", SHOULDER_POSTERIOR),
                            ("shoulder_medial", SHOULDER_MEDIAL_FRONT if fwd_ok else SHOULDER_MEDIAL_BACK)):
            v = m[f"{side}_{code}"]
            n, x = _interp(table, e, 1), _interp(table, e, 2)
            sev = "reject" if v > x + TOLERANCE else "hold" if v > n + TOLERANCE else None
            if sev: out.append(_finding(code, side, v, sev, n if sev == "hold" else x))
        el = m[side + "_hip_elevation"]
        d = _hip_d(m, side)
        for code, (n, x) in (("hip_posterior", HIP_POSTERIOR), ("hip_medial", HIP_MEDIAL)):
            v = m[f"{side}_{code}"]
            sev = "reject" if v > x + TOLERANCE else "hold" if v > n + TOLERANCE else None
            if sev: out.append(_finding(code, side, v, sev, n if sev == "hold" else x))
        knee = m[side + "_knee_flexion"]
        n, x = hip_emax(d, "normal", knee), hip_emax(d, "extreme", knee)
        sev = "reject" if el > x + TOLERANCE else "hold" if el > n + TOLERANCE else None
        if sev: out.append(_finding("hip_cone", side, el, sev, round(n if sev == "hold" else x, 1)))
    return out


def _hip_d(m, side):
    e, th = math.radians(m[side + "_hip_elevation"]), math.radians(m[side + "_hip_azimuth"])
    return np.array([math.sin(e) * math.cos(th), -math.cos(e), math.sin(e) * math.sin(th)])


def _coco(pose):
    """COCO17 world points (head approximations from Head) for capsule checks."""
    r, p = pose.sk.role, pose.pos
    idx = {0: "head", 1: "head", 2: "head", 3: "head", 4: "head",
           5: "left_upperarm", 6: "right_upperarm", 7: "left_forearm", 8: "right_forearm",
           9: "left_hand", 10: "right_hand", 11: "left_thigh", 12: "right_thigh",
           13: "left_shin", 14: "right_shin", 15: "left_foot", 16: "right_foot"}
    return np.array([p[r[idx[i]]] for i in range(17)]), np.ones(17)


def collision_findings(pose) -> tuple[list[dict], dict]:
    kp, sc = _coco(pose)
    r = pose.sk.role
    tips = {s: (pose.pos[r[s + "_handtip"]] if r.get(s + "_handtip", -1) >= 0 else None) for s in SIDES}
    measures = {}
    for s in SIDES:
        measures[f"{s}_arm/torso"] = arm_torso_penetration(kp, s + "_arm", sc, hand_tip=tips[s], samples=21)
        measures[f"{s}_leg/torso"] = leg_torso_penetration(kp, s + "_leg", sc)
        for t in SIDES:
            measures[f"{s}_arm/{t}_leg"] = arm_leg_penetration(kp, s + "_arm", t + "_leg", sc, hand_tip=tips[s])
    measures["shin/shin"] = leg_leg_penetration(kp, sc)
    out = []
    for name, cm in measures.items():
        if not cm.available or cm.depth <= COLLISION_HOLD: continue
        sev = "reject" if cm.depth > COLLISION_REJECT else "hold"
        out.append({"code": "collision", "side": None, "label": f"몸 관통 {name}",
                    "value": round(cm.depth, 3), "limit": COLLISION_HOLD if sev == "hold" else COLLISION_REJECT,
                    "severity": sev, "source": "src.collision capsule proxy (torso lengths)"})
    return out, {k: round(v.depth, 4) for k, v in measures.items() if v.available}


def balance(pose) -> dict:
    """de Leva CoM vs. foot support polygon. Information only (jumps, leans, sits)."""
    r, p = pose.sk.role, pose.pos
    mid = lambda a, b, f: p[a] + (p[b] - p[a]) * f
    neck, pelvis = p[r["neck"]], p[r["pelvis"]]
    parts = [(DE_LEVA["head"][0], p[r["head"]]),
             (DE_LEVA["trunk"][0], neck + (pelvis - neck) * .5138)]
    for s in SIDES:
        g = lambda k: r[f"{s}_{k}"]
        parts += [(DE_LEVA["upperarm"][0], mid(g("upperarm"), g("forearm"), .5772)),
                  (DE_LEVA["forearm"][0], mid(g("forearm"), g("hand"), .4574)),
                  (DE_LEVA["hand"][0], p[g("hand")]),
                  (DE_LEVA["thigh"][0], mid(g("thigh"), g("shin"), .4095)),
                  (DE_LEVA["shank"][0], mid(g("shin"), g("foot"), .4395)),
                  (DE_LEVA["foot"][0], p[g("foot")] if r[s + "_toe"] < 0 else (p[g("foot")] + p[r[s + "_toe"]]) / 2)]
    com = sum(w * x for w, x in parts) / sum(w for w, _ in parts)
    leg = np.linalg.norm(p[r["left_thigh"]] - p[r["left_shin"]]) + np.linalg.norm(p[r["left_shin"]] - p[r["left_foot"]])
    floor = p[:, 1].min()
    candidates = {k: p[r[k]] for k in r if r[k] >= 0 and k.split("_", 1)[-1] in
                  ("foot", "toe", "hand", "handtip", "shin", "pelvis", "thigh") or k == "pelvis"}
    contact = sorted(k for k, x in candidates.items() if x[1] - floor < .1 * leg)
    feet_only = all(k.split("_", 1)[-1] in ("foot", "toe") for k in contact)
    pts = np.array([candidates[k][[0, 2]] for k in contact])
    outside = _hull_distance(com[[0, 2]], pts) / leg
    return {"contacts": contact, "feet_only": feet_only, "com_outside_leg": round(float(outside), 3),
            "unsupported": bool(feet_only and outside > .25)}


def _hull_distance(q, pts):
    if len(pts) == 0: return 0.0
    if len(pts) == 1: return float(np.linalg.norm(q - pts[0]))
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    hull = []
    for seq in (pts, pts[::-1]):
        part = []
        for x in seq:
            while len(part) >= 2 and cross(part[-2], part[-1], x) <= 0: part.pop()
            part.append(x)
        hull += part[:-1]
    hull = np.array(hull) if len(hull) >= 2 else pts[:2]
    inside = len(hull) >= 3 and all(cross(hull[i], hull[(i + 1) % len(hull)], q) >= 0 for i in range(len(hull)))
    if inside: return 0.0
    best = math.inf
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        t = np.clip((q - a) @ (b - a) / max((b - a) @ (b - a), 1e-12), 0, 1)
        best = min(best, float(np.linalg.norm(q - (a + t * (b - a)))))
    return best


def evaluate(joints, frame) -> dict:
    sk = Skeleton(joints)
    if sk.missing:
        return {"version": VERSION, "verdict": "unsupported", "missing": sk.missing, "findings": []}
    pose = Pose(sk, np.asarray(frame, float))
    m = measure(pose)
    findings = rom_findings(m)
    coll, depths = collision_findings(pose)
    findings += coll
    bal = balance(pose)
    order = {"reject": 2, "hold": 1}
    verdict = max((f["severity"] for f in findings), key=order.get, default="pass")
    findings.sort(key=lambda f: (-order[f["severity"]], f["code"]))
    return {"version": VERSION, "verdict": verdict, "findings": findings,
            "measures": m, "collisions": depths, "balance": bal}


# ---------------------------------------------------------------- correction
def _set_world(pose, j, world):
    parent = pose.sk.parent[j]
    pose.local[j] = world if parent < 0 else pose.rot[parent].T @ world


def _refresh(pose):
    pose.frame = recompute(pose.sk.joints, pose.frame, pose.local)
    pose.pos, pose.rot, pose.local = fk_world(pose.sk.joints, pose.frame)


def _clamp(v, lim):
    return min(max(v, lim[0]), lim[1])


def _fix_chain(pose, chain, rest_key, rel, parent_frame, codes, signs=(1, 1, 1)):
    """Clamp an XZY relative rotation and spread the change along a chain."""
    a, b, c = rel
    want = [s * _clamp(s * v, LIMITS[k][1]) for v, k, s in zip((a, b, c), codes, signs)]
    if np.allclose(want, rel, atol=1e-6): return False
    top = chain[-1]
    S = pose.rot[top] @ pose.sk.rest_frames[rest_key]
    S_new = parent_frame @ euler_matrix("XZY", want)
    delta = S_new @ S.T
    worlds = {j: fractional(delta, (k + 1) / len(chain)) @ pose.rot[j] for k, j in enumerate(chain)}
    for j in chain:
        parent = pose.sk.parent[j]
        pw = worlds.get(parent, pose.rot[parent])
        pose.local[j] = pw.T @ worlds[j]
    _refresh(pose)
    return True


def correct(joints, frame) -> tuple[np.ndarray, list[str]]:
    """Clamp every measure into the normal tier. Returns (frame, changed steps).

    Proximal first so distal joints are measured after their parents settle.
    Elbow/knee off-axis bending is fixed by twisting the upper segment about
    its own axis, which leaves every joint position unchanged.
    """
    sk = Skeleton(joints)
    if sk.missing: return np.asarray(frame, float), []
    pose = Pose(sk, np.asarray(frame, float).copy())
    changed = []
    P, C = pose.seg("pelvis"), pose.seg("chest")
    if _fix_chain(pose, sk.spine, "chest", euler_angles(P.T @ C, "XZY"), P,
                  ("trunk_flexion", "trunk_lateral", "trunk_rotation")):
        changed.append("trunk")
    C, H = pose.seg("chest"), pose.seg("head")
    if _fix_chain(pose, sk.neck_chain, "head", euler_angles(C.T @ H, "XZY"), C,
                  ("neck_flexion", "neck_lateral", "neck_rotation")):
        changed.append("neck")
    for side, (_, s) in SIDES.items():
        if _fix_girdle(pose, side, s):
            changed.append(side + "_clavicle")
        for upper, lower, prox, project, twist, hinge, joint in (
                ("upperarm", "forearm", "chest", _project_shoulder, "humeral_rotation", 1, "elbow"),
                ("thigh", "shin", "pelvis", _project_hip, "hip_rotation", -1, "knee")):
            knee = euler_angles(pose.seg(side + "_thigh").T @ pose.seg(side + "_shin"), "XZY")[0]
            project_here = (lambda d, k=knee: _project_hip(d, k)) if joint == "knee" else project
            if _fix_ball(pose, side, s, upper, prox, project_here, twist):
                changed.append(f"{side}_{'shoulder' if joint == 'elbow' else 'hip'}")
            if _align_hinge(pose, side, s, upper, lower, prox, hinge, twist, joint + "_offaxis"):
                changed.append(f"{side}_{joint}_hinge")
        # Elbow: flexion/off-axis, then forearm rotation shared with the wrist.
        U, F, Hd = (pose.seg(f"{side}_{k}") for k in ("upperarm", "forearm", "hand"))
        a, b, c_e = euler_angles(U.T @ F, "XZY")
        c_w = euler_angles(F.T @ Hd, "XZY")[2]
        total = s * (c_e + c_w)
        excess = total - _clamp(total, LIMITS["forearm_rotation"][1])
        share = abs(c_e) / (abs(c_e) + abs(c_w)) if abs(c_e) + abs(c_w) > 1e-9 else 1.0
        new_e = (-_clamp(-a, LIMITS["elbow_flexion"][1]), _clamp(b, LIMITS["elbow_offaxis"][1]),
                 c_e - s * excess * share)
        if _set_relative(pose, side, "upperarm", "forearm", (a, b, c_e), new_e):
            changed.append(side + "_elbow")
        F, Hd = pose.seg(side + "_forearm"), pose.seg(side + "_hand")
        aw, bw, cw = euler_angles(F.T @ Hd, "XZY")
        total = s * (euler_angles(pose.seg(side + "_upperarm").T @ F, "XZY")[2] + cw)
        excess = total - _clamp(total, LIMITS["forearm_rotation"][1])
        new_w = (_clamp(aw, LIMITS["wrist_deviation"][1]),
                 s * _clamp(s * bw, LIMITS["wrist_extension"][1]), cw - s * excess)
        if _set_relative(pose, side, "forearm", "hand", (aw, bw, cw), new_w):
            changed.append(side + "_wrist")
        for seg, child, codes, signs, name in (
                ("thigh", "shin", ("knee_flexion", "knee_offaxis", "knee_rotation"), (1, 1, s), "knee"),
                ("shin", "foot", ("ankle_plantar", "ankle_eversion", "ankle_toe_out"), (1, s, s), "ankle")):
            A, B = pose.seg(f"{side}_{seg}"), pose.seg(f"{side}_{child}")
            rel = euler_angles(A.T @ B, "XZY")
            knee = {f"{side}_knee_flexion": euler_angles(
                pose.seg(side + "_thigh").T @ pose.seg(side + "_shin"), "XZY")[0]}
            want = tuple(sg * _clamp(sg * v, limits_for(k, knee, side)[0])
                         for v, k, sg in zip(rel, codes, signs))
            if _set_relative(pose, side, seg, child, rel, want):
                changed.append(f"{side}_{name}")
    return pose.frame, changed


def _fix_girdle(pose, side, s):
    """Lift (or settle) the clavicle; the humerus keeps its world orientation."""
    if side not in pose.sk.rest_clavicle: return False
    C = pose.seg("chest")
    m = {side + "_shoulder_elevation": _direction_elevation(C, pose.seg(side + "_upperarm"), s),
         side + "_clavicle_elevation": clavicle_elevation(pose, C, side)}
    clav = m[side + "_clavicle_elevation"]
    short = girdle_shortfall(m, side)
    if short is None or short <= GIRDLE_SHORTFALL: return False
    target = clav + min(short, GIRDLE_MAX_LIFT)
    r = pose.sk.role
    arm_world = pose.rot[r[side + "_upperarm"]].copy()
    j = r[side + "_clavicle"]
    _set_world(pose, j, axis_angle(C[:, 2], s * (target - clav)) @ pose.rot[j])
    _refresh(pose)
    _set_world(pose, r[side + "_upperarm"], arm_world)
    _refresh(pose)
    return True


def _direction_elevation(P, S, s):
    d = _direction(P, S, s)
    return math.degrees(math.acos(max(-1.0, min(1.0, -d[1]))))


def _set_relative(pose, side, seg, child, old, new):
    if np.allclose(new, old, atol=1e-6): return False
    A = pose.seg(f"{side}_{seg}")
    key = f"{side}_{child}"
    _set_world(pose, pose.sk.role[key], A @ euler_matrix("XZY", new) @ pose.sk.rest_frames[key].T)
    _refresh(pose)
    return True


def _align_hinge(pose, side, s, upper, lower, proximal, hinge, twist_code, offaxis_code):
    """Twist the upper segment so the lower one bends in its hinge plane."""
    key, low = f"{side}_{upper}", f"{side}_{lower}"
    U, L = pose.seg(key), pose.seg(low)
    off = euler_angles(U.T @ L, "XZY")[1]
    lim = LIMITS[offaxis_code][1]
    f = U.T @ -L[:, 1]
    if lim[0] <= off <= lim[1] or -f[1] > math.cos(math.radians(25)):
        return False  # in tolerance, or too straight for a stable hinge plane
    phi = math.degrees(math.atan2(hinge * f[0], hinge * f[2]))
    P = pose.seg(proximal)
    best = 0.0
    for t in np.linspace(1, 0, 21):
        Ut = U @ _axis_rot("Y", t * phi)
        M = P.T @ Ut
        if M[1, 1] <= math.cos(math.radians(TWIST_MAX_ELEVATION)):
            best = t; break
        tw = s * swing_twist(M)
        if LIMITS[twist_code][1][0] <= tw <= LIMITS[twist_code][1][1]:
            best = t; break
    if abs(best * phi) < 1.0: return False
    lower_world = pose.rot[pose.sk.role[low]].copy()
    _set_world(pose, pose.sk.role[key], U @ _axis_rot("Y", best * phi) @ pose.sk.rest_frames[key].T)
    _refresh(pose)
    _set_world(pose, pose.sk.role[low], lower_world)
    _refresh(pose)
    return True


def _project_shoulder(d):
    for _ in range(4):
        e, post, med = shoulder_limits(d, "normal")
        d = _cap(d, 2, post); d = _cap(d, 0, med)
    return d


def _project_hip(d, knee_flexion=90.0):
    for _ in range(4):
        d = _cap(d, 2, HIP_POSTERIOR[0]); d = _cap(d, 0, HIP_MEDIAL[0])
        e = math.degrees(math.acos(max(-1.0, min(1.0, -d[1]))))
        emax = hip_emax(d, "normal", knee_flexion)
        if e > emax:
            th = math.atan2(d[2], d[0]); em = math.radians(emax)
            d = np.array([math.sin(em) * math.cos(th), -math.cos(em), math.sin(em) * math.sin(th)])
    return d


def _cap(d, axis, limit_deg):
    """Ensure -d[axis] <= sin(limit): closest point on the sphere cap boundary."""
    lo = -math.sin(math.radians(limit_deg))
    if d[axis] >= lo: return d
    rest = [i for i in range(3) if i != axis]
    norm = math.hypot(d[rest[0]], d[rest[1]])
    out = np.array(d, float); out[axis] = lo
    k = math.sqrt(1 - lo * lo) / norm if norm > 1e-9 else 0.0
    for i in rest: out[i] = d[i] * k
    if norm <= 1e-9: out[1] = -math.sqrt(1 - lo * lo)
    return out


def _fix_ball(pose, side, s, seg, proximal, project, twist_code):
    key = f"{side}_{seg}"
    P, S = pose.seg(proximal), pose.seg(key)
    d = _direction(P, S, s)
    d2 = project(d.copy())
    moved = False
    if np.linalg.norm(d2 - d) > 1e-6:
        to_world = lambda v: P @ np.array([s * v[0], v[1], v[2]])
        G = rotation_between(to_world(d), to_world(d2))
        S = G @ S; moved = True
    M = P.T @ S
    if M[1, 1] > math.cos(math.radians(TWIST_MAX_ELEVATION)):  # twist defined
        tw = s * swing_twist(M)
        want = _clamp(tw, LIMITS[twist_code][1])
        if abs(want - tw) > 1e-6:
            S = S @ _axis_rot("Y", s * (want - tw)); moved = True
    if moved:
        _set_world(pose, pose.sk.role[key], S @ pose.sk.rest_frames[key].T)
        _refresh(pose)
    return moved
