"""Contact-preserving, stepwise ankle correction for planted feet.

Clamping a planted ankle turns the foot about the ankle, so the sole sinks into
or lifts off the floor. Here the sole keeps its floor contacts and the legs
(hip, knee, ankle rotations) plus a small root shift absorb the change, as in
footskate cleanup (Kovar et al. 2002) solved with damped least squares
(Levenberg-Marquardt; Buss 2004). Excess dorsiflexion may instead lift the heel
about the ball of the foot, the usual human compensation in a deep squat.

The correction is tried at 25/50/75/100 % of the way into the normal range and
the passing step with the largest ankle improvement is recommended ("better or
unchanged"); every step is kept for the reviewer.
Feet rotated far past the extreme tier (e.g. reversed) are rebuilt instead of
projected and are always marked for human review.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from . import plausibility as pl

STEPS = (0.25, 0.5, 0.75, 1.0)
MARGIN = 1.5            # degrees inside the normal boundary
CONTACT_EPS = 0.03      # leg lengths: a sole point this close to the floor is planted
GATE_CONTACT = 0.012    # leg lengths (~1 cm on an adult): planted points may not drift more
GATE_FLOOR = 0.006      # leg lengths below the floor
GATE_SILHOUETTE = 0.05  # RMS body-keypoint change / body height
GATE_ROOT = 0.08        # leg lengths of root shift
GROSS = 45.0            # degrees beyond the extreme tier: rebuild instead of projecting
ANKLE = ("ankle_plantar", "ankle_eversion", "ankle_toe_out")
LEG_LIMITS = ("knee_flexion", "knee_offaxis", "knee_rotation")
HEEL_BACK = 0.25        # heel sits this fraction of ankle->ball behind the ankle
W_CONTACT, W_FLOOR, W_ANKLE, W_LIMIT, W_TURN, W_ROOT = 300.0, 1000.0, 1.0, 1.0, 1.0, 4.0


# ---------------------------------------------------------------- geometry
@dataclass
class Foot:
    side: str
    s: float
    chain: tuple          # thigh, shin, foot, toe joint indices
    heel: np.ndarray      # offsets in the foot joint's rest frame (rest rotation = I)
    ball: np.ndarray
    tip: np.ndarray       # offset in the toe joint's rest frame


def feet(sk: pl.Skeleton):
    """Sole points per side, measured once on the rest pose."""
    r, p = sk.role, sk.rest_pos
    floor = p[:, 1].min()
    up, fwd = sk.rest_body[:, 1], sk.rest_body[:, 2]
    out = {}
    for side, (_, s) in pl.SIDES.items():
        idx = [r[f"{side}_{k}"] for k in ("thigh", "shin", "foot", "toe")]
        if min(idx) < 0 or not all(sk.parent[b] == a for a, b in zip(idx, idx[1:])):
            continue
        ankle, toe = p[idx[2]], p[idx[3]]
        tips = [c for c in sk.children[idx[3]]]
        tip = p[tips[0]] if tips else toe
        flat = lambda x: x - up * ((x - p[idx[2]]) @ up) + up * ((floor - p[idx[2]]) @ up)
        heel = flat(ankle) - fwd * HEEL_BACK * max((toe - ankle) @ fwd, 0.0)
        ball, tip_floor = flat(toe), flat(tip)
        out[side] = Foot(side, s, tuple(idx), heel - ankle, ball - ankle, tip_floor - toe)
    return out


def leg_length(sk):
    r, p = sk.role, sk.rest_pos
    return float(np.linalg.norm(p[r["left_thigh"]] - p[r["left_shin"]])
                 + np.linalg.norm(p[r["left_shin"]] - p[r["left_foot"]]))


def sole_points(pos, rot, foot: Foot):
    _, _, f, t = foot.chain
    return {"heel": pos[f] + rot[f] @ foot.heel, "ball": pos[f] + rot[f] @ foot.ball,
            "tip": pos[t] + rot[t] @ foot.tip}


def focus_joints(sk, sides):
    """Knee-to-toe joints of the given sides, for close-up drawings."""
    out = []
    for side in sides:
        idx = [sk.role[f"{side}_{k}"] for k in ("shin", "foot", "toe")]
        out += [i for i in idx if i >= 0] + (sk.children[idx[2]] if idx[2] >= 0 else [])
    return out


# ---------------------------------------------------------------- measures
def ankle_angles(sk, rot, side, s):
    K = rot[sk.role[side + "_shin"]] @ sk.rest_frames[side + "_shin"]
    F = rot[sk.role[side + "_foot"]] @ sk.rest_frames[side + "_foot"]
    a, b, c = pl.euler_angles(K.T @ F, "XZY")
    return np.array([a, s * b, s * c])


def knee_angles(sk, rot, side, s):
    T = rot[sk.role[side + "_thigh"]] @ sk.rest_frames[side + "_thigh"]
    K = rot[sk.role[side + "_shin"]] @ sk.rest_frames[side + "_shin"]
    a, b, c = pl.euler_angles(T.T @ K, "XZY")
    return np.array([a, b, s * c])


def hip_excess(sk, rot, side, s, knee_flexion):
    """Degrees beyond the normal hip swing cone, posterior/medial caps and twist."""
    P = rot[sk.role["pelvis"]] @ sk.rest_frames["pelvis"]
    T = rot[sk.role[side + "_thigh"]] @ sk.rest_frames[side + "_thigh"]
    d = pl._direction(P, T, s)
    e = math.degrees(math.acos(max(-1.0, min(1.0, -d[1]))))
    out = max(e - pl.hip_emax(d, "normal", knee_flexion), 0.0)
    out += max(pl._deg_asin(-d[2]) - pl.HIP_POSTERIOR[0], 0.0)
    out += max(pl._deg_asin(-d[0]) - pl.HIP_MEDIAL[0], 0.0)
    if e < pl.TWIST_MAX_ELEVATION:
        lo, hi = pl.LIMITS["hip_rotation"][1]
        tw = s * pl.swing_twist(P.T @ T)
        out += max(lo - tw, 0.0) + max(tw - hi, 0.0)
    return out


def ankle_bounds(knee_flexion, side, tier=0):
    m = {f"{side}_knee_flexion": knee_flexion}
    return [pl.limits_for(code, m, side)[tier] for code in ANKLE]


def excess(values, bounds):
    return sum(max(lo - v, 0.0) + max(v - hi, 0.0) for v, (lo, hi) in zip(values, bounds))


# ---------------------------------------------------------------- solver
class LegSolve:
    """Unknowns: root shift (3) + rotation vectors on thigh/shin/foot per leg."""

    def __init__(self, joints, frame, sk, legs, contacts, targets, ref_pos, length):
        self.joints, self.sk, self.legs = joints, sk, legs
        self.frame = np.asarray(frame, float)
        self.pos, self.rot, self.local = pl.fk_world(joints, self.frame)
        self.contacts, self.targets, self.ref, self.L = contacts, targets, ref_pos, length
        self.floor = ref_pos["floor"]
        self.base_knee = {f.side: knee_angles(sk, self.rot, f.side, f.s) for f in legs}
        self.base_hip = {f.side: hip_excess(sk, self.rot, f.side, f.s, self.base_knee[f.side][0])
                         for f in legs}
        # Toe end sites ride on the toe joint and must stay above the floor too.
        self.ends = {f.side: [self.rot[f.chain[3]].T @ (self.pos[c] - self.pos[f.chain[3]])
                              for c in sk.children[f.chain[3]]] for f in legs}
        self.n = 3 + 9 * len(legs)

    def apply(self, x):
        """Positions and rotations of every leg joint for unknowns x."""
        shift = x[:3] * self.L
        pelvis = self.sk.role["pelvis"]
        pos, rot = {}, {pelvis: self.rot[pelvis]}
        for k, foot in enumerate(self.legs):
            base = self.sk.parent[foot.chain[0]]
            p_pos, p_rot = self.pos[base] + shift, self.rot[base]
            for m, j in enumerate(foot.chain):
                local = self.local[j]
                if m < 3:
                    w = x[3 + 9 * k + 3 * m: 6 + 9 * k + 3 * m]
                    angle = float(np.linalg.norm(w))
                    if angle > 1e-12:
                        local = local @ pl.axis_angle(w / angle, math.degrees(angle))
                parent = self.sk.parent[j]
                t = self.rot[parent].T @ (self.pos[j] - self.pos[parent])
                pos[j] = p_pos + p_rot @ t
                rot[j] = p_rot @ local
                p_pos, p_rot = pos[j], rot[j]
        return pos, rot, shift

    def residuals(self, x):
        pos, rot, shift = self.apply(x)
        out = []
        for foot in self.legs:
            pts = sole_points(pos, rot, foot)
            for name, p in pts.items():
                want = self.contacts.get((foot.side, name))
                if want is not None:
                    d = (p - want[:3]) if want[3] else np.array([0.0, p[1] - want[1], 0.0])
                    out.extend(W_CONTACT * d / self.L)
                out.append(W_FLOOR * max(self.floor - p[1], 0.0) / self.L)
            toe = foot.chain[3]
            for p in [pos[j] for j in foot.chain[1:]] + [pos[toe] + rot[toe] @ e for e in self.ends[foot.side]]:
                out.append(W_FLOOR * max(self.floor - p[1], 0.0) / self.L)
            ang = ankle_angles(self.sk, rot, foot.side, foot.s)
            target = self.targets.get(foot.side)
            if target is not None:
                out.extend(W_ANKLE * (ang - target))
            knee = knee_angles(self.sk, rot, foot.side, foot.s)
            for v, v0, code in zip(knee, self.base_knee[foot.side], LEG_LIMITS):
                lo, hi = pl.LIMITS[code][1]
                new = max(lo - v, 0.0) + max(v - hi, 0.0)
                old = max(lo - v0, 0.0) + max(v0 - hi, 0.0)
                out.append(W_LIMIT * max(new - old, 0.0))
            hip = hip_excess(self.sk, rot, foot.side, foot.s, knee[0])
            out.append(W_LIMIT * max(hip - self.base_hip[foot.side], 0.0))
        out.extend(W_TURN * x[3:])
        out.extend(W_ROOT * x[:3])
        return np.array(out)

    def solve(self, iterations=40):
        """Levenberg-Marquardt with a finite-difference Jacobian."""
        x = np.zeros(self.n)
        r = self.residuals(x)
        cost, lam = r @ r, 1e-2
        for _ in range(iterations):
            J = np.empty((r.size, self.n))
            for i in range(self.n):
                dx = np.zeros(self.n); dx[i] = 1e-5
                J[:, i] = (self.residuals(x + dx) - r) / 1e-5
            g, H = J.T @ r, J.T @ J
            improved = False
            for _ in range(8):
                step = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), -g)
                cap = np.max(np.abs(step[3:])) if self.n > 3 else 0.0
                if cap > math.radians(10):  # at most 10 degrees per iteration
                    step *= math.radians(10) / cap
                r2 = self.residuals(x + step)
                if r2 @ r2 < cost:
                    x, r, cost, lam, improved = x + step, r2, r2 @ r2, lam * 0.3, True
                    break
                lam *= 4
            if not improved or np.linalg.norm(step) < 1e-7:
                break
        return x

    def frame_for(self, x):
        pos, rot, shift = self.apply(x)
        local = self.local.copy()
        for foot in self.legs:
            for j in foot.chain[:3]:
                parent = self.sk.parent[j]
                parent_rot = rot.get(parent, self.rot[parent])
                local[j] = parent_rot.T @ rot[j]
        frame = pl.recompute(self.joints, self.frame, local)
        root = self.joints[0]
        chans = [c for c in root[3] if c.endswith("position")]
        for k, ch in enumerate(root[3]):
            if ch.endswith("position"):
                frame[k] += shift["XYZ".index(ch[0])]
        if not chans and np.linalg.norm(shift) > 1e-9:
            raise ValueError("root has no position channels")
        return frame


# ---------------------------------------------------------------- driver
def planted(joints, frame, sk=None):
    """Sides whose sole touches the floor, with the touching points."""
    sk = sk or pl.Skeleton(joints)
    pos, rot, _ = pl.fk_world(joints, frame)
    L = leg_length(sk)
    soles = {side: sole_points(pos, rot, f) for side, f in feet(sk).items()}
    floor = min([pos[:, 1].min()] + [p[1] for pts in soles.values() for p in pts.values()])
    return {side: {n: p for n, p in pts.items() if p[1] - floor < CONTACT_EPS * L}
            for side, pts in soles.items()
            if any(p[1] - floor < CONTACT_EPS * L for p in pts.values())}, floor


def _keypoints(joints, frame):
    sk = pl.Skeleton(joints)
    pos, _, _ = pl.fk_world(joints, frame)
    return pl._coco(pl.Pose(sk, frame))[0], pos[:, 1].max() - pos[:, 1].min()


def correct_ankles(joints, original, base=None):
    """Fix planted ankles of ``base`` (default: original) against original contacts.

    Returns a report dict with every attempted step and the chosen frame.
    """
    original = np.asarray(original, float)
    base = original if base is None else np.asarray(base, float)
    sk = pl.Skeleton(joints)
    if sk.missing:
        return {"status": "unsupported", "frame": base, "steps": []}
    L = leg_length(sk)
    contact, floor = planted(joints, original, sk)
    model = feet(sk)
    legs = [model[s] for s in pl.SIDES if s in model]
    pos0, rot0, _ = pl.fk_world(joints, base)
    sides, gross, targets_full, start = {}, [], {}, base.copy()
    for foot in legs:
        if foot.side not in contact:
            continue
        ang = ankle_angles(sk, rot0, foot.side, foot.s)
        knee = knee_angles(sk, rot0, foot.side, foot.s)[0]
        normal = ankle_bounds(knee, foot.side, 0)
        extreme = ankle_bounds(knee, foot.side, 1)
        shrunk = [(lo + MARGIN, hi - MARGIN) for lo, hi in normal]
        target = np.array([min(max(v, lo), hi) for v, (lo, hi) in zip(ang, shrunk)])
        moved = max((np.linalg.norm(sole_points(pos0, rot0, foot)[n] - p)
                     for n, p in contact[foot.side].items()), default=0.0)
        flagged = excess(ang, [(lo - pl.TOLERANCE, hi + pl.TOLERANCE) for lo, hi in normal]) > 0
        targets_full[foot.side] = ang  # planted feet keep their ankle unless corrected
        if not flagged and moved <= GATE_CONTACT * L:
            continue
        if excess(ang, extreme) > GROSS:
            gross.append(foot.side)
            target = np.array([target[0], 0.0, 0.0])
        sides[foot.side] = (ang, target, normal, flagged)
    if not sides:
        return {"status": "nothing", "frame": base, "steps": []}
    kept = {s: t for s, t in targets_full.items() if s not in sides}
    if gross:  # rebuild reversed/broken feet first; contacts then only pin the height
        pose = pl.Pose(sk, start)
        for side in gross:
            want = sides[side][1]
            s = model[side].s
            shin = pose.seg(side + "_shin")
            rel = pl.euler_matrix("XZY", (want[0], s * want[1], s * want[2]))
            pl._set_world(pose, sk.role[side + "_foot"], shin @ rel @ sk.rest_frames[side + "_foot"].T)
            pl._refresh(pose)
        start = pose.frame
    ref = {"floor": floor}
    kp0, height = _keypoints(joints, base)  # other fixes may already have moved the base
    before = pl.evaluate(joints, original)
    before_keys = {(f["code"], f["side"]): f["severity"] for f in before["findings"]}
    rank = {"hold": 1, "reject": 2}
    steps = []
    alphas = (1.0,) if gross else STEPS if any(v[3] for v in sides.values()) else ()
    for alpha in alphas:
        best = None
        heel_modes = [False]
        if any(v[3] and v[0][0] < v[2][0][0] and "heel" in contact[s] and len(contact[s]) > 1
               for s, v in sides.items()):
            heel_modes.append(True)  # excess dorsiflexion: may lift the heel instead
        for release_heel in heel_modes:
            targets = dict(kept)
            targets.update({s: (t if s in gross else sides[s][0] + alpha * (t - sides[s][0]))
                            for s, (_, t, _, _) in sides.items()})
            pins = {}
            for side, pts in contact.items():
                for name, p in pts.items():
                    if release_heel and name == "heel" and side in sides:
                        continue
                    full = side not in gross
                    pins[(side, name)] = np.append(p, 1.0 if full else 0.0)
            solver = LegSolve(joints, start, sk, legs, pins, targets, ref, L)
            x = solver.solve()
            frame = solver.frame_for(x)
            report = _gate(joints, sk, original, frame, contact, sides, floor, L, kp0, height,
                           before_keys, rank, release_heel, gross)
            report.update(alpha=alpha, heel_lift=release_heel,
                          root_shift=round(float(np.linalg.norm(x[:3])), 4))
            report["_frame"] = frame
            if best is None or (report["ok"], -report["silhouette"]) > (best["ok"], -best["silhouette"]):
                best = report
        steps.append(best)
    passing = [s for s in steps if s["ok"]]
    chosen = max(passing, key=lambda s: (s["ankle_gain"], -s["silhouette"]), default=None)
    status = "gross" if gross else "corrected"
    if chosen is None or not any(flag for *_, flag in sides.values()):
        # Cannot improve the ankle safely: at least put the feet back where they were.
        keep = {s: ang for s, (ang, *_rest) in sides.items()}
        keep.update(kept)
        pins = {(side, name): np.append(p, 1.0) for side, pts in contact.items() for name, p in pts.items()}
        solver = LegSolve(joints, base, sk, legs, pins, keep, ref, L)
        x = solver.solve()
        frame = solver.frame_for(x)
        report = _gate(joints, sk, original, frame, contact, {}, floor, L, kp0, height,
                       before_keys, rank, False, [])
        report.update(alpha=0.0, heel_lift=False, root_shift=round(float(np.linalg.norm(x[:3])), 4),
                      _frame=frame)
        if chosen is None:
            if report["ok"]:
                chosen, status = report, "replanted"
            else:  # restore the planted legs exactly as they were
                pose = pl.Pose(sk, base.copy())
                orig_local = pl.fk_world(joints, original)[2]
                for side in contact:
                    for j in model[side].chain:
                        pose.local[j] = orig_local[j]
                pl._refresh(pose)
                chosen = dict(report, ok=True, reasons=[], _frame=pose.frame)
                status = "restored"
    return {
        "status": status,
        "sides": sorted(s for s, v in sides.items() if v[3]), "gross_sides": gross,
        "frame": chosen["_frame"],
        "alpha": chosen["alpha"],
        "steps": steps,
    }


def _gate(joints, sk, original, frame, contact, sides, floor, L, kp0, height,
          before_keys, rank, release_heel, gross):
    pos, rot, _ = pl.fk_world(joints, frame)
    reasons = []
    drift = 0.0
    model = feet(sk)
    for side, pts in contact.items():
        now = sole_points(pos, rot, model[side])
        for name, p in pts.items():
            if release_heel and name == "heel" and side in sides:
                continue
            d = now[name] - p
            drift = max(drift, abs(d[1]) if side in gross else float(np.linalg.norm(d)))
    if drift > GATE_CONTACT * L: reasons.append("접지 이동")
    lowest = min([pos[:, 1].min()] + [p[1] for f in model.values()
                                      for p in sole_points(pos, rot, f).values()])
    if lowest < floor - GATE_FLOOR * L: reasons.append("바닥 관통")
    gain = []
    for side, (ang0, _, normal, flagged) in sides.items():
        if not flagged:
            continue  # this leg only needs its contacts restored
        ang = ankle_angles(sk, rot, side, model[side].s)
        knee = knee_angles(sk, rot, side, model[side].s)[0]
        bounds = ankle_bounds(knee, side, 0)
        gain.append(excess(ang0, normal) - excess(ang, bounds))
    if gain and min(gain) <= 0.5: reasons.append("발목 개선 없음")
    after = pl.evaluate(joints, frame)
    for f in after["findings"]:
        old = before_keys.get((f["code"], f["side"]))
        if f["code"] not in ANKLE and (old is None or rank[f["severity"]] > rank[old]):
            reasons.append(f"새 문제: {f['label']}")
    kp, _ = _keypoints(joints, frame)
    silhouette = float(np.sqrt(((kp - kp0) ** 2).sum(1).mean()) / height)
    if silhouette > GATE_SILHOUETTE: reasons.append("실루엣 변화")
    shift = float(np.linalg.norm(pos[0] - pl.fk_world(joints, original)[0][0]))
    if shift > GATE_ROOT * L: reasons.append("몸 이동")
    return {"ok": not reasons, "reasons": reasons, "contact_drift": round(drift / L, 4),
            "silhouette": round(silhouette, 4), "ankle_gain": round(min(gain), 1) if gain else 0.0,
            "verdict": after["verdict"]}
