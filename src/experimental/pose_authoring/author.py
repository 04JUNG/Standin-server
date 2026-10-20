"""Explicit 3D pose authoring on the Quaternius rig (cm, degrees).

A pose spec is a plain dict. Everything is an artist decision; nothing is
inferred from images. Limb placement uses analytic two-bone IK with fixed bone
lengths, elbow/knee hinges kept in anatomical planes, and explicit forearm
pronation / wrist angles. Validation lives in anatomy.py.

Spec keys (all optional unless noted)
  root:   {"y": hip height cm, "yaw","pitch","roll": deg, "x","z": cm}
  spine:  {"pitch": fwd+, "roll": left+, "yaw": left+}
  head:   {"yaw": left+, "pitch": down+, "roll": left+}
  clav:   {"L": [elev, protract], "R": [...]}   (auto if omitted)
  arms:   {"L": arm, "R": arm}; arm = {"to": [x,y,z], "frame": "c|p|h|w",
           "pole": [x,y,z] (elbow direction, same frame, optional),
           "pron": deg (+ palm down when forearm forward), "flex": deg,
           "dev": deg (radial+)}
  legs:   {"L": leg, "R": leg}; leg = {"to": ankle [x,y,z], "frame": "w|p",
           "pole": knee direction (same frame), "foot": {...}}
           foot = {"mode": "flat"|"free", "yaw": deg (body frame),
                   "pitch": deg (toe down+), "roll": deg, "toe": deg (toe bend up+)}
  hands:  {"L": preset, "R": preset}
  place:  "floor" (lowest point to y=0) | "none"
Frames: w = body-world (yaw of root only; origin on floor under root x/z),
        p = pelvis (Hips joint), c = chest (Spine2), h = head (Head joint).
Axes: +X actor-left, +Y up, +Z forward.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from .rig import Skeleton, rot_between, frame_from, unit, FINGERS

# Per-finger weights (Thumb, Index, Middle, Ring, Pinky) applied to the captured
# fist rotation of the CC0 source rig. 0 = straight, 1 = captured fist.
HAND_PRESETS = {
    "open": (0.0, 0.0, 0.0, 0.0, 0.0),
    "flat": (0.05, 0.02, 0.02, 0.02, 0.02),
    "relaxed": (0.2, 0.22, 0.26, 0.3, 0.34),
    "soft": (0.3, 0.35, 0.4, 0.45, 0.5),
    "cup": (0.45, 0.55, 0.6, 0.62, 0.65),
    "grip": (0.75, 0.82, 0.85, 0.87, 0.9),
    "fist": (1.0, 1.0, 1.0, 1.0, 1.0),
    "pinch": (0.7, 0.45, 0.6, 0.65, 0.7),
    "point": (0.85, 0.0, 1.0, 1.0, 1.0),
    "v": (0.9, 0.0, 0.0, 1.0, 1.0),
    "thumb": (0.0, 1.0, 1.0, 1.0, 1.0),
    "ok": (0.55, 0.62, 0.1, 0.08, 0.06),
    "phone": (0.35, 0.45, 0.5, 0.55, 0.6),
    "shh": (0.85, 0.0, 0.95, 1.0, 1.0),
}

SPINE_SHARE = (("Spine", 0.30), ("Spine1", 0.35), ("Spine2", 0.35))


def _euler_ypr(yaw, pitch, roll):
    """Body convention: yaw left+, pitch forward+, roll toward actor-left+."""
    return (R.from_euler("Y", yaw, degrees=True)
            * R.from_euler("X", pitch, degrees=True)
            * R.from_euler("Z", -roll, degrees=True)).as_matrix()


def rotvec(axis, deg):
    return R.from_rotvec(unit(axis) * np.deg2rad(deg)).as_matrix()


class Author:
    def __init__(self, skeleton: Skeleton | None = None):
        self.sk = skeleton or Skeleton()
        sk = self.sk
        self.len = {s: (np.linalg.norm(sk.child_offset(s + "Arm")),
                        np.linalg.norm(sk.child_offset(s + "ForeArm")))
                    for s in ("Left", "Right")}
        self.leg_len = {s: (np.linalg.norm(sk.child_offset(s + "UpLeg")),
                            np.linalg.norm(sk.child_offset(s + "Leg")))
                        for s in ("Left", "Right")}

    # -------------------------------------------------------------------
    partner = None
    partner_prefix = "B"

    def frames(self):
        out = self.base_frames()
        if self.partner is not None:
            for k, v in self.partner.base_frames().items():
                out[self.partner_prefix + "." + k] = v
        return out

    def base_frames(self):
        """Origin/rotation of the authoring frames after trunk posing."""
        sk = self.sk
        rot, pos = sk.world()
        yaw = self.root_yaw
        body = R.from_euler("Y", yaw, degrees=True).as_matrix()
        root = pos[sk.idx("Hips")]
        yaw = getattr(self, "root_yaw", 0.0)
        body = R.from_euler("Y", yaw, degrees=True).as_matrix()
        return {
            "w": (np.array([root[0], 0.0, root[2]]), body),
            "p": (root, rot[sk.idx("Hips")]),
            "c": (pos[sk.idx("Spine2")], rot[sk.idx("Spine2")]),
            "h": (pos[sk.idx("Head")], rot[sk.idx("Head")]),
            "g": (np.zeros(3), np.eye(3)),
            "tL": (pos[sk.idx("LeftUpLeg")], rot[sk.idx("LeftUpLeg")]),
            "tR": (pos[sk.idx("RightUpLeg")], rot[sk.idx("RightUpLeg")]),
            "kL": (pos[sk.idx("LeftLeg")], rot[sk.idx("LeftLeg")]),
            "kR": (pos[sk.idx("RightLeg")], rot[sk.idx("RightLeg")]),
            "hL": (pos[sk.idx("LeftHand")], rot[sk.idx("LeftHand")]),
            "hR": (pos[sk.idx("RightHand")], rot[sk.idx("RightHand")]),
        }

    def to_world(self, vec, frame, frames, direction=False):
        origin, m = frames[frame]
        v = m @ np.asarray(vec, float)
        return v if direction else origin + v

    # -------------------------------------------------------------------
    def build(self, spec: dict) -> dict:
        sk = self.sk
        sk.reset()
        info = {"ik_error_cm": {}, "clamped": {}}
        root = spec.get("root", {})
        self.root_yaw = root.get("yaw", 0.0)
        sk.root_pos = np.array([root.get("x", 0.0), root.get("y", 91.7), root.get("z", 0.0)])
        sk.set_local("Hips", _euler_ypr(root.get("yaw", 0), root.get("pitch", 0), root.get("roll", 0)))
        sp = spec.get("spine", {})
        for name, share in SPINE_SHARE:
            sk.set_local(name, _euler_ypr(sp.get("yaw", 0) * share, sp.get("pitch", 0) * share,
                                          sp.get("roll", 0) * share))
        hd = spec.get("head", {})
        for name, share in (("Neck", 0.4), ("Head", 0.6)):
            sk.set_local(name, _euler_ypr(hd.get("yaw", 0) * share, hd.get("pitch", 0) * share,
                                          hd.get("roll", 0) * share))
        frames = self.frames()
        # ---- legs ------------------------------------------------------
        for key, side in (("L", "Left"), ("R", "Right")):
            leg = spec.get("legs", {}).get(key)
            if leg is None:
                sign = 1 if key == "L" else -1
                leg = {"to": [9 * sign, 10.4, -1.0], "frame": "w"}
            self._leg(side, leg, frames, info)
        frames = self.frames()
        # ---- arms ------------------------------------------------------
        for key, side in (("L", "Left"), ("R", "Right")):
            arm = spec.get("arms", {}).get(key)
            if arm is None:
                arm = {"to": [25.0 * (1 if key == "L" else -1), -4.0, 6.0], "frame": "p"}
            self._arm(side, arm, frames, spec.get("clav", {}).get(key), info)
        # ---- hands -----------------------------------------------------
        for key, side in (("L", "Left"), ("R", "Right")):
            self._fingers(side, spec.get("hands", {}).get(key, "relaxed"))
        return info

    # -------------------------------------------------------------------
    def _arm(self, side, arm, frames, clav, info):
        sk = self.sk
        sign = 1 if side == "Left" else -1
        frame = arm.get("frame", "c")
        target = self.to_world(arm["to"], frame, frames)
        chest_rot = frames["c"][1]
        # Clavicle: automatic mild elevation/protraction from the target.
        if clav is None:
            shoulder0 = sk.position(side + "Arm")
            rel = chest_rot.T @ (target - shoulder0)
            d = unit(rel)
            elev = np.rad2deg(np.arcsin(np.clip(d[1], -1, 1)))
            elev_c = float(np.clip((elev - 20) * 0.22, 0, 14))
            prot_c = float(np.clip(d[2] * 8 - max(0, -d[0] * sign) * 0, -6, 10))
            if d[0] * sign < 0:  # reaching across the body protracts the scapula
                prot_c = float(np.clip(prot_c + (-d[0] * sign) * 10, -6, 16))
            clav = [elev_c, prot_c]
        elev, prot = clav
        local = (R.from_euler("Z", sign * elev, degrees=True)
                 * R.from_euler("Y", -sign * prot, degrees=True)).as_matrix()
        sk.set_local(side + "Shoulder", local)
        info.setdefault("clavicle", {})[side] = [round(elev, 2), round(prot, 2)]
        shoulder = sk.position(side + "Arm")
        a, b = self.len[side]
        if "pole" in arm:
            pole_dir = self.to_world(arm["pole"], frame, frames, direction=True)
        else:
            # Default elbow direction: down, slightly back and out.
            pole_dir = chest_rot @ np.array([0.35 * sign, -0.8, -0.45])
        elbow, wrist, err = _two_bone(shoulder, target, pole_dir, a, b)
        info["ik_error_cm"][side + "Arm"] = round(err, 3)
        u, f = unit(elbow - shoulder), unit(wrist - elbow)
        d = f - u * (u @ f)
        if np.linalg.norm(d) < np.sin(np.deg2rad(4)):
            pp = pole_dir - u * (u @ pole_dir)
            d = -pp
        d = unit(d)
        rest_u = unit(sk.child_offset(side + "Arm"))
        rest_f = unit(sk.child_offset(side + "ForeArm"))
        z = np.array([0.0, 0.0, 1.0])
        upper = frame_from(u, d) @ frame_from(rest_u, z).T
        sk.set_global(side + "Arm", upper)
        h = np.cross(u, d)
        lower = frame_from(f, np.cross(h, f)) @ frame_from(rest_f, z).T
        pron = arm.get("pron", 0.0)
        lower = rotvec(f, sign * pron) @ lower
        sk.set_global(side + "ForeArm", lower)
        palm = lower @ np.array([0, -1.0, 0])
        thumb = lower @ np.array([0, 0, 1.0])
        hand = lower
        if arm.get("dev"):
            hand = rotvec(np.cross(f, thumb), arm["dev"]) @ hand
        if arm.get("flex"):
            hand = rotvec(np.cross(f, palm), arm["flex"]) @ hand
        sk.set_global(side + "Hand", hand)

    def _leg(self, side, leg, frames, info):
        sk = self.sk
        sign = 1 if side == "Left" else -1
        frame = leg.get("frame", "w")
        target = self.to_world(leg["to"], frame, frames)
        hip = sk.position(side + "UpLeg")
        a, b = self.leg_len[side]
        if "pole" in leg:
            pole_dir = self.to_world(leg["pole"], frame, frames, direction=True)
        else:
            pole_dir = frames["p"][1] @ np.array([0.12 * sign, 0.0, 1.0])
        knee, ankle, err = _two_bone(hip, target, pole_dir, a, b)
        info["ik_error_cm"][side + "Leg"] = round(err, 3)
        u, f = unit(knee - hip), unit(ankle - knee)
        d = f - u * (u @ f)
        if np.linalg.norm(d) < np.sin(np.deg2rad(3)):
            d = -(pole_dir - u * (u @ pole_dir))
        d = unit(d)
        rest_u = unit(sk.child_offset(side + "UpLeg"))
        rest_f = unit(sk.child_offset(side + "Leg"))
        back = np.array([0.0, 0.0, -1.0])
        upper = frame_from(u, d) @ frame_from(rest_u, back).T
        sk.set_global(side + "UpLeg", upper)
        h = np.cross(u, d)
        lower = frame_from(f, np.cross(h, f)) @ frame_from(rest_f, back).T
        sk.set_global(side + "Leg", lower)
        foot = leg.get("foot", {})
        mode = foot.get("mode", "flat")
        knee_fwd = -d  # knee points opposite to shin flexion
        if mode == "flat":
            if "yaw" in foot:
                yaw = self.root_yaw + foot["yaw"]
            else:
                kf = knee_fwd.copy()
                kf[1] = 0
                if np.linalg.norm(kf) < 0.2:
                    kf = frames["p"][1] @ np.array([0, 0, 1.0])
                    kf[1] = 0
                kf = unit(kf)
                yaw = np.rad2deg(np.arctan2(kf[0], kf[2]))
            m = (R.from_euler("Y", yaw, degrees=True)
                 * R.from_euler("X", foot.get("pitch", 0), degrees=True)
                 * R.from_euler("Z", -foot.get("roll", 0) * sign, degrees=True)).as_matrix()
        else:  # foot follows shin with explicit ankle angles
            m = (lower @ R.from_euler("X", foot.get("pitch", 0), degrees=True).as_matrix())
        sk.set_global(side + "Foot", m)
        if foot.get("toe"):
            sk.set_local(side + "ToeBase", R.from_euler("X", -foot["toe"], degrees=True).as_matrix())

    def _fingers(self, side, preset):
        sk = self.sk
        weights = HAND_PRESETS[preset] if isinstance(preset, str) else preset
        cap = sk.captured
        for i, j in enumerate(sk.joints):
            name = j[0]
            if j[4] or not name.startswith(side + "Hand") or name == side + "Hand":
                continue
            finger = next(f for f in FINGERS if f in name)
            w = weights[FINGERS.index(finger)]
            start = sk.starts[i]
            axes = "".join(c[0] for c in j[3] if c.endswith("rotation"))
            vals = [cap[start + n] for n, c in enumerate(j[3]) if c.endswith("rotation")]
            rv = R.from_euler(axes, vals, degrees=True).as_rotvec()
            local = R.from_rotvec(rv * w).as_matrix()
            if name.endswith("Thumb1") and w < 0.6:
                # The rest thumb is opposed ~6 cm below the palm; bring it into
                # the palm plane beside the index for open shapes.
                sgn = 1.0 if side == "Left" else -1.0
                rest = np.array([0.73 * sgn, -0.47, 0.52])
                want = np.array([0.86 * sgn, 0.04, 0.50])
                fold = rot_between(rest, want)
                blend = 1.0 - w / 0.6
                fold = R.from_rotvec(R.from_matrix(fold).as_rotvec() * blend).as_matrix()
                local = fold @ local
            sk.local[i] = local


def _two_bone(origin, target, pole, a, b):
    direction = np.asarray(target, float) - origin
    req = float(np.linalg.norm(direction))
    axis = direction / req
    dist = float(np.clip(req, abs(a - b) + 0.5, (a + b) * 0.998))
    n = np.asarray(pole, float) - axis * (axis @ pole)
    if np.linalg.norm(n) < 1e-6:
        n = np.cross(axis, [1.0, 0, 0])
    n = unit(n)
    along = (a * a + dist * dist - b * b) / (2 * dist)
    mid = origin + along * axis + np.sqrt(max(0.0, a * a - along * along)) * n
    end = origin + dist * axis
    return mid, end, abs(dist - req)
