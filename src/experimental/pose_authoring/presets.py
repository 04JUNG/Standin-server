"""Reusable stance / prop building blocks for pose specs (cm, degrees)."""
from __future__ import annotations

import copy

SEAT_H = 45.0
DESK_H = 73.0


def merge(*parts):
    out = {}
    for p in parts:
        for k, v in copy.deepcopy(p).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k].update(v)
            elif k in ("props", "support_points") and k in out:
                out[k] = out[k] + v
            else:
                out[k] = v
    return out


def leg(x, y, z, pole=None, frame="w", planted=None, **foot):
    d = {"to": [x, y, z], "frame": frame}
    if planted is not None:
        d["planted"] = planted
    if pole is not None:
        d["pole"] = pole
    if foot:
        d["foot"] = foot
    return d


def arm(x, y, z, frame="c", pole=None, pron=0, flex=0, dev=0):
    d = {"to": [x, y, z], "frame": frame}
    if pole is not None:
        d["pole"] = pole
    if pron:
        d["pron"] = pron
    if flex:
        d["flex"] = flex
    if dev:
        d["dev"] = dev
    return d


def stand(width=9.5, lead=0.0, y=91.2, toe_out=8, **extra):
    """Upright stance. lead: left foot forward (cm, right back by same)."""
    return merge({
        "root": {"y": y},
        "legs": {"L": leg(width, 10.4, -1 + lead, foot_yaw(toe_out)),
                 "R": leg(-width, 10.4, -1 - lead, foot_yaw(-toe_out))},
    }, extra)


def foot_yaw(deg):
    # knee pole that also turns the foot; returned as pole direction
    import math
    r = math.radians(deg)
    return [math.sin(r) * 1.0, 0.0, math.cos(r)]


def chair(seat=SEAT_H, knee_spread=11, feet_z=40, feet_spread=12, lean_back=False):
    s = {
        "root": {"y": seat + 8.4},
        "legs": {"L": leg(feet_spread, 10.4, feet_z, [0.15, 0.2, 1]),
                 "R": leg(-feet_spread, 10.4, feet_z, [-0.15, 0.2, 1])},
        "props": [{"kind": "seat", "min": [-23, 0, -20], "max": [23, seat, 24]}],
        "support": "seat",
    }
    return s


def desk(z0=24, h=DESK_H):
    return {"props": [{"kind": "desk", "min": [-60, h - 4, z0], "max": [60, h, z0 + 60]}]}


def wall_behind(z=-14):
    return {"props": [{"kind": "wall", "min": [-80, 0, z - 10], "max": [80, 200, z], "wall": True,
                       "supports": True, "snap": "z-"}]}


def floor_sit():
    return {"support": "floor"}


# Skin-surface anchors for palm contacts (frame, xyz). Head frame origin is the
# skull base; chest frame origin Spine2; pelvis frame origin Hips.
ANCHOR = {
    "mouth": ("h", [0, 2.0, 11.8]),
    "chin": ("h", [0, -3.5, 10.5]),
    "nose": ("h", [0, 6.5, 12.5]),
    "eyeL": ("h", [3.6, 8.0, 11.0]), "eyeR": ("h", [-3.6, 8.0, 11.0]),
    "forehead": ("h", [0, 12.5, 10.0]),
    "top": ("h", [0, 18.0, 0.0]),
    "crownL": ("h", [6.0, 15.5, -1.0]), "crownR": ("h", [-6.0, 15.5, -1.0]),
    "templeL": ("h", [9.5, 10.0, 3.0]), "templeR": ("h", [-9.5, 10.0, 3.0]),
    "earL": ("h", [10.5, 6.5, 0.5]), "earR": ("h", [-10.5, 6.5, 0.5]),
    "backhead": ("h", [0, 8.0, -9.0]),
    "cheekL": ("h", [7.5, 3.5, 8.0]), "cheekR": ("h", [-7.5, 3.5, 8.0]),
    "chestL": ("c", [7.0, 2.0, 12.5]), "chestR": ("c", [-7.0, 2.0, 12.5]),
    "sternum": ("c", [0.0, 0.0, 13.0]),
    "belly": ("p", [0.0, 12.0, 12.5]),
    "hipL": ("p", [19.5, 13.0, 1.0]), "hipR": ("p", [-19.5, 13.0, 1.0]),
    # thigh front/top (rest +Z is the front of the thigh) and knee caps
    "thighL": ("tL", [0.5, -24.0, 6.8]), "thighR": ("tR", [-0.5, -24.0, 6.8]),
    "kneeL": ("kL", [0.0, 1.0, 5.6]), "kneeR": ("kR", [0.0, 1.0, 5.6]),
    "shinL": ("kL", [0.0, -14.0, 4.8]), "shinR": ("kR", [0.0, -14.0, 4.8]),
    "back": ("c", [0.0, 0.0, -12.5]), "backL": ("c", [8.0, 0.0, -12.0]), "backR": ("c", [-8.0, 0.0, -12.0]),
    "lowbackL": ("p", [9.0, 18.0, -10.5]), "lowbackR": ("p", [-9.0, 18.0, -10.5]),
    "shoulderL": ("c", [16.0, 18.5, -6.0]), "shoulderR": ("c", [-16.0, 18.5, -6.0]),
    "collar": ("c", [0.0, 11.0, 11.5]),
    "waistL": ("p", [13.5, 20.0, 1.0]), "waistR": ("p", [-13.5, 20.0, 1.0]),
    "wristL": ("hL", [0.0, 0.0, 0.0]), "wristR": ("hR", [0.0, 0.0, 0.0]),
    "handL": ("hL", [9.0, -2.0, 0.0]), "handR": ("hR", [-9.0, -2.0, 0.0]),
}


def touch(name, offset=(0, 0, 0), **kw):
    """Arm spec that puts the palm skin on an anatomical anchor."""
    frame, xyz = ANCHOR[name]
    target = [a + b for a, b in zip(xyz, offset)]
    d = {"to": list(target), "frame": frame, "palm": target, "pframe": frame}
    d.update(kw)
    return d


def palm_at(x, y, z, frame="w", face="down", load=None, **kw):
    d = {"to": [x, y, z], "frame": frame, "palm": [x, y, z], "pframe": frame, "face": face}
    if load is None:
        load = frame in ("w", "g") and y < 3   # palm on the floor carries weight
    if load:
        d["load"] = True
    d.update(kw)
    return d


def folded(lower="L", y=-12.0, z=0.0):
    """Crossed arms (팔짱) in the chest frame; `lower` forearm sits underneath."""
    up = "R" if lower == "L" else "L"
    sl = 1 if lower == "L" else -1
    return {
        lower: {"to": [-8 * sl, y, 20 + z], "frame": "c", "pole": [1 * sl, -1, 0.25], "pron": 70},
        up: {"to": [7 * sl, y + 6, 24 + z], "frame": "c", "pole": [-1 * sl, -1, 0.35], "pron": 70},
    }


def ptouch(name, offset=(0, 0, 0), **kw):
    """Palm contact on the OTHER actor's anatomy (two-person sets)."""
    frame, xyz = ANCHOR[name]
    target = [a + b for a, b in zip(xyz, offset)]
    d = {"to": list(target), "frame": "O." + frame, "palm": target, "pframe": "O." + frame}
    d.update(kw)
    return d
