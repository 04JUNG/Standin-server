"""Author -> place -> measure -> bounded auto-separation -> verdict."""
from __future__ import annotations

import copy

import numpy as np

from .author import Author
from .rig import RIG, unit
from . import anatomy as A

REST = A.rest_ankles(RIG)
MAX_FIX_CM = 8.0       # larger automatic moves mean the recipe itself is wrong
SPEC_LIMITS = A.LIMITS


def measure(author: Author, spec: dict, partner_prims=None):
    sk = author.sk
    rot, pos = sk.world()
    lm = A.landmarks(sk.joints, pos)
    lm["head_rot"] = rot[sk.idx("Head")]
    rig = {"hips": rot[sk.idx("Hips")], "chest": rot[sk.idx("Spine2")], "head": rot[sk.idx("Head")]}
    metrics, bad = A.rom(lm, REST, rig)
    for key in ("L", "R"):
        arm = spec.get("arms", {}).get(key) or {}
        if abs(arm.get("pron", 0)) > SPEC_LIMITS["pron"]:
            bad.append(f"{key} 아래팔 회전 과다 {arm['pron']}°")
        fl = arm.get("flex", 0)
        ext_lim = SPEC_LIMITS["wrist_ext_load"] if arm.get("load") else SPEC_LIMITS["wrist_ext"]
        if fl > SPEC_LIMITS["wrist_flex"] or -fl > ext_lim:
            bad.append(f"{key} 손목 꺾임 과다 {fl}°")
        dv = arm.get("dev", 0)
        if dv > SPEC_LIMITS["wrist_rad"] or -dv > SPEC_LIMITS["wrist_uln"]:
            bad.append(f"{key} 손목 옆꺾임 과다 {dv}°")
    prims = A.primitives(lm)
    hits, detail = A.self_collisions(prims, detail=True)
    cross = {}
    cross_detail = {}
    if partner_prims is not None:
        cross, cross_detail = A.cross_collisions(prims, partner_prims, detail=True)
    props = [p for p in spec.get("props", []) if p.get("collide", True)]
    phits, pdetail = A.prop_collisions(prims, props, detail=True) if props else ({}, {})
    low = A.lowest(prims)
    floor = []
    if low < -A.FLOOR_TOL:
        floor.append(f"바닥 관통 {-low:.1f}cm")
    # Planted feet must actually rest on the floor (heel and ball).
    for key, side in (("L", "Left"), ("R", "Right")):
        leg = spec.get("legs", {}).get(key)
        planted = leg is None or (leg.get("frame", "w") in ("w", "g") and leg["to"][1] <= 12
                                  and leg.get("foot", {}).get("mode", "flat") == "flat"
                                  and not leg.get("foot", {}).get("pitch"))
        if leg is not None and leg.get("planted") is not None:
            planted = leg["planted"]
        if planted:
            sole = [p for p in prims if p.part == key + "foot"][0]
            heel = sole.a[1] - sole.ra
            ball = sole.b[1] - sole.rb
            if max(abs(heel), abs(ball)) > 2.0:
                floor.append(f"{key} 발이 바닥에 안 붙음(뒤꿈치 {heel:.1f}, 앞꿈치 {ball:.1f}cm)")
    sup = spec.get("support", "floor")
    extra = list(spec.get("support_points", [])) + A.prop_contacts(prims, props)
    if partner_prims is not None and spec.get("partner_support"):
        extra += [[p[0], p[2]] for p in spec["partner_support"]]
    balance = A.support(lm, prims, extra or None)
    if sup != "none" and balance["hull_distance_cm"] > A.BALANCE_TOL:
        floor.append(f"무게중심이 지지면 밖 {balance['hull_distance_cm']:.1f}cm(넘어짐)")
    for seat in [p for p in props if p["kind"] == "seat"]:
        bottom = min(min(q.a[1] - q.ra, q.b[1] - q.rb) for q in prims if q.part in ("pelvis", "glute"))
        if bottom - seat["max"][1] > 2.0:
            floor.append(f"엉덩이가 의자에서 뜸 {bottom - seat['max'][1]:.1f}cm")
    return {
        "lm": lm, "prims": prims, "metrics": metrics, "rom": bad,
        "collisions": hits, "collision_detail": detail,
        "props": phits, "prop_detail": pdetail,
        "cross": cross, "cross_detail": cross_detail,
        "floor": floor, "balance": balance, "lowest_cm": round(low, 2),
    }


def place(author: Author, spec: dict):
    mode = spec.get("place", "floor")
    if mode == "none":
        return 0.0
    rot, pos = author.sk.world()
    lm = A.landmarks(author.sk.joints, pos)
    lm["head_rot"] = rot[author.sk.idx("Head")]
    prims = A.primitives(lm)
    if mode != "all":   # arms never decide the floor height
        prims = [p for p in prims if p.group not in ("Larm", "Rarm")]
    low = A.lowest(prims)
    author.sk.root_pos[1] -= low
    return -low


def snap_props(author, eff):
    """Move leaning surfaces (wall, backrest, seat) onto the body surface.

    "snap": "z-" puts the prop's max-z face at the body's rearmost surface,
    "x-" its max-x face at the body's right-most surface, "y+" a seat's top at
    the pelvis underside. Moves are limited so furniture stays plausible.
    """
    props = eff.get("props", [])
    if not any(p.get("snap") for p in props):
        return
    rot, pos = author.sk.world()
    lm = A.landmarks(author.sk.joints, pos)
    lm["head_rot"] = rot[author.sk.idx("Head")]
    prims = A.primitives(lm)
    body = [p for p in prims if p.group not in ("Larm", "Rarm")]
    for prop in props:
        mode = prop.get("snap")
        if not mode:
            continue
        lo, hi = np.array(prop["min"], float), np.array(prop["max"], float)
        if mode == "z-":
            sel = [p for p in body if p.group in ("trunk", "head")]
            edge = min(min(p.a[2] - p.ra, p.b[2] - p.rb) for p in sel) + 0.4
            shift = np.array([0, 0, edge - hi[2]])
        elif mode == "z+":
            sel = [p for p in body if p.group in ("trunk", "head")]
            edge = max(max(p.a[2] + p.ra, p.b[2] + p.rb) for p in sel) - 0.4
            shift = np.array([0, 0, edge - lo[2]])
        elif mode == "x-":
            sel = [p for p in prims if p.group in ("trunk", "Rarm") and not "hand" in p.tags]
            edge = min(min(p.a[0] - p.ra, p.b[0] - p.rb) for p in sel) + 0.4
            shift = np.array([edge - hi[0], 0, 0])
        elif mode == "y+":
            pel = [p for p in prims if p.part in ("pelvis", "glute")]
            edge = min(min(q.a[1] - q.ra, q.b[1] - q.rb) for q in pel) + 0.6
            shift = np.array([0, edge - hi[1], 0])
            lo[1] = 0.0
            hi[1] += shift[1]
            shift = np.zeros(3)
            prop.setdefault("snap_shift_cm", 0)
            prop["snap_shift_cm"] = round(float(hi[1] - prop["max"][1]), 2)
        else:
            raise ValueError(mode)
        prop["min"] = [float(v) for v in lo + shift]
        prop["max"] = [float(v) for v in hi + shift]
        prop["snap"] = None
        prop["snapped"] = mode


def _mover(part):
    """Which IK target can resolve a collision on this primitive."""
    if part.group in ("Larm", "Rarm") and part.part[1:] != "uarm":
        return ("arms", part.group[0])
    if part.group in ("Lleg", "Rleg") and part.part[1:] != "thigh":
        return ("legs", part.group[0])
    return None


def settle(author, eff, span=15, step=3):
    """Rock a lying body on the floor until it rests stably (COM inside contacts)."""
    base = dict(eff.get("root", {}))
    best = None
    for dp in range(-span, span + 1, step):
        for dr in range(-span, span + 1, step):
            root = dict(base)
            root["pitch"] = base.get("pitch", 0) + dp
            root["roll"] = base.get("roll", 0) + dr
            trial = dict(eff, root=root)
            author.build(trial)
            place(author, trial)
            rot, pos = author.sk.world()
            lm = A.landmarks(author.sk.joints, pos)
            lm["head_rot"] = rot[author.sk.idx("Head")]
            prims = A.primitives(lm)
            sup = A.support(lm, prims)
            score = sup["hull_distance_cm"] + 0.15 * (abs(dp) + abs(dr)) - 0.05 * sup["contacts"]
            if best is None or score < best[0]:
                best = (score, root)
    eff["root"] = best[1]
    eff["settled"] = {"pitch_offset": best[1].get("pitch", 0) - base.get("pitch", 0),
                      "roll_offset": best[1].get("roll", 0) - base.get("roll", 0)}


def solve(spec: dict, author: Author | None = None, partner=None, max_iter=12):
    """Build a spec, separating contacts by moving distal IK targets.

    partner: optional (author, spec_effective) of an already-solved actor.
    Returns (author, effective_spec, report).
    """
    author = author or Author()
    eff = copy.deepcopy(spec)
    if eff.get("settle"):
        settle(author, eff)
    moved = {}
    report = None
    partner_prims = None
    if partner is not None:
        if isinstance(partner, tuple):
            author.partner, partner_prims = partner
            author.partner_prefix = "O"
        else:
            partner_prims = partner
    for it in range(max_iter):
        info = author.build(eff)
        place(author, eff)
        if it == 0:
            snap_props(author, eff)
            for n in range(10):
                if n in (0, 3, 6):
                    for key, side in (("L", "Left"), ("R", "Right")):
                        limb = eff.get("arms", {}).get(key)
                        if limb and "palm" in limb and limb.get("face", "auto") is not None \
                                and "pron_fixed" not in limb:
                            orient_palm(author, eff, key, side)
                    info = author.build(eff)
                    place(author, eff)
                if not _palm_step(author, eff, info):
                    break
                info = author.build(eff)
                place(author, eff)
        report = measure(author, eff, partner_prims)
        report["ik"] = info
        hits = dict(report["collisions"])
        hits.update(report["cross"])
        hits.update(report["props"])
        if not hits:
            break
        frames = author.frames()
        pushes = {}
        details = dict(report["collision_detail"])
        details.update(report["cross_detail"])
        details.update(report["prop_detail"])
        for key, depth in hits.items():
            entry = details[key]
            d, (pa, qa), p, q = entry[:4]
            movers = []
            mp = _mover(p)
            mq = _mover(q) if q is not None else None
            is_cross = key.startswith("A.")
            if mp and (mp[0] == "arms" or not mq):
                movers.append((mp, unit(pa - qa)))
            if mq and not is_cross and (mq[0] == "arms" or not mp):
                movers.append((mq, unit(qa - pa)))
            if not movers and mp:
                movers.append((mp, unit(pa - qa)))
            share = 1.0 / max(1, len(movers))
            for (kind, side), direction in movers:
                pushes.setdefault((kind, side), np.zeros(3))
                tol = A.SOFT_PAIRS.get(key, A.PENETRATION_TOL)
                pushes[(kind, side)] += direction * (depth - tol + 0.6) * share
        if not pushes:
            break
        for (kind, side), delta in pushes.items():
            limb = eff.setdefault(kind, {}).get(side)
            if limb is None:
                break
            if kind == "legs" and limb.get("frame", "w") in ("w", "g") and limb["to"][1] <= 12:
                delta = delta.copy()
                delta[1] = max(delta[1], 0.0)  # planted feet slide, never sink
            frame = limb.get("frame", "c" if kind == "arms" else "w")
            m = np.eye(3) if frame == "g" else frames[frame][1]
            local = m.T @ delta
            limb["to"] = [float(v) for v in np.asarray(limb["to"], float) + local]
            moved[(kind, side)] = moved.get((kind, side), np.zeros(3)) + delta
    if eff.get("bar"):
        grips = [palm_point(author, s_, True) for s_ in ("Left", "Right")]
        c = np.mean(grips, axis=0)
        eff.setdefault("props", []).append({"kind": "bar", "min": [-60, c[1] - 1.6, c[2] - 1.6],
                                            "max": [60, c[1] + 1.6, c[2] + 1.6], "collide": False})
    report["iterations"] = it + 1
    report["auto_moves_cm"] = {f"{k}.{s}": round(float(np.linalg.norm(v)), 2) for (k, s), v in moved.items()}
    big = [k for k, v in report["auto_moves_cm"].items() if v > MAX_FIX_CM]
    if big:
        report["rom"].append("자동 보정량 과다: " + ", ".join(big))
    for k, v in report["ik"]["ik_error_cm"].items():
        if v > 4.0:
            report["rom"].append(f"{k} 목표 미도달 {v:.1f}cm")
    frames = author.frames()
    for key, side in (("L", "Left"), ("R", "Right")):
        limb = eff.get("arms", {}).get(key)
        if limb and "palm" in limb:
            pframe = limb.get("pframe", limb.get("frame", "c"))
            target = author.to_world(limb["palm"], pframe, frames)
            miss = float(np.linalg.norm(target - palm_point(author, side, limb.get("grip", False))))
            report.setdefault("palm_miss_cm", {})[key] = round(miss, 2)
            if miss > 5.0:
                report["rom"].append(f"{key} 손 접촉점 미도달 {miss:.1f}cm")
            elif miss > 3.0:
                report.setdefault("warnings", []).append(f"{key} 손 접촉 {miss:.1f}cm 떨어짐")
    report["passed"] = not (report["rom"] or report["collisions"] or report["cross"]
                            or report["props"] or report["floor"])
    return author, eff, report


def palm_point(author, side, grip=False):
    sk = author.sk
    rot, pos = sk.world()
    wrist = pos[sk.idx(side + "Hand")]
    mid = pos[sk.idx(side + "HandMiddle1")]
    ring = pos[sk.idx(side + "HandRing1")]
    idx = pos[sk.idx(side + "HandIndex1")]
    center = wrist + ((mid + ring + idx) / 3 - wrist) * 0.6
    normal = rot[sk.idx(side + "Hand")] @ np.array([0, -1.0, 0])
    tip = pos[sk.idx(side + "HandMiddle3")]
    if grip == "tip":
        return pos[sk.idx(side + "HandIndex3_End")]
    if grip:
        return center
    # palm skin, nudged toward the finger pads when fingers curl
    return center + normal * 2.5 + (tip - center) * 0.1


def _surface_normal(author, point):
    """Outward normal of the nearest non-arm body surface at a world point."""
    rot, pos = author.sk.world()
    lm = A.landmarks(author.sk.joints, pos)
    lm["head_rot"] = rot[author.sk.idx("Head")]
    best = None
    cands = [p for p in A.primitives(lm) if p.group not in ("Larm", "Rarm")]
    if getattr(author, "partner", None) is not None:
        cands += actor_prims(author.partner)
    for p in cands:
        ab = p.b - p.a
        t = 0.0 if ab @ ab < 1e-9 else float(np.clip((point - p.a) @ ab / (ab @ ab), 0, 1))
        c = p.a + ab * t
        r = p.ra + (p.rb - p.ra) * t
        d = np.linalg.norm(point - c) - r
        if best is None or d < best[0]:
            best = (d, unit(point - c))
    return best[1]


def orient_palm(author, eff, key, side):
    """Pick forearm roll and wrist angles (inside ROM) so the palm faces its contact."""
    limb = eff["arms"][key]
    face = limb.get("face", "auto")
    if face is None:
        return
    frames = author.frames()
    pframe = limb.get("pframe", limb.get("frame", "c"))
    target = author.to_world(limb["palm"], pframe, frames)
    if face == "auto":
        desired = -_surface_normal(author, target)
    elif face == "down":
        desired = np.array([0, -1.0, 0])
    else:
        desired = unit(author.to_world(face, pframe, frames, direction=True))
    lim = A.LIMITS
    ext_lim = lim["wrist_ext_load"] if limb.get("load") else lim["wrist_ext"]
    rot, pos = author.sk.world()
    lm = A.landmarks(author.sk.joints, pos)
    lm["head_rot"] = rot[author.sk.idx("Head")]
    others = [p for p in A.primitives(lm) if p.group != key + "arm"]
    opts, orad = [], []
    for q in others:
        pts, r = A._sample(q, 7)
        opts.append(pts)
        orad.append(r)
    opts, orad = np.vstack(opts), np.concatenate(orad)
    props = eff.get("props", [])

    def evaluate(pron, flex):
        trial = dict(limb, pron=pron, flex=flex)
        author._arm(side, trial, frames, eff.get("clav", {}).get(key), {"ik_error_cm": {}})
        rot, pos = author.sk.world()
        hand = rot[author.sk.idx(side + "Hand")]
        normal = hand @ np.array([0, -1.0, 0])
        fingers = hand @ np.array([1.0 if side == "Left" else -1.0, 0, 0])
        lm = A.landmarks(author.sk.joints, pos)
        mine = [p for p in A.primitives(lm) if p.group == key + "arm" and "hand" in p.tags]
        hp, hr = [], []
        for p in mine:
            pts, r = A._sample(p, 4)
            hp.append(pts)
            hr.append(r)
        hp, hr = np.vstack(hp), np.concatenate(hr)
        # evaluate the hand as it will sit once the palm reaches its anchor
        hp = hp + (target - palm_point(author, side, limb.get("grip", False)))
        d = np.linalg.norm(hp[:, None] - opts[None], axis=-1) - hr[:, None] - orad[None]
        pen = max(0.0, float(-d.min()), float(-(hp[:, 1] - hr).min()))
        for prop in props:
            sd, _ = A.box_signed_distance(hp, prop["min"], prop["max"])
            pen = max(pen, float((hr - sd).max()))
        return (1 - normal @ desired + 0.002 * abs(pron) / 10 + 0.003 * abs(flex) / 10
                + 2.0 * max(0.0, fingers @ desired - 0.1) + 0.6 * max(0.0, pen - 0.5)
                + (0.6 * (1 - fingers @ fdir) if fdir is not None else 0.0))

    fdir = limb.get("fdir")
    if fdir is not None:
        fdir = unit(author.to_world(fdir, pframe, frames, direction=True))
    flex_lo = -int(ext_lim)
    best = None
    for pron in range(-80, 81, 20):
        for flex in range(-60, 61, 15):
            flex = max(flex, flex_lo)
            c = evaluate(pron, flex)
            if best is None or c < best[0]:
                best = (c, pron, flex)
    _, p0, f0 = best
    for pron in range(p0 - 10, p0 + 11, 5):
        for flex in range(f0 - 10, f0 + 11, 5):
            if abs(pron) > 85 or flex > 60 or flex < flex_lo:
                continue
            c = evaluate(pron, flex)
            if c < best[0]:
                best = (c, pron, flex)
    limb["pron"], limb["flex"] = best[1], best[2]


def _palm_step(author, eff, info=None):
    """Move wrist IK targets so palm contact points land on their anchors."""
    moved = False
    frames = author.frames()
    for key, side in (("L", "Left"), ("R", "Right")):
        limb = eff.get("arms", {}).get(key)
        if not limb or "palm" not in limb:
            continue
        pframe = limb.get("pframe", limb.get("frame", "c"))
        target = author.to_world(limb["palm"], pframe, frames)
        cur = palm_point(author, side, limb.get("grip", False))
        delta = target - cur
        if info and info["ik_error_cm"].get(side + "Arm", 0) > 1.0:
            # unreachable: pull the target back onto the reachable sphere
            sh = author.sk.position(side + "Arm")
            rot, pos = author.sk.world()
            wrist = pos[author.sk.idx(side + "Hand")]
            delta = wrist - author.to_world(limb["to"], limb.get("frame", "c"), frames)
        if np.linalg.norm(delta) < 0.3:
            continue
        frame = limb.get("frame", "c")
        m = np.eye(3) if frame == "g" else frames[frame][1]
        limb["to"] = [float(v) for v in np.asarray(limb["to"], float) + m.T @ delta]
        moved = True
    return moved


def mirror_frame(author: Author):
    """Mirror the posed skeleton across the sagittal plane (S R S, swap sides)."""
    sk = author.sk
    S = np.diag([-1.0, 1.0, 1.0])
    old = {i: m.copy() for i, m in sk.local.items()}
    for i, j in enumerate(sk.joints):
        if j[4]:
            continue
        name = j[0]
        peer = name.replace("Left", "@").replace("Right", "Left").replace("@", "Right")
        k = sk.idx(peer)
        sk.local[i] = S @ old[k] @ S
    sk.root_pos[0] *= -1


def strip_partner(spec):
    """Copy of a spec whose limbs referencing the other actor fall back to rest."""
    out = copy.deepcopy(spec)
    for kind in ("arms", "legs"):
        for key, limb in list(out.get(kind, {}).items()):
            if limb and (str(limb.get("frame", "")).startswith("O.") or str(limb.get("pframe", "")).startswith("O.")):
                out[kind].pop(key)
    return out


def actor_prims(author):
    rot, pos = author.sk.world()
    lm = A.landmarks(author.sk.joints, pos)
    lm["head_rot"] = rot[author.sk.idx("Head")]
    return A.primitives(lm)


def solve_set(spec_a, spec_b, passes=3):
    """Solve two actors that touch each other; returns both authors/specs/reports."""
    a, b = Author(), Author()
    solve(strip_partner(spec_b), b)
    results = {}
    order = ["a", "b"] * passes
    authors = {"a": a, "b": b}
    specs = {"a": spec_a, "b": spec_b}
    for who in order:
        other = "b" if who == "a" else "a"
        if other not in results and other == "a":
            solve(strip_partner(spec_a), a)
        author, eff, rep = solve(specs[who], authors[who],
                                 partner=(authors[other], actor_prims(authors[other])))
        results[who] = (author, eff, rep)
    # final cross-check with both actors frozen
    pa, pb = actor_prims(a), actor_prims(b)
    hits, closest = A.cross_collisions(pa, pb)
    return results, {"cross_hits": hits, "closest_cm": None if closest is None else round(-closest, 2)}
