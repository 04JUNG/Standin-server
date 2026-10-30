"""Transactional, destination-surface application of offline contact parts.

One algorithm for every recipe/side/pose. Targets are anatomical surface queries;
the optimizer has no image, evaluation judgment, case ID or world-space override.
It adjusts an elbow swivel and a rigid hand contact frame, then checks native skin.
"""
import math
import numpy as np
import bpy
from mathutils import Matrix, Vector, Quaternion, Euler
from mathutils.bvhtree import BVHTree

from converter.hand_library import load_library
from converter.hand_geometry import hand_frame, FingerRest, finger_rotations
from src.experimental.part_composition import shortest_arc
from src.scene_refinement.rig import validate_rig
from .blender_adapter import snapshot, body_frame
from .contact_parts import admission_reasons, contact_axis


def unit(v):
    v = Vector(v)
    if v.length < 1e-8:
        raise ValueError("degenerate contact frame")
    return v.normalized()


def frame(u, n):
    u = unit(u); n = unit(Vector(n) - u * u.dot(Vector(n)))
    return Matrix((u, n, u.cross(n))).transposed()


def bones(arm):
    return {p.name.split(":")[-1]: p for p in arm.pose.bones}


def region(name):
    n = name.split(":")[-1]
    if n.startswith(("Spine", "Hips")):
        return "Torso"
    if n in ("Head", "Neck"):
        return "Head"
    for side in ("Left", "Right"):
        if n.startswith(side + "Hand"):
            return side + "Hand"
    return n


def surface(meshes, label):
    verts, normals, faces, weights, thumb_weights = [], [], [], [], []
    graph = bpy.context.evaluated_depsgraph_get()
    for obj in meshes:
        ev = obj.evaluated_get(graph); mesh = ev.to_mesh(); base = len(verts)
        normal_matrix = ev.matrix_world.to_3x3().inverted().transposed()
        names = {g.index: region(g.name) for g in obj.vertex_groups}
        if label.endswith('Knee'):
            names={i:label if n in [label[:-4]+'UpLeg',label[:-4]+'Leg'] else n for i,n in names.items()}
        raw_names = {g.index: g.name.split(':')[-1] for g in obj.vertex_groups}
        labels = []
        for v in mesh.vertices:
            ws = {}
            for g in v.groups:
                n = names.get(g.group)
                if n:
                    ws[n] = ws.get(n, 0) + g.weight
            labels.append(max(ws, key=ws.get) if ws else None)
            verts.append(ev.matrix_world @ v.co)
            normals.append((normal_matrix @ v.normal).normalized())
            weights.append(ws.get(label, 0))
            thumb_weights.append(sum(g.weight for g in v.groups if raw_names.get(g.group,'').startswith(label+'Thumb')))
        mesh.calc_loop_triangles()
        faces.extend(tuple(base + i for i in t.vertices) for t in mesh.loop_triangles
                     if sum(labels[i] == label for i in t.vertices) >= 2)
        ev.to_mesh_clear()
    used = sorted({i for f in faces for i in f})
    if not faces:
        raise ValueError("contact surface missing: " + label)
    return BVHTree.FromPolygons(verts, faces, all_triangles=True), np.array([verts[i] for i in used]), {
        "points": np.array(verts), "normals": np.array(normals), "weights": np.array(weights), "thumb_weights": np.array(thumb_weights)}


def set_hand_shape(arm, side, part):
    b = bones(arm)
    for n, p in b.items():
        if n.startswith(side + "Hand") and n != side + "Hand":
            p.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    pos = {n: arm.matrix_world @ p.head for n, p in b.items()}
    rot = {n: (arm.matrix_world @ p.matrix).to_3x3().normalized() for n, p in b.items()}
    bases = {f: np.array(pos[side + "Hand" + f.title() + "1"]) for f in ("thumb", "index", "middle", "ring", "pinky")}
    hf = hand_frame(side.lower(), np.array(pos[side + "Hand"]), bases)
    rests = {(f, i): FingerRest(np.array(pos[side + "Hand" + f.title() + str(i + 1)]),
                              np.array(rot[side + "Hand" + f.title() + str(i + 1)] @ Vector((0, 1, 0))),
                              np.array(rot[side + "Hand" + f.title() + str(i + 1)]))
             for f in bases for i in range(3)}
    for (f, i), r in finger_rotations(side.lower(), load_library().preset(part.hand_preset), hf, rests, bases).items():
        b[side + "Hand" + f.title() + str(i + 1)].matrix_basis = Matrix(r).to_4x4()
    bpy.context.view_layer.update()
    if part.planar_thumb:
        # Flat surface contact cannot use a thumb curled toward that surface.
        # Remove its out-of-palm-plane bend using the actual bone geometry.
        normal = Vector(hf.palm_normal)
        for i in range(1,4):
            p = b[side+'HandThumb'+str(i)]
            head = arm.matrix_world @ p.head; ray = arm.matrix_world @ p.tail-head
            tangent = unit(ray-normal*ray.dot(normal))
            delta = Matrix(shortest_arc(np.array(ray),np.array(tangent))).to_4x4()
            p.matrix = arm.matrix_world.inverted() @ Matrix.Translation(head) @ delta @ Matrix.Translation(-head) @ arm.matrix_world @ p.matrix
            bpy.context.view_layer.update()


def target_frame(arm, meshes, side, part):
    b = bones(arm); basis = body_frame(arm)
    left, up, forward = map(Vector, (-basis[:, 0], basis[:, 1], -basis[:, 2]))
    outward = left if side == "Left" else -left
    if part.target == "waist_side":
        tree, pts, _ = surface(meshes, "Torso")
        waist = arm.matrix_world @ ((b["Hips"].head + b["Spine1"].head) * .5)
        size = max(np.ptp(pts @ np.array(outward)), .1)
        hit = tree.ray_cast(waist + outward * size * 2, -outward, size * 4)
        if hit[0] is None:
            raise ValueError("waist side ray missed")
        anchor, normal = hit[:2]; normal = unit(normal)
        if normal.dot(outward) < 0: normal = -normal
        hu, hn = -up, -normal
        pole_direction = unit(outward - up * .5)
    elif part.target in ('abdomen_front','chest_front','collar_front'):
        tree,pts,_=surface(meshes,'Torso')
        landmark={'abdomen_front':'Spine','chest_front':'Spine2','collar_front':'Neck'}[part.target]
        center=arm.matrix_world@b[landmark].head
        if part.target=='collar_front':center=(center+arm.matrix_world@b['Spine2'].head)*.5
        depth=max(np.ptp(pts@np.array(forward)),.1)
        hit=tree.ray_cast(center+forward*depth*2,-forward,depth*4)
        if hit[0] is None:raise ValueError('front torso ray missed')
        anchor=hit[0];normal=forward
        hu=-outward if part.target=='abdomen_front' else up
        hn=-normal;pole_direction=unit(outward-up*.7+forward*.25)
    elif part.target=='same_knee':
        tree,pts,_=surface(meshes,side+'Knee')
        center=arm.matrix_world@b[side+'Leg'].head
        length=(arm.matrix_world@b[side+'Foot'].head-center).length
        hit=tree.ray_cast(center+up*length,-up,length*2)
        if hit[0] is None:raise ValueError('knee support ray missed')
        anchor=hit[0];normal=up;hu=-outward;hn=unit((-normal).cross(hu))
        pole_direction=unit(outward-up*.5+forward*.1)
    else:
        tree, pts, _ = surface(meshes, "Head")
        h = b["Head"]; center = arm.matrix_world @ h.head
        hup = unit(arm.matrix_world @ h.tail - center)
        hl = unit((arm.matrix_world @ h.matrix).to_3x3().col[0])
        if hl.dot(left) < 0: hl = -hl
        hf = unit(hl.cross(hup)); axes = np.array([hl, hup, hf])
        local = (pts - np.array(center)) @ axes.T
        low, high = local.min(axis=0), local.max(axis=0); span = high - low
        if part.target == "head_side":
            side_axis = hl if side == "Left" else -hl
            origin = center + hup * (low[1] + span[1] * .67) + hf * (low[2] + span[2] * .60)
            hit = tree.ray_cast(origin + side_axis * span[0] * 2, -side_axis, span[0] * 4)
            if hit[0] is None: raise ValueError("head side ray missed")
            anchor = hit[0]
            # An ear triangle's normal is not the anatomical side of the head.
            # Use the head frame for orientation; actual skin still sets the point.
            normal = side_axis
            hu, hn = unit(hup - normal * hup.dot(normal)), -normal
            pole_direction = unit(forward - up * .55 + outward * .6)
        elif part.target in ('neck_back','forehead_front'):
            back=part.target=='neck_back';normal=-hf if back else hf
            center_y=low[1]+span[1]*(.16 if back else .83)
            offset=hl*(1 if side=='Left' else -1)*span[0]*(.22 if back else .1)
            q=center+hup*center_y+offset+normal*span[2]*2
            hit=tree.ray_cast(q,-normal,span[2]*4)
            if hit[0] is None:raise ValueError('head front/back ray missed')
            anchor=hit[0];hu=hup if back else -(hl if side=='Left' else -hl);hn=-normal
            pole_direction=unit(outward+up*.35+forward*(-.3 if back else .25))
        else:
            profile = []
            for f in np.linspace(.04, .44, 41):
                lateral=hl*(1 if side=='Left' else -1)*span[0]*part.lateral_fraction
                q = center + lateral + hup * (low[1] + span[1] * f) + hf * span[2] * 2
                hit = tree.ray_cast(q, -hf, span[2] * 4)
                if hit[0] is not None:
                    profile.append((hit[0], (hit[0] - center).dot(hf)))
            threshold = .8 * max(z for _, z in profile)
            anchor = next(p for p, z in profile if z >= threshold)
            normal, hu, hn = -hup, hup, -hf
            pole_direction = unit(-up + forward * .2 + outward * .35)
    return tree, Vector(anchor), unit(normal), frame(hu, hn), pole_direction, (outward, up, forward)


def hand_samples(arm, meshes, side, part):
    b = bones(arm); w = arm.matrix_world @ b[side + "Hand"].head
    bases = {f: np.array(arm.matrix_world @ b[side + "Hand" + f.title() + "1"].head) for f in ("thumb", "index", "middle", "ring", "pinky")}
    hf = hand_frame(side.lower(), np.array(w), bases); h = frame(hf.length_axis, hf.palm_normal)
    _, _, raw = surface(meshes, side + "Hand")
    ids = np.flatnonzero(raw["weights"] >= .9)
    local = (raw["points"][ids] - np.array(w)) @ np.array(h)
    if part.patch == "palm":
        length = np.linalg.norm(bases["middle"] - np.array(w))
        width = np.linalg.norm(bases["index"] - bases["pinky"])
        normals = raw["normals"][ids] @ np.array(h)
        patch_ids = np.flatnonzero((local[:, 0] > length * .25) & (local[:, 0] < length * .85)
                                  & (abs(local[:, 2]) < width * .35) & (normals[:, 1] > .35))
    elif part.patch == 'thumb_pad':
        thumb = raw['thumb_weights'][ids] > .8
        patch_ids = np.flatnonzero(thumb & (local[:,1] > np.quantile(local[thumb,1],.6)))
    else:
        axis=contact_axis(part)
        patch_ids = np.flatnonzero(local[:, axis] > np.quantile(local[:, axis], .95))
        top = local[patch_ids]; center = np.median(top,axis=0)
        # The support patch is the central knuckle area, not the entire fist rim.
        across=[i for i in range(3) if i!=axis]
        rank = np.argsort(np.sum((top[:,across]-center[across])**2,axis=1))
        patch_ids = patch_ids[rank[:max(12,len(rank)//4)]]
    if len(patch_ids) < 8: raise ValueError("insufficient actual contact patch")
    patch = local[patch_ids]
    # Full actual skin is checked after fitting; rigid samples only guide the fit.
    sampled = local
    return h, sampled, patch


def elbow_for(s, w, lengths, preferred, swivel):
    v = w - s; d = v.length; a, c = lengths
    if not abs(a - c) + .001 < d < a + c - .001:
        raise ValueError("contact target unreachable with fixed arm lengths")
    axis = v / d; t = (a * a - c * c + d * d) / (2 * d)
    bend = unit(preferred - axis * preferred.dot(axis))
    bend = Quaternion(axis, swivel) @ bend
    return s + axis * t + bend * math.sqrt(max(0, a * a - t * t))


def elbow_rule_margin(part,e,w,s,lengths,axes):
    outward,up,forward=axes
    if part.elbow_rule=='outward_low':
        return min((e-w).dot(outward)-lengths[0]*.12,(s-e).dot(up)-lengths[0]*.12)
    if part.elbow_rule=='outward_raised':
        return min((e-w).dot(outward)-lengths[0]*.1,(e-s).dot(up)+lengths[0]*.1)
    if part.elbow_rule=='side_below_shoulder':
        return (s-e).dot(up)-lengths[0]*.1
    return min((w-e).dot(up)-lengths[1]*.25,(e-s).dot(forward)+lengths[0]*.05)


def fit_frame(objective, initial, bounds):
    """Bounded local pattern search; no native poses are rendered during fitting.

    Blender's bundled Python needs no external numerical runtime. Feasibility is
    still decided by native evidence afterwards, regardless of convergence.
    """
    x = np.array(initial, float); value = objective(x); count = 1
    step = np.array([.35, .15, .20, .15, .003])
    for level in range(10):
        for _ in range(8):
            changed = False
            for k in range(len(x)):
                options = [(value, x)]
                for sign in (-1, 1):
                    trial = x.copy(); trial[k] = np.clip(trial[k] + sign * step[k], *bounds[k])
                    options.append((objective(trial), trial)); count += 1
                best, candidate = min(options, key=lambda row: row[0])
                if best < value:
                    x, value, changed = candidate, best, True
            if not changed:
                # A contact plane often needs tilt and normal displacement to
                # change together. Axis-only steps can stick at the safety edge.
                options = [(value,x)]
                for k in (1,2,3):
                    for a in (-1,1):
                        for c in (-1,1):
                            trial=x.copy();trial[k]=np.clip(trial[k]+a*step[k],*bounds[k]);trial[4]=np.clip(trial[4]+c*step[4],*bounds[4])
                            options.append((objective(trial),trial));count+=1
                best,candidate=min(options,key=lambda row:row[0])
                if best<value:x,value,changed=candidate,best,True
            if not changed: break
        step *= .5
    return x, {"success": all(math.isfinite(v) for v in value), "evaluations": count,
               "message": "bounded local contact-frame fit; admission requires native checks", "parameters": x.tolist()}


def wrist_guard(arm, side, rig_profile):
    b = bones(arm); fore, hand = b[side + "ForeArm"], b[side + "Hand"]
    rest = Matrix(rig_profile['wrist_rest_relative'][side])
    q = (fore.matrix.to_3x3().normalized().transposed() @ hand.matrix.to_3x3().normalized() @ rest.transposed()).to_quaternion()
    v = Vector((0, q.y, 0)); twist = Quaternion((q.w, v.x, v.y, v.z))
    if twist.magnitude < 1e-8: return 180., 180.
    twist.normalize(); swing = q @ twist.conjugated()
    angle = lambda q: min(math.degrees(q.angle), 360 - math.degrees(q.angle))
    return angle(swing), angle(twist)


def apply_native(arm, side, e, w, desired_frame, original_hand_frame, rig_profile):
    b = bones(arm); upper, fore, hand = [b[side + t] for t in ("Arm", "ForeArm", "Hand")]
    hand_world = (arm.matrix_world @ hand.matrix).copy()
    local_frame = hand_world.to_3x3().normalized().transposed() @ original_hand_frame
    for p, child, target in [(upper, fore, e), (fore, hand, w)]:
        head = arm.matrix_world @ p.head; ray = arm.matrix_world @ child.head - head
        delta = Matrix(shortest_arc(np.array(ray), np.array(target - head))).to_4x4()
        p.matrix = arm.matrix_world.inverted() @ Matrix.Translation(head) @ delta @ Matrix.Translation(-head) @ arm.matrix_world @ p.matrix
        bpy.context.view_layer.update()
    hand_rotation = desired_frame @ local_frame.transposed()
    fore_world = (arm.matrix_world @ fore.matrix).copy(); axis = unit(w - e)
    options = []
    # The input rest frame can already contain arbitrary baked axial rotation.
    # Search the full circle relative to that frame; wrist swing/twist admission
    # is still measured against the canonical master rig (65/35 degrees).
    for roll in range(-180, 181, 2):
        fore.matrix = arm.matrix_world.inverted() @ Matrix.Translation(e) @ Matrix.Rotation(math.radians(roll), 4, axis) @ Matrix.Translation(-e) @ fore_world
        bpy.context.view_layer.update()
        hand.matrix = arm.matrix_world.inverted() @ Matrix.LocRotScale(w, hand_rotation.to_quaternion(), hand_world.to_scale())
        bpy.context.view_layer.update(); swing, twist = wrist_guard(arm, side, rig_profile)
        options.append((max(0, swing - 65) + max(0, twist - 35), twist + abs(roll) * .01, roll, fore.matrix.copy(), hand.matrix.copy()))
    best = min(options, key=lambda x: x[:2]); fore.matrix = best[3]
    bpy.context.view_layer.update(); hand.matrix = best[4]
    bpy.context.view_layer.update()
    return best[2]


def apply_contact_part(arm, meshes, part, side, skin_check, rig_profile, *, pole_bias=0.):
    """Return a receipt; on failure restore every original local bone matrix."""
    validate_rig(arm, meshes)
    if side not in ("left", "right"): raise ValueError("explicit side required")
    if arm.animation_data or any(p.constraints for p in arm.pose.bones): raise ValueError("bake rig before contact apply")
    scale = arm.matrix_world.to_scale()
    if min(scale) <= 0 or max(scale) - min(scale) > 1e-5: raise ValueError("unsupported armature scale")
    side = side.title(); b = bones(arm); before = snapshot(arm)
    originals = {p.name: p.matrix_basis.copy() for p in arm.pose.bones}
    initial_skin = skin_check(arm); initial_depth = {tuple(c["regions"]): c["max_depth_mm"] for c in initial_skin["collisions"]}
    s, e0, w0 = [arm.matrix_world @ b[side + t].head for t in ("Arm", "ForeArm", "Hand")]
    lengths = ((e0 - s).length, (w0 - e0).length)
    allowed = {p.name for n, p in b.items() if n in (side + "Arm", side + "ForeArm", side + "Hand") or n.startswith(side + "Hand")}
    receipt = {"part": part.record(), "side": side.lower(), "accepted": False, "release_allowed": False,
               "method": "common_destination_contact_fit_v1", "trials": [], "initial_skin": initial_skin}
    try:
        set_hand_shape(arm, side, part)
        shaped = {p.name: p.matrix_basis.copy() for p in arm.pose.bones}
        target, anchor, normal, desired, preferred, axes = target_frame(arm, meshes, side, part)
        preferred=unit(preferred+axes[0]*float(pole_bias))
        receipt["semantic_pole_bias"]=float(pole_bias)
        original_hand, samples, patch = hand_samples(arm, meshes, side, part)
        offset = patch.mean(axis=0)
        def decode(z):
            r = np.array(desired) @ np.array(Euler(tuple(z[1:4]), 'XYZ').to_matrix())
            w = anchor + normal * z[4] - Vector(r @ offset)
            e = elbow_for(s, w, lengths, preferred, z[0])
            return e, w, Matrix(r), samples @ r.T + np.array(w), patch @ r.T + np.array(w)
        def distances(pts, signed=False):
            values = []
            for p in pts:
                hit, n, _, d = target.find_nearest(Vector(p))
                values.append((Vector(p) - hit).dot(n) if signed else d)
            return np.array(values)
        def elbow_margin(e, w):
            return elbow_rule_margin(part,e,w,s,lengths,axes)
        def measure(z):
            e, w, h, pts, pp = decode(z)
            ds = distances(pp); signed = distances(pts, True)
            hu = h.col[0]; contact_normal = h.col[contact_axis(part)]
            return e, w, h, ds, signed, math.degrees(contact_normal.angle(-normal)), math.degrees(unit(w - e).angle(hu)), elbow_margin(e, w)
        def objective(z):
            try:
                e, w, h, ds, signed, norm, wrist, margin = measure(z)
                violation = max(0, -float(signed.min()) - .0005, (wrist-55)*.001, -margin, (norm-25)*.001)
                appearance = float(np.mean(ds ** 2) * 100 + norm ** 2 * .00001 + .0002 * np.sum(z[:4] ** 2))
                return violation, appearance
            except ValueError: return 1e3, 1e3
        bounds = [(-1.05, 1.05), (-.30, .30), (-.45, .45), (-.30, .30), (.001, .008)]
        fitted, fit_receipt = fit_frame(objective, [0, 0, 0, 0, .003], bounds)
        receipt["optimizer"] = fit_receipt
        e, w, h, *_ = decode(fitted)
        for p in arm.pose.bones: p.matrix_basis = shaped[p.name]
        bpy.context.view_layer.update()
        roll = apply_native(arm, side, e, w, h, original_hand, rig_profile)
        after = snapshot(arm); skin = skin_check(arm)
        # Resample the same central patch on the actually deformed hand surface.
        _, _, actual_patch = hand_samples(arm, meshes, side, part)
        aw = arm.matrix_world @ b[side + "Hand"].head
        hf = frame(arm.matrix_world @ b[side + "HandMiddle1"].head - aw,
                   (arm.matrix_world @ b[side + "HandIndex1"].head - aw).cross(arm.matrix_world @ b[side + "HandPinky1"].head - aw) * (-1 if side == "Left" else 1))
        world_patch = actual_patch @ np.array(hf).T + np.array(aw)
        ds = distances(world_patch); swing, twist = wrist_guard(arm, side, rig_profile)
        ae = arm.matrix_world @ b[side + "ForeArm"].head
        new = [c for c in skin["collisions"] if c["max_depth_mm"] > initial_depth.get(tuple(c["regions"]), 0) + 2]
        contact_normal = hf.col[contact_axis(part)]
        metrics = {"patch_median_m": float(np.median(ds)), "patch_p90_m": float(np.quantile(ds, .9)), "patch_samples": len(ds),
                   "hand_frame_error_deg": math.degrees((hf.transposed() @ h).to_quaternion().angle),
                   "normal_error_deg": math.degrees(contact_normal.angle(-normal)), "wrist_axis_deg": math.degrees(unit(aw - ae).angle(hf.col[0])),
                   "elbow_flexion_deg": math.degrees(unit(ae - s).angle(unit(aw - ae))),
                   "wrist_swing_deg": swing, "wrist_twist_deg": twist, "elbow_rule_margin_m": elbow_margin(ae, aw),
                   "bone_length_error_m": max(abs((ae - s).length - lengths[0]), abs((aw - ae).length - lengths[1])),
                   "wrist_goal_error_m": (aw - w).length,
                   "protected_matrix_delta": max(float(np.max(abs(after[n] - v))) for n, v in before.items() if n not in allowed),
                   "new_skin": new}
        receipt['fit_sample_min_signed_m'] = float(measure(fitted)[4].min())
        reasons = admission_reasons(metrics, part)
        receipt.update(metrics=metrics, reasons=reasons, accepted=not reasons, final_skin=skin,
                       anchor_world=list(anchor), normal_world=list(normal), wrist_goal_world=list(w), elbow_world=list(ae),
                       hand_frame_world=[list(v) for v in h], forearm_roll_deg=roll, patch_world=world_patch.tolist())
        # Diagnostic matrices may be saved by the caller, but cannot become the result.
        if reasons: receipt["rejected_pose_basis"] = {p.name: np.array(p.matrix_basis).tolist() for p in arm.pose.bones if p.name in allowed}
    except (ValueError, RuntimeError) as exc:
        receipt.update(reasons=[str(exc)])
    if not receipt["accepted"]:
        for p in arm.pose.bones: p.matrix_basis = originals[p.name]
        bpy.context.view_layer.update()
        receipt["rollback_matrix_delta"] = max(float(np.max(abs(snapshot(arm)[n] - m))) for n, m in before.items())
    return receipt
