"""Reusable arm pose parts: clavicle anchor -> upper arm -> forearm -> wrist.

Meshes and fingers are not transplanted. Source bind offsets/channel order are
recorded for auditing; cross-rig application transports directions, never Euler
angles. Shoulder/clavicle anchors stay fixed in the destination.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.bvh import channel_starts, coco17_from_fk, fk, parse_bvh
from src.collision import arm_torso_penetration, arm_leg_penetration
from src.experimental.part_composition import (
    _mapping, _set_rotation, _validate, shortest_arc, torso_frame, world_transforms,
)

VERSION = "standin.arm-pose-part.v1"
SIDES = ("left", "right")
CHAIN = ("Shoulder", "Arm", "ForeArm", "Hand")


def selected_sides(selection):
    if selection not in ("left", "right", "both"):
        raise ValueError("selection must be subject-relative left/right/both")
    return SIDES if selection == "both" else (selection,)


def checked_chain(joints, side):
    names = _mapping(joints)
    keys = [side.title() + suffix for suffix in CHAIN]
    if any(key not in names for key in keys):
        raise ValueError("unsupported rig: expected explicit clavicle/arm/forearm/hand chain")
    indices = [names[key] for key in keys]
    if any(joints[b][1] != a for a, b in zip(indices, indices[1:])):
        raise ValueError("unsupported arm parent chain")
    return keys, indices


def extract_part(source, selection, *, label=None, relation="independent", frame_index=0):
    """Extract from a real BVH. Source labels are provenance, not detector output."""
    sides = selected_sides(selection)
    if relation not in ("independent", "paired") or (relation == "paired" and selection != "both"):
        raise ValueError("paired relation requires both arms")
    source = Path(source)
    joints, frames = parse_bvh(str(source))
    if not isinstance(frame_index, int) or not 0 <= frame_index < len(frames):
        raise ValueError("invalid source frame")
    frame = _validate(joints, frames[frame_index])
    world = world_transforms(joints, frame)
    bind = world_transforms(joints, np.zeros_like(frame))
    basis = torso_frame(joints, frame)
    positions, scores = coco17_from_fk(joints, fk(joints, frame))
    # coco17 has hip-centered coordinates; use actual shoulder midpoint for transforms.
    names = _mapping(joints)
    origin = (world[names['LeftArm']][:3, 3] + world[names['RightArm']][:3, 3]) * .5
    scale = float(np.linalg.norm((positions[5]+positions[6]-positions[11]-positions[12])*.5))
    if scale < 1e-7:
        raise ValueError("degenerate torso")
    starts = channel_starts(joints)
    arms = {}
    for side in sides:
        keys, indices = checked_chain(joints, side)
        root, upper, fore, hand = indices
        points = np.array([world[i][:3, 3] for i in (upper, fore, hand)])
        rays = np.diff(points, axis=0)
        lengths = np.linalg.norm(rays, axis=1)
        if min(lengths) < 1e-7:
            raise ValueError("zero-length arm")
        directions = rays / lengths[:, None]
        flex = float(np.degrees(np.arccos(np.clip(np.dot(*directions), -1, 1))))
        records = []
        for key, i in zip(keys, indices):
            records.append({"name": key, "source_name": joints[i][0],
                "parent": joints[joints[i][1]][0].split(':')[-1],
                "offset": joints[i][2].tolist(), "channels": joints[i][3],
                "values": frame[starts[i]: starts[i]+len(joints[i][3])].tolist(),
                "bind_world": bind[i].tolist(), "pose_world": world[i].tolist()})
        arms[side] = {"chain": records,
            "anchor_position_torso": (basis.T @ (world[root][:3, 3]-origin)/scale).tolist(),
            "shoulder_position_torso": (basis.T @ (points[0]-origin)/scale).tolist(),
            "elbow_position_torso": (basis.T @ (points[1]-origin)/scale).tolist(),
            "wrist_position_torso": (basis.T @ (points[2]-origin)/scale).tolist(),
            "directions_torso": (directions @ basis).tolist(),
            "lengths_torso": (lengths/scale).tolist(), "elbow_flexion_deg": flex,
            "extension": "straight" if flex < 25 else "bent",
            "wrist_rotation_torso": (basis.T @ world[hand][:3, :3]).tolist()}
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {"schema": VERSION, "id": f"{source.stem}--{selection}",
        "label": label or source.stem, "selection": selection, "side_convention": "subject_relative",
        "relation": relation, "source": {"pose_id": source.stem, "sha256": digest,
            "frame_index": frame_index}, "coordinate_frame": "anatomical_torso_right_up_cross",
        "source_torso_scale": scale, "source_torso_frame": basis.tolist(),
        "source_shoulder_center_world": origin.tolist(), "arms": arms,
        "apply_policy": {"anchors": "fixed_clavicle_and_shoulder_position",
            "mutable": [s.title()+n for s in sides for n in ("Arm", "ForeArm", "Hand")],
            "wrist": "preserve_destination_world_orientation", "fingers": "preserve_destination_local_pose",
            "length": "preserve_destination", "requires_review": True},
        "release_allowed": False}


def validate_part(part):
    if part.get("schema") != VERSION or part.get("side_convention") != "subject_relative":
        raise ValueError("unsupported arm part")
    sides = selected_sides(part["selection"])
    if set(part["arms"]) != set(sides):
        raise ValueError("part side mismatch")
    if part["relation"] not in ("independent", "paired") or (part["relation"] == "paired" and len(sides) != 2):
        raise ValueError("invalid part relation")
    for side in sides:
        arm = part["arms"][side]
        directions = np.asarray(arm["directions_torso"], float)
        lengths = np.asarray(arm["lengths_torso"], float)
        if directions.shape != (2,3) or not np.isfinite(directions).all() or not np.allclose(np.linalg.norm(directions,axis=1),1,atol=1e-6):
            raise ValueError("invalid arm directions")
        if lengths.shape != (2,) or not np.isfinite(lengths).all() or min(lengths) <= 0:
            raise ValueError("invalid arm lengths")
        if [r['name'] for r in arm['chain']] != [side.title()+s for s in CHAIN]:
            raise ValueError("invalid part chain")
    return sides


def require_pair_quality(part):
    """Held folded-arm sources cannot silently enter the apply path."""
    semantics = part.get('semantics', {})
    if semantics.get('action') == 'fold_arms':
        quality = semantics.get('pair_quality') or {}
        if not quality.get('eligible_strict_fold', False):
            raise ValueError('paired folded-arm contact quality is on hold')


def write_part(part, path):
    validate_part(part)
    Path(path).write_text(json.dumps(part, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def read_part(path):
    part = json.loads(Path(path).read_text())
    validate_part(part)
    return part


def standalone_bvh(part):
    """An actual one-frame BVH with arms only, in torso units; no hand geometry.

    Synthetic common root plus source clavicle/upper/forearm/wrist chain. Each
    clavicle's parent rotation is baked into its first rotation. Not a full-body
    replacement BVH: the JSON application mask is authoritative.
    """
    from scipy.spatial.transform import Rotation
    sides = validate_part(part)
    scale = part['source_torso_scale']
    basis = np.asarray(part['source_torso_frame'])
    lines = ['HIERARCHY', 'ROOT ArmPartAnchor', '{', '  OFFSET 0 0 0',
             '  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation']
    values = [0.]*6
    for side in sides:
        arm = part['arms'][side]
        for depth, row in enumerate(arm['chain']):
            indent = '  '*(depth+1)
            if depth == 0:
                offset = arm['anchor_position_torso']
                matrix = basis.T @ np.asarray(row['pose_world'])[:3,:3]
            else:
                offset = np.asarray(row['offset'])/scale
                # Frame values are applied in the source channel order.
                matrix = np.eye(3)
                from src.bvh import _rot
                for channel, val in zip(row['channels'],row['values']):
                    if channel.endswith('position'):
                        offset['XYZ'.index(channel[0])] += val/scale
                    elif channel.endswith('rotation'):
                        matrix = matrix @ _rot(channel[0], val)
            lines += [indent+'JOINT '+row['name'], indent+'{',
                      indent+'  OFFSET '+' '.join(f'{v:.12g}' for v in offset),
                      indent+'  CHANNELS 3 Zrotation Yrotation Xrotation']
            values += Rotation.from_matrix(matrix).as_euler('ZYX',degrees=True).tolist()
        lines += ['          End Site', '          {', '            OFFSET 0 0 0', '          }']
        for depth in reversed(range(4)):
            lines.append('  '*(depth+1)+'}')
    lines += ['}', 'MOTION', 'Frames: 1', 'Frame Time: 0.033333',
              ' '.join(f'{v:.12g}' for v in values)]
    return '\n'.join(lines)+'\n'


def apply_part(joints, frame, part, *, selection=None):
    """Direction transport; exact preservation of unselected local channels.

    Clavicles are fixed. Wrist orientation follows destination, fingers retain
    local pose. Failed geometry gates return the untouched input (atomic pair).
    """
    available = validate_part(part)
    require_pair_quality(part)
    sides = available if selection is None else selected_sides(selection)
    if not set(sides) <= set(available):
        raise ValueError('selection absent from part')
    if part['relation'] == 'paired' and tuple(sides) != SIDES:
        raise ValueError('paired part must be applied atomically to both arms')
    base = _validate(joints, frame)
    basis = torso_frame(joints, base)
    world0 = world_transforms(joints, base)
    out = base.copy()
    changed = set()
    for side in sides:
        keys, ids = checked_chain(joints, side)
        _, upper, fore, hand = ids
        wrist_rotation = world0[hand][:3,:3].copy()
        for k, (i, child) in enumerate(((upper,fore),(fore,hand))):
            world = world_transforms(joints,out)
            current = world[child][:3,3]-world[i][:3,3]
            desired = basis @ np.asarray(part['arms'][side]['directions_torso'][k])
            rotation = shortest_arc(current,desired) @ world[i][:3,:3]
            parent_rotation = world[joints[i][1]][:3,:3]
            _set_rotation(joints,out,i,parent_rotation.T @ rotation)
            changed.add(i)
        world = world_transforms(joints,out)
        _set_rotation(joints,out,hand,world[fore][:3,:3].T @ wrist_rotation)
        changed.add(hand)
    after = world_transforms(joints,out)
    starts = channel_starts(joints)
    mutable = {starts[i]+k for i in changed for k,c in enumerate(joints[i][3]) if c.endswith('rotation')}
    protected_channels = [i for i in range(len(base)) if i not in mutable]
    # Hand descendants move in world space, as intended. Their local shape stays exact.
    moving = set(changed)
    for i,j in enumerate(joints):
        if j[1] in moving:
            moving.add(i)
    protected = [i for i in range(len(joints)) if i not in moving]
    delta = max(np.linalg.norm(after[i][:3,3]-world0[i][:3,3]) for i in protected)
    bp, scores = coco17_from_fk(joints,fk(joints,base))
    ap, _ = coco17_from_fk(joints,fk(joints,out))
    collisions, reasons = {}, []
    for side in sides:
        limb = side+'_arm'
        before = arm_torso_penetration(bp,limb,scores)
        current = arm_torso_penetration(ap,limb,scores)
        collisions[limb] = {'before':before.depth,'after':current.depth}
        if not (before.available and current.available):
            reasons.append('collision_proxy_unavailable')
        elif current.depth > before.depth+.02:
            reasons.append('new_torso_proxy_penetration:'+side)
        for leg in ('left_leg','right_leg'):
            prev = arm_leg_penetration(bp,limb,leg,scores)
            cur = arm_leg_penetration(ap,limb,leg,scores)
            if prev.available and cur.available and cur.depth > prev.depth+.02:
                reasons.append('new_arm_leg_proxy_penetration:'+side)
        if part['arms'][side]['elbow_flexion_deg'] > 165:
            reasons.append('elbow_limit:'+side)
    if delta > 1e-7 or not np.array_equal(base[protected_channels],out[protected_channels]):
        reasons.append('protected_pose_changed')
    diagnostics = {'accepted':not reasons,'reasons':reasons,'selection':list(sides),
        'part_id':part['id'],'protected_position_max_delta':float(delta),
        'protected_channels_exact':bool(np.array_equal(base[protected_channels],out[protected_channels])),
        'collision_proxy':collisions,'release_allowed':False,
        'limitations':['proxy_not_mesh_collision','shoulder_twist_limits_not_certified','no_rough_ground_truth']}
    return (out if not reasons else base.copy()), diagnostics


def search_parts(parts, target, selection, *, top_k=5):
    """Manual torso-frame direction query; never an image-detector accuracy claim."""
    sides = selected_sides(selection)
    hits = []
    for part in parts:
        available = validate_part(part)
        if not set(sides) <= set(available) or (part['relation']=='paired' and selection!='both'):
            continue
        errors = []
        for side in sides:
            q = np.array(target[side],dtype=float,copy=True)
            if q.shape!=(2,3) or not np.isfinite(q).all() or min(np.linalg.norm(q,axis=1))<1e-7:
                raise ValueError('target needs two finite nonzero torso directions per selected arm')
            q /= np.linalg.norm(q,axis=1)[:,None]
            p = np.asarray(part['arms'][side]['directions_torso'])
            errors.extend(np.degrees(np.arccos(np.clip(np.sum(q*p,axis=1),-1,1))).tolist())
        hits.append({'part_id':part['id'],'mean_direction_error_deg':float(np.mean(errors)),
                     'selection':selection,'requires_review':True})
    return sorted(hits,key=lambda h:(h['mean_direction_error_deg'],h['part_id']))[:top_k]
