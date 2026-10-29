"""Reviewed native leg recipes, with explicit language and measured anatomy.

Directions are transported in the destination pelvis frame; no image depth,
floor or anatomical side is guessed. A skin check remains mandatory at commit.
"""
import hashlib
import json
import re
import numpy as np
from .relations import parse_leg_language, relation_matches

ROLES = ("UpLeg", "Leg", "Foot", "ToeBase")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _bones(arm):
    return {p.name.split(":")[-1]: p for p in arm.pose.bones}


def _unit(v):
    v = np.asarray(v, float)
    if not np.isfinite(v).all() or np.linalg.norm(v) < 1e-8:
        raise ValueError("degenerate leg direction")
    return v / np.linalg.norm(v)


def measure_legs(arm):
    b = _bones(arm)
    points = {s: [np.array(arm.matrix_world @ b[s.title()+r].head) for r in ROLES]
              for s in ("left", "right")}
    pelvis = np.array((arm.matrix_world @ b["Hips"].matrix).to_3x3().normalized())
    up = _unit(pelvis[:,1])
    spine = np.array(arm.matrix_world @ b["Spine2"].head-arm.matrix_world @ b["Hips"].head)
    hip_delta = points["right"][0] - points["left"][0]
    if spine @ up <= 0 or abs(hip_delta @ pelvis[:,0]) < .02:
        raise ValueError("uncalibrated pelvis axes")
    right = _unit(pelvis[:,0]) * np.sign(hip_delta @ pelvis[:,0])
    knees = {s: float(np.degrees(np.arccos(np.clip(
        _unit(p[1]-p[0]) @ _unit(p[2]-p[1]), -1, 1)))) for s, p in points.items()}
    delta = points["right"][1] - points["left"][1]
    # A knee-over-knee stack reverses hip-to-knee lateral order. It need not
    # have an X-shaped centerline crossing in the reference camera.
    lateral = float(delta @ right)
    height = float(delta @ up)
    top = ("right" if height > .003 else "left" if height < -.003 else None) if lateral < -.003 else None
    return {"knee_flex_deg": knees, "knee_order_lateral_m": lateral,
            "right_knee_above_left_m": height, "top_side": top,
            "basis": "anatomical_hip_axis_reversal_and_body_up_knee_order",
            "shin_body_up": {s: float(_unit(p[2]-p[1]) @ up) for s, p in points.items()},
            "scope": "knee_over_knee_joint_relation_not_skin_or_photo_score"}


def plan_legs(sentence, *, selection, source, reference):
    if source not in ("user_confirmed", "reviewed_library") or not reference:
        raise ValueError("attributed leg intent required")
    if selection not in ("left", "right", "both"):
        raise ValueError("explicit anatomical leg selection required")
    intent = parse_leg_language(sentence)
    # Extend the shared parser conservatively for the accepted thigh-over-thigh
    # reference. Other crossing families continue to require their own checker.
    compact = "".join(sentence.split())
    thigh_stack = re.search(r"(왼|오른)(?:쪽)?허벅지를(왼|오른)(?:쪽)?허벅지위", compact)
    if thigh_stack and thigh_stack[1] == thigh_stack[2]:
        raise ValueError("same thigh cannot be above itself")
    if thigh_stack:
        side = "right" if thigh_stack[1] == "오른" else "left"
        intent.update(crossing=True, top_side=side, bottom_side="left" if side == "right" else "right")
    if intent["crossing"] and "허벅지" in compact and "굽" in compact:
        intent.update(cross_region="knee", cross_style="seated_knee_over_knee")
    if intent["crossing"] and (selection != "both" or intent["top_side"] is None):
        raise ValueError("crossing requires both legs and explicit anatomical top side")
    if intent["crossing"] and intent["cross_style"] != "seated_knee_over_knee":
        raise ValueError("this adapter requires measured knee-over-knee; use a camera-depth checker for ankle/shin crosses")
    return {"schema": "standin.refine-v4.leg-plan.v1", "sentence": sentence,
            "selection": selection, "sides": ["left", "right"] if selection == "both" else [selection],
            "intent": intent, "evidence": {"source": source, "reference": reference},
            "automatic_image_extraction": False}


def extract_native_leg_part(arm, plan, *, source_sha256):
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("exact source asset hash required")
    b = _bones(arm)
    pelvis = np.array((arm.matrix_world @ b["Hips"].matrix).to_3x3().normalized())
    rays = {}
    for s in plan["sides"]:
        p = [np.array(arm.matrix_world @ b[s.title()+r].head) for r in ROLES]
        rays[s] = [(pelvis.T @ _unit(v-u)).tolist() for u, v in zip(p, p[1:])]
    measurements = measure_legs(arm)
    if plan["intent"].get("crossing") and measurements["top_side"] != plan["intent"]["top_side"]:
        raise ValueError("authoring source does not match requested anatomical order")
    part = {"schema": "standin.refine-v4.native-leg-part.v1", "source_sha256": source_sha256,
            "sides": plan["sides"], "directions_pelvis": rays, "intent": plan["intent"],
            "evidence": plan["evidence"], "measurements": measurements,
            "skin_transfer": "not_included_pose_specific_skin_asset_is_separate"}
    part["id"] = _digest(part)
    return part


def apply_native_leg_part(arm, meshes, part, plan, *, skin_check, locked_feet=()):
    import bpy
    from mathutils import Matrix
    from src.experimental.part_composition import shortest_arc
    from src.scene_refinement.rig import validate_rig
    from .pipeline import snapshot, skin_nonregression
    validate_rig(arm, meshes)
    if part.get("id") != _digest({k:v for k,v in part.items() if k != "id"}):
        raise ValueError("leg part integrity mismatch")
    if part["sides"] != plan["sides"] or not relation_matches(plan["intent"], part["intent"]):
        raise ValueError("selected leg part does not match anatomical intent")
    if not callable(skin_check):
        raise ValueError("actual skin checker required")
    if not set(locked_feet) <= {"left", "right"}:
        raise ValueError("invalid locked foot")
    b = _bones(arm)
    if arm.animation_data or any(p.constraints for p in arm.pose.bones):
        raise ValueError("bake rig before leg refinement")
    scales = arm.matrix_world.to_scale()
    if min(scales) <= 0 or max(scales)-min(scales) > 1e-5:
        raise ValueError("unsupported armature scale")
    bases = {p.name:p.matrix_basis.copy() for p in arm.pose.bones}
    before = snapshot(arm)
    feet = {s: np.array(arm.matrix_world @ b[s.title()+"Foot"].head) for s in locked_feet}
    initial_skin = skin_check(arm)
    pelvis = np.array((arm.matrix_world @ b["Hips"].matrix).to_3x3().normalized())
    receipt = {"accepted": False, "reasons": [], "selected_parts": [part["id"]],
               "plan": plan, "method": "native_pelvis_direction_transport", "skin_asset_transferred": False}
    try:
        for s in plan["sides"]:
            for k, (parent, child) in enumerate(zip(ROLES, ROLES[1:])):
                pb, cb = b[s.title()+parent], b[s.title()+child]
                if cb.parent != pb:
                    raise ValueError("unsupported native leg chain")
                a, c = arm.matrix_world @ pb.head, arm.matrix_world @ cb.head
                direction = _unit(pelvis @ np.asarray(part["directions_pelvis"][s][k]))
                rotation = Matrix(shortest_arc(np.array(c-a), direction).tolist()).to_4x4()
                world = arm.matrix_world @ pb.matrix
                pb.matrix = arm.matrix_world.inverted() @ Matrix.Translation(a) @ rotation @ Matrix.Translation(-a) @ world
                bpy.context.view_layer.update()
                actual = _unit(np.array(arm.matrix_world @ cb.head-arm.matrix_world @ pb.head))
                if np.linalg.norm(actual-direction) > 1e-4:
                    raise ValueError("leg direction transport failed")
        measurements = measure_legs(arm)
        receipt["measurements"] = measurements
        if max(measurements["knee_flex_deg"].values()) > 155:
            raise ValueError("knee_flex_over_155")
        intent = plan["intent"]
        if intent.get("crossing"):
            if measurements["top_side"] != intent["top_side"]:
                raise ValueError("crossed_leg_order_does_not_match_reference")
            top = intent["top_side"]
            if measurements["knee_flex_deg"][top] < 45 or max(measurements["shin_body_up"].values()) > -.3:
                raise ValueError("upper_knee_must_fold_and_calves_hang_down")
        for s, p in feet.items():
            if np.linalg.norm(np.array(arm.matrix_world @ b[s.title()+"Foot"].head)-p) > 1e-5:
                raise ValueError("support_foot_would_slide")
        check = skin_nonregression(initial_skin, skin_check(arm))
        receipt["skin_check"] = check
        if not check["accepted"]:
            raise ValueError("leg_skin_rejected")
        receipt["accepted"] = True
    except Exception as exc:
        receipt["reasons"].append(str(exc))
        for p in arm.pose.bones:
            p.matrix_basis = bases[p.name]
        bpy.context.view_layer.update()
        receipt["rollback_matrix_delta"] = max(float(np.max(abs(snapshot(arm)[n]-v))) for n,v in before.items())
    return receipt
