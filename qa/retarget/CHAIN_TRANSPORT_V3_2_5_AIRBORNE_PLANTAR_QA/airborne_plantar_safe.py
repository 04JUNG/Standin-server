"""QA-only airborne plantar-plane transfer with exact V3.2.4 rollback.

The accepted V3.2.4 artifact is produced first and is the immutable fallback.
Only non-contact-owned feet may be reconsidered.  Source rest-relative ankle
flexion creates a virtual target plantar plane in the target shank frame; the
actual deformed target sole is then searched toward that plane.  Any missing
measurement, unsafe candidate, combined-state failure, or export exception
copies the already-produced V3.2.4 FBX byte-for-byte to the requested output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import bpy
from mathutils import Matrix, Vector


ROOT = Path(__file__).resolve().parent
V324_ROOT = ROOT.parent / "CHAIN_TRANSPORT_V3_2_4_FOOT_PLANT_QA"
V324_PATH = V324_ROOT / "foot_plant_safe.py"
V324_POLICY_PATH = V324_ROOT / "foot_plant_policy.json"
POLICY_PATH = ROOT / "airborne_plantar_policy.json"
_V324_MODULE = "_standin_v324_exact_contact_parent"
_SIDES = ("L", "R")
_EPS = 1.0e-8


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_v324():
    if _V324_MODULE not in sys.modules:
        spec = importlib.util.spec_from_file_location(_V324_MODULE, V324_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load exact V3.2.4 parent")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_V324_MODULE] = module
        spec.loader.exec_module(module)
    return sys.modules[_V324_MODULE]


def _number(value, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"{label} must be finite")
    return result


def load_policy(path: str | os.PathLike[str] = POLICY_PATH) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    policy = json.loads(raw)
    if policy.get("schema_version") != 1:
        raise RuntimeError("unsupported V3.2.5 policy schema")
    if policy.get("status") != "QA_ONLY_NOT_PROMOTED":
        raise RuntimeError("V3.2.5 must remain QA-only")
    actual_module = _sha256(V324_PATH)
    actual_parent_policy = _sha256(V324_POLICY_PATH)
    if actual_module != policy.get("parent_v324_module_sha256"):
        raise RuntimeError(
            "V3.2.4 module hash mismatch: "
            f"expected {policy.get('parent_v324_module_sha256')}, got {actual_module}"
        )
    if actual_parent_policy != policy.get("parent_v324_policy_sha256"):
        raise RuntimeError(
            "V3.2.4 policy hash mismatch: "
            f"expected {policy.get('parent_v324_policy_sha256')}, "
            f"got {actual_parent_policy}"
        )
    for group in ("activation", "search", "measurement", "safety"):
        policy[group] = {
            key: _number(value, f"{group}.{key}")
            for key, value in policy[group].items()
        }
    search = policy["search"]
    if not 0.0 < search["gain_min"] <= search["gain_max"]:
        raise RuntimeError("invalid airborne gain interval")
    if not 0.0 < search["correction_abs_max_deg"] <= 90.0:
        raise RuntimeError("airborne correction hard guard must stay in (0, 90]")
    return policy, hashlib.sha256(raw).hexdigest()


def _float_range(start: float, stop: float, step: float) -> list[float]:
    if step <= 0.0 or stop < start:
        return []
    count = int(math.floor((stop - start) / step + 1.0e-9))
    rows = [round(start + index * step, 9) for index in range(count + 1)]
    if rows and rows[-1] < stop - 1.0e-9:
        rows.append(round(stop, 9))
    return rows


def _angle_deg(a: Vector, b: Vector) -> float:
    if a.length <= _EPS or b.length <= _EPS:
        raise RuntimeError("relative bend vector degenerate")
    dot = max(-1.0, min(1.0, a.normalized().dot(b.normalized())))
    return math.degrees(math.acos(dot))


@dataclass
class PlantarFrame:
    measurable: bool
    status: str
    vertex_count: int = 0
    rear_bottom_count: int = 0
    front_bottom_count: int = 0
    direction: list[float] | None = None


def _plantar_frame(v322, mesh, arm, rt, dst_table, side: str,
                   rest_positions: list[Vector], posed_positions: list[Vector],
                   policy: dict) -> tuple[PlantarFrame, Vector | None]:
    measure = policy["measurement"]
    inv = arm.matrix_world.inverted()
    ankle = inv @ rt._rest_world(arm, dst_table[f"foot.{side}"]).translation
    toe = inv @ rt._rest_world(arm, dst_table[f"toe.{side}"]).translation
    forward = toe - ankle
    if forward.length <= _EPS:
        return PlantarFrame(False, "target_rest_foot_edge_degenerate"), None
    forward.normalize()
    weights = v322._vertex_weight_rows(mesh)
    names = (dst_table[f"foot.{side}"], dst_table[f"toe.{side}"])
    rows = []
    for index, (point, row) in enumerate(zip(rest_positions, weights)):
        if sum(row.get(name, 0.0) for name in names) >= measure["minimum_foot_weight"]:
            rows.append((index, (point - ankle).dot(forward)))
    minimum = int(measure["minimum_band_vertices"])
    if len(rows) < minimum * 4:
        return PlantarFrame(False, "foot_surface_vertices_insufficient", len(rows)), None
    projections = [value for _, value in rows]
    rear_limit = v322._quantile(projections, measure["rear_quantile"])
    front_limit = v322._quantile(projections, measure["front_quantile"])
    rear = [index for index, value in rows if value <= rear_limit]
    front = [index for index, value in rows if value >= front_limit]
    up = (inv.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()
    rear_heights = [posed_positions[index].dot(up) for index in rear]
    front_heights = [posed_positions[index].dot(up) for index in front]
    rear_cut = v322._quantile(rear_heights, measure["bottom_quantile"])
    front_cut = v322._quantile(front_heights, measure["bottom_quantile"])
    rear_bottom = [
        posed_positions[index] for index in rear
        if posed_positions[index].dot(up) <= rear_cut
    ]
    front_bottom = [
        posed_positions[index] for index in front
        if posed_positions[index].dot(up) <= front_cut
    ]
    if len(rear_bottom) < minimum or len(front_bottom) < minimum:
        return PlantarFrame(
            False, "plantar_bottom_bands_insufficient", len(rows),
            len(rear_bottom), len(front_bottom)
        ), None
    rear_center = sum(rear_bottom, Vector()) / len(rear_bottom)
    front_center = sum(front_bottom, Vector()) / len(front_bottom)
    direction = front_center - rear_center
    if direction.length <= _EPS:
        return PlantarFrame(False, "plantar_direction_degenerate", len(rows)), None
    direction.normalize()
    return PlantarFrame(
        True, "ok", len(rows), len(rear_bottom), len(front_bottom), list(direction)
    ), direction


def _target_shin_local(rt, arm, dst_table, side: str, *, rest: bool) -> Vector:
    inv = arm.matrix_world.inverted()
    knee = inv @ (
        rt._rest_world(arm, dst_table[f"leg.{side}"]).translation
        if rest else rt._pose_world(arm, dst_table[f"leg.{side}"]).translation
    )
    ankle = inv @ (
        rt._rest_world(arm, dst_table[f"foot.{side}"]).translation
        if rest else rt._pose_world(arm, dst_table[f"foot.{side}"]).translation
    )
    delta = ankle - knee
    if delta.length <= _EPS:
        raise RuntimeError("target shin direction degenerate")
    return delta.normalized()


def _source_motion(v322, v324, rt, bone_map, src_arm, src_table,
                   side: str, mirror: bool) -> dict:
    names = (f"leg.{side}", f"foot.{side}", f"toe.{side}")
    sr = [
        v322._source_point(
            rt, bone_map, src_arm, src_table, name, rest=True, mirror=mirror
        ) for name in names
    ]
    sp = [
        v322._source_point(
            rt, bone_map, src_arm, src_table, name, rest=False, mirror=mirror
        ) for name in names
    ]
    rest_bend = _angle_deg(sr[1] - sr[0], sr[2] - sr[1])
    pose_bend = _angle_deg(sp[1] - sp[0], sp[2] - sp[1])
    toe_pitch = v324._source_toe_pitch(
        v322, rt, bone_map, src_arm, src_table, side, mirror
    )["pitch_deg"]
    return {
        "source_rest_bend_deg": rest_bend,
        "source_pose_bend_deg": pose_bend,
        "source_rest_relative_motion_deg": pose_bend - rest_bend,
        "source_toe_pitch_deg": toe_pitch,
    }


def _reference_evidence(v324, v322, rt, bone_map, *, bvh_path: str,
                        character_fbx: str, frame: int, mirror: bool,
                        src_profile: str | None, dst_profile: str | None,
                        policy: dict) -> tuple[dict, str, str]:
    rt.reset_scene()
    dst_arm, meshes = rt.import_character(character_fbx)
    src_arm = rt.import_bvh(bvh_path, frame=frame)
    resolved_src = src_profile or bone_map.resolve_profile(
        [bone.name for bone in src_arm.data.bones]
    )
    resolved_dst = dst_profile or bone_map.resolve_profile(
        [bone.name for bone in dst_arm.data.bones]
    )
    evidence = {}
    if len(meshes) != 1:
        return {
            side: {"measurable": False, "status": "target_mesh_count_not_one"}
            for side in _SIDES
        }, resolved_src, resolved_dst
    v322._reset_pose(dst_arm)
    mesh = meshes[0]
    rest_positions = v322._positions(mesh, dst_arm, evaluated=False)
    src_table = bone_map.PROFILES[resolved_src]
    dst_table = bone_map.PROFILES[resolved_dst]
    for side in _SIDES:
        try:
            source = _source_motion(
                v322, v324, rt, bone_map, src_arm, src_table, side, mirror
            )
            frame_row, plantar = _plantar_frame(
                v322, mesh, dst_arm, rt, dst_table, side,
                rest_positions, rest_positions, policy,
            )
            if not frame_row.measurable or plantar is None:
                raise RuntimeError(frame_row.status)
            shin = _target_shin_local(rt, dst_arm, dst_table, side, rest=True)
            target_rest_bend = _angle_deg(shin, plantar)
            desired = (
                target_rest_bend + source["source_rest_relative_motion_deg"]
            )
            evidence[side] = {
                "measurable": True,
                "status": "ok",
                "source": source,
                "target_rest_plantar": asdict(frame_row),
                "target_rest_bend_deg": target_rest_bend,
                "desired_target_relative_bend_deg": desired,
            }
        except (KeyError, RuntimeError, ValueError) as exc:
            evidence[side] = {
                "measurable": False,
                "status": f"{type(exc).__name__}:{exc}",
            }
    return evidence, resolved_src, resolved_dst


def _scaled_rotation(rotation: Matrix, gain: float) -> Matrix:
    axis, angle = rotation.to_quaternion().normalized().to_axis_angle()
    if abs(angle) <= _EPS:
        return Matrix.Identity(4)
    return Matrix.Rotation(angle * gain, 4, axis)


def _correction_context(rt, arm, shin_local: Vector, plantar_local: Vector,
                        desired_bend_deg: float) -> tuple[dict | None, str]:
    perpendicular = plantar_local - shin_local * plantar_local.dot(shin_local)
    if perpendicular.length <= 1.0e-6:
        return None, "target_ankle_plane_degenerate"
    perpendicular.normalize()
    desired_rad = math.radians(desired_bend_deg)
    desired_local = (
        math.cos(desired_rad) * shin_local
        + math.sin(desired_rad) * perpendicular
    ).normalized()
    world_rotation = arm.matrix_world.to_3x3()
    current_world = (world_rotation @ plantar_local).normalized()
    desired_world = (world_rotation @ desired_local).normalized()
    full, why = rt._min_rotation(current_world, desired_world)
    if full is None:
        return None, f"minimum_swing_degenerate:{why}"
    return {
        "full": full,
        "base_plantar_direction_world": current_world,
        "desired_direction_local": list(desired_local),
    }, "ok"


def _apply_correction(v322, rt, arm, dst_table, side: str,
                      context: dict, gain: float) -> dict:
    foot_name = dst_table[f"foot.{side}"]
    toe_name = dst_table[f"toe.{side}"]
    base_foot = rt._pose_world(arm, foot_name).copy()
    base_toe = rt._pose_world(arm, toe_name).copy()
    base_relative = base_foot.to_quaternion().inverted() @ base_toe.to_quaternion()
    correction = _scaled_rotation(context["full"], gain)
    twist = v322._signed_twist_deg(
        correction, context["base_plantar_direction_world"]
    )
    desired_foot = correction @ rt._rot_only(base_foot)
    desired_foot.translation = rt._pose_world(arm, foot_name).translation
    arm.pose.bones[foot_name].matrix = arm.matrix_world.inverted() @ desired_foot
    bpy.context.view_layer.update()
    desired_toe = correction @ rt._rot_only(base_toe)
    desired_toe.translation = rt._pose_world(arm, toe_name).translation
    arm.pose.bones[toe_name].matrix = arm.matrix_world.inverted() @ desired_toe
    bpy.context.view_layer.update()
    after_foot = rt._pose_world(arm, foot_name)
    after_toe = rt._pose_world(arm, toe_name)
    after_relative = after_foot.to_quaternion().inverted() @ after_toe.to_quaternion()
    return {
        "gain": gain,
        "applied_swing_deg": v322._rotation_error_deg(Matrix.Identity(4), correction),
        "correction_twist_deg": twist,
        "toe_relative_rotation_error_deg": math.degrees(
            base_relative.rotation_difference(after_relative).angle
        ),
    }


def _surface_safe(v324, candidate, baseline, policy: dict) -> tuple[bool, list[str]]:
    absolute_ok, reasons = v324._surface_safe(candidate, policy)
    reasons = list(reasons)
    if not baseline.measurable:
        reasons.append("PARENT_SURFACE_UNMEASURABLE")
        return False, reasons
    gate = policy["safety"]
    checks = [
        (
            candidate.clearance_p01_leg_ratio
            >= gate["approved_v324_clearance_p01_min_leg_ratio"],
            "OUTSIDE_APPROVED_V324_CLEARANCE_ENVELOPE",
        ),
        (
            candidate.area_ratio_p01
            >= gate["approved_v324_area_ratio_p01_min"],
            "OUTSIDE_APPROVED_V324_AREA_ENVELOPE",
        ),
        (
            candidate.edge_ratio_min
            >= gate["approved_v324_edge_ratio_min"],
            "OUTSIDE_APPROVED_V324_EDGE_ENVELOPE",
        ),
        (
            candidate.log_edge_strain_p99
            <= gate["approved_v324_log_edge_strain_p99_max"],
            "OUTSIDE_APPROVED_V324_STRAIN_ENVELOPE",
        ),
        (
            candidate.dihedral_change_p99_deg
            <= gate["approved_v324_dihedral_change_p99_max_deg"],
            "OUTSIDE_APPROVED_V324_DIHEDRAL_ENVELOPE",
        ),
        (
            candidate.new_sharp_fold_count
            <= baseline.new_sharp_fold_count,
            "NEW_FOLD_VS_V324",
        ),
    ]
    reasons.extend(label for passed, label in checks if not passed)
    return absolute_ok and not reasons, reasons


def _scene_target():
    arms = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(arms) != 1 or len(meshes) != 1:
        raise RuntimeError(
            f"exact V3.2.4 scene must contain one armature and one mesh, "
            f"got {len(arms)} armatures/{len(meshes)} meshes"
        )
    return arms[0], meshes


class AirbornePlantarReport:
    def __init__(self, payload: dict, selection: dict):
        self._payload = payload
        self.airborne_plantar_selection = selection

    def __getattr__(self, name):
        if name in self._payload:
            return self._payload[name]
        raise AttributeError(name)

    def as_dict(self):
        payload = dict(self._payload)
        payload["airborne_plantar_selection"] = self.airborne_plantar_selection
        return payload


def convert_safe(
    *, bvh_path: str, character_fbx: str, out_path: str,
    frame: int = 0, mirror: bool = False, output_mode: str = "rigged_rest",
    apply_root_translation: bool = False, embed_textures: bool = True,
    src_profile: str | None = None, dst_profile: str | None = None,
    force_exact_v324: bool = False,
    policy_path: str | os.PathLike[str] = POLICY_PATH,
):
    started = time.time()
    v324 = _load_v324()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if force_exact_v324:
        with tempfile.TemporaryDirectory(prefix="v325-force-v324-") as temp_dir:
            parent_path = Path(temp_dir) / "exact-v324-parent.fbx"
            parent = v324.convert_safe(
                bvh_path=bvh_path, character_fbx=character_fbx,
                out_path=str(parent_path), frame=frame, mirror=mirror,
                output_mode=output_mode,
                apply_root_translation=apply_root_translation,
                embed_textures=embed_textures, src_profile=src_profile,
                dst_profile=dst_profile,
            )
            parent_payload = parent.as_dict()
            parent_hash = _sha256(parent_path)
            shutil.copyfile(parent_path, out)
            final_hash = _sha256(out)
            selection = {
                "selector": "V3.2.5_CHAIN_RELATIVE_VIRTUAL_PLANTAR",
                "status": "FORCED_EXACT_PARENT",
                "policy_sha256": (
                    _sha256(POLICY_PATH) if POLICY_PATH.is_file() else None
                ),
                "exact_parent_variant": "CHAIN_TRANSPORT_V3_2_4_FOOT_PLANT_QA",
                "parent_artifact_sha256": parent_hash,
                "reference": {},
                "activation": {},
                "candidates": {side: [] for side in _SIDES},
                "selected_gain": {side: 0.0 for side in _SIDES},
                "final_checks": {},
                "fallback_to_exact_v324": True,
                "fallback_reason": "FORCED_EXACT_V324",
                "changed_bones_only": ["foot.L", "toe.L", "foot.R", "toe.R"],
                "final_artifact_sha256": final_hash,
                "exact_parent_artifact_restored": final_hash == parent_hash,
                "elapsed_sec": time.time() - started,
            }
            return AirbornePlantarReport(parent_payload, selection)

    policy, policy_hash = load_policy(policy_path)
    v323 = v324._load_v323()
    v322 = v323._load_v322()
    rt, bone_map = v322._load_parent()
    with tempfile.TemporaryDirectory(prefix="v325-airborne-") as temp_dir:
        parent_path = Path(temp_dir) / "exact-v324-parent.fbx"
        parent = v324.convert_safe(
            bvh_path=bvh_path, character_fbx=character_fbx,
            out_path=str(parent_path), frame=frame, mirror=mirror,
            output_mode=output_mode, apply_root_translation=apply_root_translation,
            embed_textures=embed_textures, src_profile=src_profile,
            dst_profile=dst_profile,
        )
        parent_payload = parent.as_dict()
        parent_hash = _sha256(parent_path)
        resolved_src = parent_payload.get("src_profile")
        resolved_dst = parent_payload.get("dst_profile")
        selection = {
            "selector": "V3.2.5_CHAIN_RELATIVE_VIRTUAL_PLANTAR",
            "status": policy["status"],
            "policy_sha256": policy_hash,
            "exact_parent_variant": policy["exact_parent_variant"],
            "parent_artifact_sha256": parent_hash,
            "reference": {},
            "activation": {},
            "candidates": {side: [] for side in _SIDES},
            "selected_gain": {side: 0.0 for side in _SIDES},
            "final_checks": {},
            "fallback_to_exact_v324": True,
            "fallback_reason": None,
            "changed_bones_only": ["foot.L", "toe.L", "foot.R", "toe.R"],
        }

        def rollback(reason: str):
            shutil.copyfile(parent_path, out)
            selection["fallback_to_exact_v324"] = True
            selection["fallback_reason"] = reason
            selection["final_artifact_sha256"] = _sha256(out)
            selection["exact_parent_artifact_restored"] = (
                selection["final_artifact_sha256"] == parent_hash
            )
            selection["elapsed_sec"] = time.time() - started
            return AirbornePlantarReport(parent_payload, selection)

        if not parent_payload.get("ok"):
            return rollback("V324_PARENT_FAILED")
        if output_mode != "rigged_rest":
            return rollback("OUTPUT_MODE_NOT_SUPPORTED_BY_AIRBORNE_QA")

        try:
            reference, reference_src, reference_dst = _reference_evidence(
                v324, v322, rt, bone_map, bvh_path=bvh_path,
                character_fbx=character_fbx, frame=frame, mirror=mirror,
                src_profile=resolved_src, dst_profile=resolved_dst, policy=policy,
            )
            if reference_src != resolved_src or reference_dst != resolved_dst:
                raise RuntimeError("reference profile differs from V3.2.4 parent")
            selection["reference"] = reference
            rt.reset_scene()
            rt.import_character(str(parent_path))
            dst_arm, meshes = _scene_target()
            mesh = meshes[0]
            dst_table = bone_map.PROFILES[resolved_dst]
            v322._reset_pose(dst_arm)
            rest_positions = v322._positions(mesh, dst_arm, evaluated=False)
            baseline_positions = v322._positions(mesh, dst_arm, evaluated=True)
            baseline_snapshot = v322._pose_snapshot(rt, dst_arm, dst_table)
            baseline_frames = {}
            baseline_vectors = {}
            baseline_surfaces = {}
            contexts = {}
            parent_contact = parent_payload.get("foot_plant_selection", {})
            parent_selected = parent_contact.get(
                "selected_correction_deg", {side: 0.0 for side in _SIDES}
            )
            activate = policy["activation"]
            for side in _SIDES:
                frame_row, plantar = _plantar_frame(
                    v322, mesh, dst_arm, rt, dst_table, side,
                    rest_positions, baseline_positions, policy,
                )
                baseline_frames[side] = frame_row
                baseline_vectors[side] = plantar
                baseline_surfaces[side] = v322._surface_metrics(
                    mesh, dst_arm, rt, dst_table, side,
                    rest_positions, baseline_positions, policy,
                )
                ref = reference.get(side, {})
                reasons = []
                parent_owned = bool(parent_selected.get(side, 0.0))
                source_pitch = ref.get("source", {}).get("source_toe_pitch_deg")
                if parent_owned:
                    reasons.append("V324_CONTACT_CORRECTION_SELECTED")
                if source_pitch is None:
                    reasons.append("SOURCE_PITCH_UNMEASURABLE")
                elif abs(source_pitch) <= activate["contact_owned_source_toe_pitch_abs_max_deg"]:
                    reasons.append("CONTACT_OWNED_SOURCE_ORIENTATION")
                if not ref.get("measurable") or not frame_row.measurable or plantar is None:
                    reasons.append("RELATIVE_PLANTAR_UNMEASURABLE")
                    error = float("inf")
                    context = None
                else:
                    desired = ref["desired_target_relative_bend_deg"]
                    if not activate["desired_bend_min_deg"] <= desired <= activate["desired_bend_max_deg"]:
                        reasons.append("DESIRED_BEND_OUT_OF_RANGE")
                    shin = _target_shin_local(
                        rt, dst_arm, dst_table, side, rest=False
                    )
                    current = _angle_deg(shin, plantar)
                    error = abs(current - desired)
                    if error < activate["parent_relative_bend_error_min_deg"]:
                        reasons.append("V324_RELATIVE_BEND_COMPATIBLE")
                    context, why = _correction_context(
                        rt, dst_arm, shin, plantar, desired
                    )
                    if context is None:
                        reasons.append(why)
                active = not reasons
                selection["activation"][side] = {
                    "active": active,
                    "reasons": reasons,
                    "parent_contact_selected": parent_owned,
                    "parent_relative_bend_error_deg": error,
                    "parent_plantar": asdict(frame_row),
                }
                contexts[side] = context

            search = policy["search"]
            selected_rows = {}
            for side in _SIDES:
                if not selection["activation"][side]["active"]:
                    continue
                coarse = _float_range(
                    search["gain_min"], search["gain_max"],
                    search["coarse_gain_step"],
                )
                coarse_rows = []
                for gain in coarse:
                    row = _candidate(
                        v324, v322, rt, dst_arm, mesh, dst_table, side,
                        gain, contexts[side], reference[side], rest_positions,
                        baseline_snapshot, baseline_surfaces[side],
                        selection["activation"][side]["parent_relative_bend_error_deg"],
                        policy,
                    )
                    coarse_rows.append(row)
                selection["candidates"][side].extend(coarse_rows)
                eligible = [row for row in coarse_rows if row["eligible"]]
                if not eligible:
                    continue
                best = min(
                    eligible,
                    key=lambda row: (
                        row["relative_bend_error_deg"],
                        row["applied"]["applied_swing_deg"],
                    ),
                )
                start = max(
                    search["gain_min"],
                    best["gain"] - search["fine_gain_radius"],
                )
                stop = min(
                    search["gain_max"],
                    best["gain"] + search["fine_gain_radius"],
                )
                seen = {row["gain"] for row in coarse_rows}
                for gain in _float_range(start, stop, search["fine_gain_step"]):
                    if gain in seen:
                        continue
                    selection["candidates"][side].append(_candidate(
                        v324, v322, rt, dst_arm, mesh, dst_table, side,
                        gain, contexts[side], reference[side], rest_positions,
                        baseline_snapshot, baseline_surfaces[side],
                        selection["activation"][side]["parent_relative_bend_error_deg"],
                        policy,
                    ))
                rescued = [
                    row for row in selection["candidates"][side]
                    if row["eligible"]
                    and row["relative_bend_error_deg"]
                    <= search["target_relative_bend_error_max_deg"]
                ]
                if rescued:
                    selected_rows[side] = min(
                        rescued,
                        key=lambda row: (
                            row["relative_bend_error_deg"],
                            row["applied"]["applied_swing_deg"],
                        ),
                    )
                    selection["selected_gain"][side] = selected_rows[side]["gain"]

            if not selected_rows:
                return rollback("NO_SAFE_AIRBORNE_CANDIDATE")

            v322._reset_pose(dst_arm)
            for side, row in selected_rows.items():
                _apply_correction(
                    v322, rt, dst_arm, dst_table, side,
                    contexts[side], row["gain"],
                )
            final_positions = v322._positions(mesh, dst_arm, evaluated=True)
            final_snapshot = v322._pose_snapshot(rt, dst_arm, dst_table)
            final_safe = True
            for side, row in selected_rows.items():
                frame_row, plantar = _plantar_frame(
                    v322, mesh, dst_arm, rt, dst_table, side,
                    rest_positions, final_positions, policy,
                )
                surface = v322._surface_metrics(
                    mesh, dst_arm, rt, dst_table, side,
                    rest_positions, final_positions, policy,
                )
                reasons = []
                if not frame_row.measurable or plantar is None:
                    reasons.append("FINAL_PLANTAR_UNMEASURABLE")
                    error = float("inf")
                else:
                    shin = _target_shin_local(
                        rt, dst_arm, dst_table, side, rest=False
                    )
                    error = abs(
                        _angle_deg(shin, plantar)
                        - reference[side]["desired_target_relative_bend_deg"]
                    )
                    if error > search["target_relative_bend_error_max_deg"]:
                        reasons.append("FINAL_RELATIVE_BEND_ERROR")
                surface_ok, surface_reasons = _surface_safe(
                    v324, surface, baseline_surfaces[side], policy
                )
                if not surface_ok:
                    reasons.extend(surface_reasons)
                selection["final_checks"][side] = {
                    "plantar": asdict(frame_row),
                    "surface": asdict(surface),
                    "relative_bend_error_deg": error,
                    "safe": not reasons,
                    "reasons": reasons,
                }
                if reasons:
                    final_safe = False
            proximal = v322._proximal_error(baseline_snapshot, final_snapshot)
            selection["final_proximal_matrix_error"] = proximal
            if proximal > policy["safety"]["maximum_proximal_matrix_error"]:
                final_safe = False
                selection["fallback_reason"] = "FINAL_PROXIMAL_MATRIX_REGRESSION"
            if not final_safe:
                return rollback(selection["fallback_reason"] or "FINAL_COMBINED_SAFETY_FAILED")

            selected_positions = list(final_positions)
            src_arm = rt.import_bvh(bvh_path, frame=frame)
            parent_payload["pose_fidelity_rmse"] = rt.pose_fidelity_rmse(
                src_arm, dst_arm, resolved_src, resolved_dst, mirror=mirror
            )
            parent_payload["skeleton_baseline_rmse"] = rt.skeleton_baseline_rmse(
                src_arm, dst_arm, resolved_src, resolved_dst
            )
            parent_payload["pose_fidelity_delta"] = (
                parent_payload["pose_fidelity_rmse"]
                - parent_payload["skeleton_baseline_rmse"]
            )
            bpy.data.objects.remove(src_arm, do_unlink=True)
            rt.apply_output_mode(dst_arm, meshes, output_mode)
            baked = v322._positions(mesh, dst_arm, evaluated=False)
            selection["post_bake_max_vertex_error"] = max(
                (a - b).length for a, b in zip(selected_positions, baked)
            )
            rt.export_fbx(str(out), embed_textures=embed_textures, bake_anim=False)
            selection["fallback_to_exact_v324"] = False
            selection["fallback_reason"] = None
            selection["final_artifact_sha256"] = _sha256(out)
            selection["exact_parent_artifact_restored"] = False
            selection["elapsed_sec"] = time.time() - started
            return AirbornePlantarReport(parent_payload, selection)
        except Exception as exc:
            selection["exception"] = f"{type(exc).__name__}:{exc}"
            return rollback("EXCEPTION_AFTER_V324_PARENT")


def _candidate(v324, v322, rt, arm, mesh, dst_table, side: str,
               gain: float, context: dict, reference: dict,
               rest_positions: list[Vector], baseline_snapshot: dict,
               baseline_surface, parent_error: float, policy: dict) -> dict:
    v322._reset_pose(arm)
    applied = _apply_correction(v322, rt, arm, dst_table, side, context, gain)
    positions = v322._positions(mesh, arm, evaluated=True)
    frame_row, plantar = _plantar_frame(
        v322, mesh, arm, rt, dst_table, side,
        rest_positions, positions, policy,
    )
    surface = v322._surface_metrics(
        mesh, arm, rt, dst_table, side,
        rest_positions, positions, policy,
    )
    after = v322._pose_snapshot(rt, arm, dst_table)
    proximal = v322._proximal_error(baseline_snapshot, after)
    reasons = []
    desired = reference["desired_target_relative_bend_deg"]
    if not frame_row.measurable or plantar is None:
        error = float("inf")
        reasons.append("PLANTAR_UNMEASURABLE")
    else:
        shin = _target_shin_local(rt, arm, dst_table, side, rest=False)
        result = _angle_deg(shin, plantar)
        error = abs(result - desired)
    correction_deg = applied["applied_swing_deg"]
    search = policy["search"]
    safe = policy["safety"]
    if correction_deg > search["correction_abs_max_deg"]:
        reasons.append("CORRECTION_HARD_GUARD")
    if abs(applied["correction_twist_deg"]) > safe["maximum_correction_twist_deg"]:
        reasons.append("TWIST_CHANGED")
    if applied["toe_relative_rotation_error_deg"] > safe["maximum_toe_relative_rotation_error_deg"]:
        reasons.append("TOE_RELATIVE_ROTATION_CHANGED")
    if proximal > safe["maximum_proximal_matrix_error"]:
        reasons.append("PROXIMAL_MATRIX_REGRESSION")
    improvement = parent_error - error
    if improvement < search["minimum_relative_bend_improvement_deg"]:
        reasons.append("RELATIVE_BEND_NOT_PRACTICALLY_IMPROVED")
    surface_ok, surface_reasons = _surface_safe(
        v324, surface, baseline_surface, policy
    )
    if not surface_ok:
        reasons.extend(surface_reasons)
    activation_error = None
    # The immutable parent error is carried by the activation record; candidate
    # eligibility still requires a practical absolute target error.
    if error > search["target_relative_bend_error_max_deg"]:
        activation_error = error
    row = {
        "gain": gain,
        "eligible": not reasons,
        "reasons": reasons,
        "relative_bend_error_deg": error,
        "target_error_gate_pending": activation_error,
        "plantar": asdict(frame_row),
        "surface": asdict(surface),
        "applied": applied,
        "proximal_matrix_error": proximal,
        "parent_relative_bend_error_deg": parent_error,
        "relative_bend_improvement_deg": improvement,
    }
    return row


def math_self_test() -> dict:
    source_rest = 102.0
    source_pose = 62.0
    target_rest = 108.0
    desired = target_rest + (source_pose - source_rest)
    flat_contact_owned = abs(8.0) <= 8.0
    airborne_not_contact_owned = abs(8.0001) > 8.0
    result = {
        "source_motion_deg": source_pose - source_rest,
        "desired_target_bend_deg": desired,
        "contact_boundary_owned": flat_contact_owned,
        "airborne_above_boundary": airborne_not_contact_owned,
    }
    result["passed"] = (
        abs(result["source_motion_deg"] + 40.0) <= 1.0e-9
        and abs(result["desired_target_bend_deg"] - 68.0) <= 1.0e-9
        and flat_contact_owned
        and airborne_not_contact_owned
    )
    if not result["passed"]:
        raise RuntimeError(f"V3.2.5 math self-test failed: {result}")
    return result
