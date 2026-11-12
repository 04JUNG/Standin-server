"""Promoted runtime source-toe / target-plantar pitch transfer.

The exact V3.2.1 pose is always the fallback.  A correction is considered
only when the source toe segment is near horizontal and the actual target
foot surface disagrees.  Candidate choice is made from the deformed mesh,
not from a filename, character label, or limb-length formula.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

import bpy
from mathutils import Matrix, Vector


ROOT = Path(__file__).resolve().parent
V323_PATH = ROOT / "ankle_clearance.py"
POLICY_PATH = ROOT / "foot_plant_policy.json"
_SIDES = ("L", "R")
_EPS = 1.0e-8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_v323():
    from . import ankle_clearance
    return ankle_clearance


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
        raise RuntimeError("unsupported V3.2.4 policy schema")
    if policy.get("status") != "PROMOTED_RUNTIME":
        raise RuntimeError("V3.2.4 policy is not promoted")
    actual = _sha256(V323_PATH)
    expected = policy.get("parent_v323_module_sha256")
    if actual != expected:
        raise RuntimeError(
            f"V3.2.3 predecessor hash mismatch: expected {expected}, got {actual}"
        )
    for group in ("activation", "search", "measurement", "safety"):
        policy[group] = {
            key: _number(value, f"{group}.{key}")
            for key, value in policy[group].items()
        }
    search = policy["search"]
    if not 0.0 < search["correction_abs_max_deg"] <= 90.0:
        raise RuntimeError("contact correction hard guard must stay in (0, 90]")
    if search["coarse_step_deg"] <= 0.0 or search["fine_step_deg"] <= 0.0:
        raise RuntimeError("contact search steps must be positive")
    return policy, hashlib.sha256(raw).hexdigest()


def _float_range(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step))
    return [round(start + index * step, 9) for index in range(count + 1)]


def _signed_pitch_deg(vector: Vector, up: Vector) -> float:
    if vector.length <= _EPS or up.length <= _EPS:
        raise RuntimeError("pitch vector degenerate")
    unit_up = up.normalized()
    vertical = vector.dot(unit_up)
    horizontal = (vector - unit_up * vertical).length
    if horizontal <= _EPS:
        raise RuntimeError("pitch horizontal component degenerate")
    return math.degrees(math.atan2(vertical, horizontal))


def _source_toe_pitch(v322, rt, bone_map, src_arm, src_table,
                      side: str, mirror: bool) -> dict:
    source_side = bone_map.mirror_name(f"toe.{side}") if mirror else f"toe.{side}"
    name = src_table[source_side]
    matrix = rt._pose_world(src_arm, name)
    head = matrix.translation.copy()
    tail = matrix @ Vector((0.0, src_arm.data.bones[name].length, 0.0))
    if mirror:
        head.x = -head.x
        tail.x = -tail.x
    delta = tail - head
    pitch = _signed_pitch_deg(delta, Vector((0.0, 0.0, 1.0)))
    return {
        "measurable": True,
        "status": "ok",
        "source_bone": name,
        "pitch_deg": pitch,
        "head": list(head),
        "tail": list(tail),
        "segment_length": delta.length,
    }


@dataclass
class PlantarMetrics:
    measurable: bool
    status: str
    vertex_count: int = 0
    rear_vertex_count: int = 0
    front_vertex_count: int = 0
    rear_bottom_count: int = 0
    front_bottom_count: int = 0
    rear_projection_limit: float | None = None
    front_projection_limit: float | None = None
    rear_height_q05: float | None = None
    front_height_q05: float | None = None
    pitch_deg: float | None = None


def _plantar_metrics(v322, mesh, arm, rt, dst_table, side: str,
                     rest_positions: list[Vector], posed_positions: list[Vector],
                     policy: dict) -> PlantarMetrics:
    measure = policy["measurement"]
    inv = arm.matrix_world.inverted()
    up = (inv.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()
    ankle = inv @ rt._rest_world(arm, dst_table[f"foot.{side}"]).translation
    toe = inv @ rt._rest_world(arm, dst_table[f"toe.{side}"]).translation
    forward = toe - ankle
    if forward.length <= _EPS:
        return PlantarMetrics(False, "target_rest_foot_edge_degenerate")
    forward.normalize()
    names = (dst_table[f"foot.{side}"], dst_table[f"toe.{side}"])
    weights = v322._vertex_weight_rows(mesh)
    rows = []
    for index, (point, row) in enumerate(zip(rest_positions, weights)):
        if sum(row.get(name, 0.0) for name in names) >= measure["minimum_foot_weight"]:
            rows.append((index, (point - ankle).dot(forward)))
    minimum = int(measure["minimum_band_vertices"])
    if len(rows) < minimum * 4:
        return PlantarMetrics(False, "foot_surface_vertices_insufficient", len(rows))
    projections = [value for _, value in rows]
    rear_limit = v322._quantile(projections, measure["rear_quantile"])
    front_limit = v322._quantile(projections, measure["front_quantile"])
    rear = [index for index, value in rows if value <= rear_limit]
    front = [index for index, value in rows if value >= front_limit]
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
        return PlantarMetrics(
            False, "plantar_bottom_bands_insufficient", len(rows), len(rear),
            len(front), len(rear_bottom), len(front_bottom)
        )
    rear_center = sum(rear_bottom, Vector()) / len(rear_bottom)
    front_center = sum(front_bottom, Vector()) / len(front_bottom)
    try:
        pitch = _signed_pitch_deg(front_center - rear_center, up)
    except RuntimeError as exc:
        return PlantarMetrics(False, str(exc))
    return PlantarMetrics(
        True, "ok", len(rows), len(rear), len(front), len(rear_bottom),
        len(front_bottom), rear_limit, front_limit,
        v322._quantile(rear_heights, 0.05),
        v322._quantile(front_heights, 0.05), pitch,
    )


def _apply_pitch(v322, rt, dst_arm, dst_table, side: str,
                 correction_deg: float) -> dict:
    foot = v322._target_point(rt, dst_arm, dst_table, f"foot.{side}", rest=False)
    toe = v322._target_point(rt, dst_arm, dst_table, f"toe.{side}", rest=False)
    forward = toe - foot
    if forward.length <= _EPS:
        raise RuntimeError("target pose foot edge degenerate")
    forward.normalize()
    lateral = forward.cross(Vector((0.0, 0.0, 1.0)))
    if lateral.length <= _EPS:
        raise RuntimeError("contact swing axis degenerate")
    lateral.normalize()
    correction = Matrix.Rotation(math.radians(correction_deg), 4, lateral)
    result = v322._apply_side_swing(
        rt, dst_arm, dst_table, side,
        {"full": correction, "base_foot_direction": forward}, 1.0,
    )
    result["correction_deg"] = correction_deg
    result["axis"] = list(lateral)
    return result


def _surface_safe(surface, policy: dict) -> tuple[bool, list[str]]:
    if not surface.measurable:
        return False, ["SURFACE_UNMEASURABLE"]
    gate = policy["safety"]
    checks = [
        (surface.clearance_p01_leg_ratio >= gate["clearance_p01_min_leg_ratio"],
         "CLEARANCE_TOO_LOW"),
        (surface.area_ratio_p01 >= gate["area_ratio_p01_min"], "AREA_COMPRESSION"),
        (surface.edge_ratio_min >= gate["edge_ratio_min"], "EDGE_COMPRESSION"),
        (surface.log_edge_strain_p99 <= gate["log_edge_strain_p99_max"],
         "EDGE_STRAIN"),
        (surface.dihedral_change_p99_deg <= gate["dihedral_change_p99_max_deg"],
         "DIHEDRAL_CHANGE"),
        (surface.new_sharp_fold_count <= gate["new_sharp_fold_count_max"],
         "NEW_SHARP_FOLD"),
    ]
    reasons = [reason for passed, reason in checks if not passed]
    return not reasons, reasons


def _candidate(rt, v322, mesh, rest_positions, src_arm, dst_arm, src_profile,
               dst_profile, dst_table, side, correction_deg, palm_mu,
               output_mode, frame, mirror, apply_root_translation, policy,
               source_pitch, parent_pitch_error) -> tuple[dict, object]:
    report = v322._run_parent(
        rt, src_arm, dst_arm, src_profile=src_profile, dst_profile=dst_profile,
        mirror=mirror, apply_root_translation=apply_root_translation,
        palm_mu=palm_mu, output_mode=output_mode, frame=frame,
    )
    row = {"correction_deg": correction_deg, "parent_ok": report.ok}
    if not report.ok:
        row.update({"eligible": False, "reasons": ["PARENT_FAILED"]})
        return row, report
    before = v322._pose_snapshot(rt, dst_arm, dst_table)
    try:
        applied = _apply_pitch(v322, rt, dst_arm, dst_table, side, correction_deg)
    except RuntimeError as exc:
        row.update({"eligible": False, "reasons": [str(exc)]})
        return row, report
    after = v322._pose_snapshot(rt, dst_arm, dst_table)
    posed = v322._positions(mesh, dst_arm, evaluated=True)
    plantar = _plantar_metrics(
        v322, mesh, dst_arm, rt, dst_table, side, rest_positions, posed, policy
    )
    surface = v322._surface_metrics(
        mesh, dst_arm, rt, dst_table, side, rest_positions, posed, policy
    )
    reasons = []
    if not plantar.measurable:
        reasons.append("PLANTAR_UNMEASURABLE")
        pitch_error = float("inf")
        improvement = float("-inf")
    else:
        pitch_error = abs(plantar.pitch_deg - source_pitch)
        improvement = parent_pitch_error - pitch_error
        if improvement < policy["search"]["minimum_pitch_improvement_deg"]:
            reasons.append("PITCH_NOT_PRACTICALLY_IMPROVED")
    surface_ok, surface_reasons = _surface_safe(surface, policy)
    if not surface_ok:
        reasons.extend(surface_reasons)
    safe = policy["safety"]
    if abs(applied["correction_twist_deg"]) > safe["maximum_correction_twist_deg"]:
        reasons.append("TWIST_CHANGED")
    if applied["toe_relative_rotation_error_deg"] > safe["maximum_toe_relative_rotation_error_deg"]:
        reasons.append("TOE_RELATIVE_ROTATION_CHANGED")
    proximal = v322._proximal_error(before, after)
    if proximal > safe["maximum_proximal_matrix_error"]:
        reasons.append("PROXIMAL_MATRIX_REGRESSION")
    row.update({
        "applied": applied,
        "plantar": asdict(plantar),
        "surface": asdict(surface),
        "pitch_error_deg": pitch_error,
        "pitch_improvement_deg": improvement,
        "proximal_matrix_error": proximal,
        "eligible": not reasons,
        "reasons": reasons,
    })
    return row, report


class FootPlantReport:
    def __init__(self, parent, selection):
        self.parent = parent
        self.foot_plant_selection = selection

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def as_dict(self):
        payload = self.parent.as_dict()
        payload["foot_plant_selection"] = self.foot_plant_selection
        return payload


def convert_safe(
    *, bvh_path: str, character_fbx: str, out_path: str,
    frame: int = 0, mirror: bool = False, output_mode: str = "rigged_rest",
    apply_root_translation: bool = False, embed_textures: bool = True,
    src_profile: str | None = None, dst_profile: str | None = None,
    policy_path: str | os.PathLike[str] = POLICY_PATH,
):
    started = time.time()
    policy, policy_hash = load_policy(policy_path)
    v323 = _load_v323()
    v323.load_policy()
    v322 = v323._load_v322()
    v322.load_policy()
    rt, bone_map = v322._load_parent()
    if output_mode not in {"rigged_rest", "static_mesh"}:
        raise ValueError("V3.2.4 supports rigged_rest/static_mesh only")
    rt.reset_scene()
    dst_arm, meshes = rt.import_character(character_fbx)
    src_arm = rt.import_bvh(bvh_path, frame=frame)
    resolved_src = src_profile or bone_map.resolve_profile(
        [bone.name for bone in src_arm.data.bones]
    )
    resolved_dst = dst_profile or bone_map.resolve_profile(
        [bone.name for bone in dst_arm.data.bones]
    )
    palm_mu = policy["parent_palm_roll_mu"]
    report = v322._run_parent(
        rt, src_arm, dst_arm, src_profile=resolved_src, dst_profile=resolved_dst,
        mirror=mirror, apply_root_translation=apply_root_translation,
        palm_mu=palm_mu, output_mode=output_mode, frame=frame,
    )
    selected = {side: 0.0 for side in _SIDES}
    requested_selected = {side: 0.0 for side in _SIDES}
    evidence = {}
    candidates = {side: [] for side in _SIDES}
    final_checks = {}
    final_applications = {}
    mesh = meshes[0] if len(meshes) == 1 else None
    rest_positions = v322._positions(mesh, dst_arm, evaluated=False) if mesh else None
    if report.ok and mesh is not None:
        src_table = bone_map.PROFILES[report.src_profile]
        dst_table = bone_map.PROFILES[report.dst_profile]
        parent_positions = v322._positions(mesh, dst_arm, evaluated=True)
        for side in _SIDES:
            try:
                source = _source_toe_pitch(
                    v322, rt, bone_map, src_arm, src_table, side, mirror
                )
                parent_plantar = _plantar_metrics(
                    v322, mesh, dst_arm, rt, dst_table, side,
                    rest_positions, parent_positions, policy,
                )
                parent_error = (
                    abs(parent_plantar.pitch_deg - source["pitch_deg"])
                    if parent_plantar.measurable else float("inf")
                )
                active = (
                    parent_plantar.measurable
                    and abs(source["pitch_deg"])
                    <= policy["activation"]["source_toe_pitch_abs_max_deg"]
                    and parent_error
                    >= policy["activation"]["parent_plantar_pitch_error_min_deg"]
                )
                reasons = []
                if not parent_plantar.measurable:
                    reasons.append("TARGET_PLANTAR_UNMEASURABLE")
                if abs(source["pitch_deg"]) > policy["activation"]["source_toe_pitch_abs_max_deg"]:
                    reasons.append("SOURCE_TOE_NOT_NEAR_HORIZONTAL")
                if parent_error < policy["activation"]["parent_plantar_pitch_error_min_deg"]:
                    reasons.append("PARENT_PLANTAR_ALREADY_COMPATIBLE")
                evidence[side] = {
                    "source": source,
                    "parent_plantar": asdict(parent_plantar),
                    "parent_pitch_error_deg": parent_error,
                    "active": active,
                    "reasons": reasons,
                }
                if not active:
                    continue

                search = policy["search"]
                limit = search["correction_abs_max_deg"]
                coarse = _float_range(-limit, limit, search["coarse_step_deg"])
                coarse_rows = []
                for correction in coarse:
                    row, _ = _candidate(
                        rt, v322, mesh, rest_positions, src_arm, dst_arm,
                        report.src_profile, report.dst_profile, dst_table, side,
                        correction, palm_mu, output_mode, frame, mirror,
                        apply_root_translation, policy, source["pitch_deg"], parent_error,
                    )
                    coarse_rows.append(row)
                candidates[side].extend(coarse_rows)
                eligible = [row for row in coarse_rows if row.get("eligible")]
                if not eligible:
                    continue
                coarse_best = min(
                    eligible,
                    key=lambda row: (row["pitch_error_deg"], abs(row["correction_deg"])),
                )
                fine_start = max(-limit, coarse_best["correction_deg"] - search["fine_radius_deg"])
                fine_stop = min(limit, coarse_best["correction_deg"] + search["fine_radius_deg"])
                seen = {row["correction_deg"] for row in coarse_rows}
                for correction in _float_range(fine_start, fine_stop, search["fine_step_deg"]):
                    if correction in seen:
                        continue
                    row, _ = _candidate(
                        rt, v322, mesh, rest_positions, src_arm, dst_arm,
                        report.src_profile, report.dst_profile, dst_table, side,
                        correction, palm_mu, output_mode, frame, mirror,
                        apply_root_translation, policy, source["pitch_deg"], parent_error,
                    )
                    candidates[side].append(row)
                rescued = [
                    row for row in candidates[side]
                    if row.get("eligible")
                    and row["pitch_error_deg"] <= search["target_pitch_error_max_deg"]
                ]
                if rescued:
                    selected[side] = min(
                        rescued,
                        key=lambda row: (
                            row["pitch_error_deg"], abs(row["correction_deg"])
                        ),
                    )["correction_deg"]
            except (KeyError, RuntimeError, ValueError) as exc:
                evidence[side] = {
                    "active": False,
                    "reasons": [f"UNMEASURABLE:{exc}"],
                }

        report = v322._run_parent(
            rt, src_arm, dst_arm, src_profile=resolved_src,
            dst_profile=resolved_dst, mirror=mirror,
            apply_root_translation=apply_root_translation,
            palm_mu=palm_mu, output_mode=output_mode, frame=frame,
        )
        if report.ok:
            dst_table = bone_map.PROFILES[report.dst_profile]
            for side in _SIDES:
                if selected[side]:
                    final_applications[side] = _apply_pitch(
                        v322, rt, dst_arm, dst_table, side, selected[side]
                    )
            requested_selected = dict(selected)
            final_positions = v322._positions(mesh, dst_arm, evaluated=True)
            combined_safe = True
            for side in _SIDES:
                if not selected[side]:
                    continue
                plantar = _plantar_metrics(
                    v322, mesh, dst_arm, rt, dst_table, side,
                    rest_positions, final_positions, policy,
                )
                surface = v322._surface_metrics(
                    mesh, dst_arm, rt, dst_table, side,
                    rest_positions, final_positions, policy,
                )
                source_pitch = evidence[side]["source"]["pitch_deg"]
                pitch_error = (
                    abs(plantar.pitch_deg - source_pitch)
                    if plantar.measurable else float("inf")
                )
                surface_ok, reasons = _surface_safe(surface, policy)
                if pitch_error > policy["search"]["target_pitch_error_max_deg"]:
                    reasons.append("FINAL_PITCH_ERROR")
                final_checks[side] = {
                    "plantar": asdict(plantar),
                    "surface": asdict(surface),
                    "pitch_error_deg": pitch_error,
                    "safe": surface_ok and not reasons,
                    "reasons": reasons,
                }
                if reasons:
                    combined_safe = False
            if not combined_safe:
                selected = {side: 0.0 for side in _SIDES}
                final_applications = {}
                report = v322._run_parent(
                    rt, src_arm, dst_arm, src_profile=resolved_src,
                    dst_profile=resolved_dst, mirror=mirror,
                    apply_root_translation=apply_root_translation,
                    palm_mu=palm_mu, output_mode=output_mode, frame=frame,
                )

    report.pose_fidelity_rmse = rt.pose_fidelity_rmse(
        src_arm, dst_arm, report.src_profile, report.dst_profile, mirror=mirror
    )
    report.skeleton_baseline_rmse = rt.skeleton_baseline_rmse(
        src_arm, dst_arm, report.src_profile, report.dst_profile
    )
    report.pose_fidelity_delta = report.pose_fidelity_rmse - report.skeleton_baseline_rmse
    selection = {
        "selector": "V3.2.4_SOURCE_TOE_TARGET_PLANTAR_CONTACT_SWING",
        "status": policy["status"],
        "policy_sha256": policy_hash,
        "exact_parent_variant": policy["exact_parent_variant"],
        "evidence": evidence,
        "candidates": candidates,
        "requested_selected_correction_deg": requested_selected,
        "selected_correction_deg": selected,
        "final_applications": final_applications,
        "final_checks": final_checks,
        "fallback_to_exact_v321": selected == {side: 0.0 for side in _SIDES},
        "changed_bones_only": ["foot.L", "toe.L", "foot.R", "toe.R"],
        "source_contact_claim": "orientation evidence only; single frame has no force/velocity",
        "availability_contract": "unmeasurable_or_unsafe_exports_exact_v321",
    }
    selected_positions = v322._positions(mesh, dst_arm, evaluated=True) if mesh else None
    bpy.data.objects.remove(src_arm, do_unlink=True)
    rt.apply_output_mode(dst_arm, meshes, output_mode)
    if mesh and selected_positions is not None:
        baked = v322._positions(mesh, dst_arm, evaluated=False)
        selection["post_bake_max_vertex_error"] = max(
            (a - b).length for a, b in zip(selected_positions, baked)
        )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    rt.export_fbx(out_path, embed_textures=embed_textures, bake_anim=False)
    report.warnings.append(f"foot_plant_qa_elapsed_sec={time.time() - started:.2f}")
    return FootPlantReport(report, selection)


def math_self_test() -> dict:
    up = Vector((0.0, 0.0, 1.0))
    flat = Vector((1.0, 0.0, 0.0))
    down = Vector((1.0, 0.0, -1.0))
    result = {
        "flat_pitch_deg": _signed_pitch_deg(flat, up),
        "down_pitch_deg": _signed_pitch_deg(down, up),
    }
    result["passed"] = (
        abs(result["flat_pitch_deg"]) <= 1.0e-9
        and abs(result["down_pitch_deg"] + 45.0) <= 1.0e-9
    )
    if not result["passed"]:
        raise RuntimeError(f"V3.2.4 pitch self-test failed: {result}")
    return result
