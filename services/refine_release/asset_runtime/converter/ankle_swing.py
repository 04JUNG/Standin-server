"""V3.2.2 promoted runtime mesh-gated ankle flexion swing selector.

The module deliberately wraps the frozen, user-approved V3.2.1 palm-roll
candidate.  It never edits the parent solver and never changes hips, leg,
hand, or root transforms.  Each candidate is rebuilt from the same parent
pose; failure or an unmeasurable mesh returns the exact parent pose.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import bpy
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree


ROOT = Path(__file__).resolve().parent
PARENT_RETARGET_PATH = ROOT / "retarget.py"
POLICY_PATH = ROOT / "ankle_swing_policy.json"
_EPS = 1.0e-8
_SIDES = ("L", "R")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_parent():
    """Load the promoted runtime-owned V3.2.1 parent."""
    from . import bone_map
    from . import retarget as rt
    return rt, bone_map


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
        raise RuntimeError("unsupported ankle swing policy schema")
    if policy.get("status") != "PROMOTED_RUNTIME":
        raise RuntimeError("ankle swing policy is not promoted")
    mus = [_number(value, "candidate_mu") for value in policy.get("candidate_mu", [])]
    if not mus or mus[0] != 0.0 or any(value < 0.0 or value > 1.0 for value in mus):
        raise RuntimeError("candidate_mu must start with exact-parent 0 and stay in [0,1]")
    if mus != sorted(set(mus)):
        raise RuntimeError("candidate_mu must be unique and sorted")
    policy["candidate_mu"] = mus
    for group in ("activation", "eligibility", "measurement"):
        if not isinstance(policy.get(group), dict):
            raise RuntimeError(f"missing policy group: {group}")
        policy[group] = {
            key: _number(value, f"{group}.{key}")
            for key, value in policy[group].items()
        }
    parent_hash = _sha256(PARENT_RETARGET_PATH)
    if parent_hash != policy.get("parent_retarget_sha256"):
        raise RuntimeError(
            "frozen V3.2.1 parent retarget hash mismatch: "
            f"expected {policy.get('parent_retarget_sha256')}, got {parent_hash}"
        )
    return policy, hashlib.sha256(raw).hexdigest()


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _angle_deg(a: Vector, b: Vector) -> float:
    if a.length <= _EPS or b.length <= _EPS:
        raise RuntimeError("degenerate ankle edge")
    dot = _clamp(a.normalized().dot(b.normalized()), -1.0, 1.0)
    return math.degrees(math.acos(dot))


def _rotation_error_deg(a: Matrix, b: Matrix) -> float:
    qa, qb = a.to_quaternion().normalized(), b.to_quaternion().normalized()
    dot = _clamp(abs(qa.dot(qb)), -1.0, 1.0)
    return math.degrees(2.0 * math.acos(dot))


def _matrix_error(a: Matrix, b: Matrix) -> float:
    return max(abs(a[row][col] - b[row][col]) for row in range(4) for col in range(4))


def _scaled_rotation(rotation: Matrix, mu: float) -> Matrix:
    if mu <= 0.0:
        return Matrix.Identity(4)
    if mu >= 1.0:
        return rotation.copy()
    axis, angle = rotation.to_quaternion().normalized().to_axis_angle()
    return Matrix.Rotation(angle * mu, 4, axis)


def _signed_twist_deg(rotation: Matrix, axis: Vector) -> float:
    if axis.length <= _EPS:
        return float("inf")
    unit = axis.normalized()
    q = rotation.to_quaternion().normalized()
    projected = q.x * unit.x + q.y * unit.y + q.z * unit.z
    theta = 2.0 * math.atan2(projected, q.w)
    while theta > math.pi:
        theta -= 2.0 * math.pi
    while theta < -math.pi:
        theta += 2.0 * math.pi
    return math.degrees(theta)


def _reset_pose(arm) -> None:
    arm.data.pose_position = "POSE"
    for pose_bone in arm.pose.bones:
        pose_bone.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()


def _world_head(rt, arm, name: str, *, rest: bool) -> Vector:
    matrix = rt._rest_world(arm, name) if rest else rt._pose_world(arm, name)
    return matrix.translation.copy()


def _source_point(rt, bone_map, arm, table: dict[str, str], canonical: str,
                  *, rest: bool, mirror: bool) -> Vector:
    source_canonical = bone_map.mirror_name(canonical) if mirror else canonical
    point = _world_head(rt, arm, table[source_canonical], rest=rest)
    if mirror:
        point.x = -point.x
    return point


def _target_point(rt, arm, table: dict[str, str], canonical: str,
                  *, rest: bool) -> Vector:
    return _world_head(rt, arm, table[canonical], rest=rest)


@dataclass
class SwingGeometry:
    measurable: bool
    status: str
    source_rest_bend_deg: float | None = None
    source_pose_bend_deg: float | None = None
    source_rest_relative_bend_deg: float | None = None
    target_rest_bend_deg: float | None = None
    parent_pose_bend_deg: float | None = None
    parent_rest_relative_bend_deg: float | None = None
    desired_bend_deg: float | None = None
    parent_bend_error_deg: float | None = None
    correction_full_deg: float | None = None


def _swing_geometry(rt, bone_map, src_arm, dst_arm, src_table, dst_table,
                    side: str, mirror: bool) -> tuple[SwingGeometry, dict | None]:
    names = (f"leg.{side}", f"foot.{side}", f"toe.{side}")
    try:
        sr = [_source_point(rt, bone_map, src_arm, src_table, name,
                            rest=True, mirror=mirror) for name in names]
        sp = [_source_point(rt, bone_map, src_arm, src_table, name,
                            rest=False, mirror=mirror) for name in names]
        tr = [_target_point(rt, dst_arm, dst_table, name, rest=True) for name in names]
        tp = [_target_point(rt, dst_arm, dst_table, name, rest=False) for name in names]
        source_rest = _angle_deg(sr[1] - sr[0], sr[2] - sr[1])
        source_pose = _angle_deg(sp[1] - sp[0], sp[2] - sp[1])
        target_rest = _angle_deg(tr[1] - tr[0], tr[2] - tr[1])
        parent_pose = _angle_deg(tp[1] - tp[0], tp[2] - tp[1])
        source_motion = source_pose - source_rest
        desired = target_rest + source_motion
        # beta is an unsigned angle in [0, 180].  Crossing either endpoint
        # loses the bend-plane sign, so inventing a clamped direction would be
        # a different solver.  Protect that ankle with the exact parent.
        if not 0.5 <= desired <= 179.5:
            return SwingGeometry(
                False,
                f"desired_bend_out_of_range:{desired:.6f}",
                source_rest_bend_deg=source_rest,
                source_pose_bend_deg=source_pose,
                source_rest_relative_bend_deg=source_motion,
                target_rest_bend_deg=target_rest,
                parent_pose_bend_deg=parent_pose,
                parent_rest_relative_bend_deg=parent_pose - target_rest,
                desired_bend_deg=desired,
                parent_bend_error_deg=abs(parent_pose - desired),
            ), None
        shin = (tp[1] - tp[0]).normalized()
        foot = (tp[2] - tp[1]).normalized()
        perpendicular = foot - shin * foot.dot(shin)
        if perpendicular.length <= 1.0e-6:
            return SwingGeometry(False, "parent_ankle_plane_degenerate"), None
        perpendicular.normalize()
        desired_rad = math.radians(desired)
        desired_direction = (
            math.cos(desired_rad) * shin + math.sin(desired_rad) * perpendicular
        ).normalized()
        full, why = rt._min_rotation(foot, desired_direction)
        if full is None:
            return SwingGeometry(False, f"minimum_swing_degenerate:{why}"), None
        geometry = SwingGeometry(
            True,
            "ok",
            source_rest_bend_deg=source_rest,
            source_pose_bend_deg=source_pose,
            source_rest_relative_bend_deg=source_motion,
            target_rest_bend_deg=target_rest,
            parent_pose_bend_deg=parent_pose,
            parent_rest_relative_bend_deg=parent_pose - target_rest,
            desired_bend_deg=desired,
            parent_bend_error_deg=abs(parent_pose - desired),
            correction_full_deg=_rotation_error_deg(Matrix.Identity(4), full),
        )
        return geometry, {
            "full": full,
            "base_foot_direction": foot,
            "desired_direction": desired_direction,
            "target_points": tp,
        }
    except (KeyError, RuntimeError, ValueError) as exc:
        return SwingGeometry(False, str(exc)), None


def _apply_side_swing(rt, dst_arm, dst_table: dict[str, str], side: str,
                      context: dict, mu: float) -> dict:
    foot_name = dst_table[f"foot.{side}"]
    toe_name = dst_table[f"toe.{side}"]
    base_foot = rt._pose_world(dst_arm, foot_name).copy()
    base_toe = rt._pose_world(dst_arm, toe_name).copy()
    base_relative = base_foot.to_quaternion().inverted() @ base_toe.to_quaternion()
    correction = _scaled_rotation(context["full"], mu)
    twist = _signed_twist_deg(correction, context["base_foot_direction"])

    desired_foot = correction @ rt._rot_only(base_foot)
    desired_foot.translation = rt._pose_world(dst_arm, foot_name).translation
    dst_arm.pose.bones[foot_name].matrix = dst_arm.matrix_world.inverted() @ desired_foot
    bpy.context.view_layer.update()

    desired_toe = correction @ rt._rot_only(base_toe)
    desired_toe.translation = rt._pose_world(dst_arm, toe_name).translation
    dst_arm.pose.bones[toe_name].matrix = dst_arm.matrix_world.inverted() @ desired_toe
    bpy.context.view_layer.update()

    after_foot = rt._pose_world(dst_arm, foot_name)
    after_toe = rt._pose_world(dst_arm, toe_name)
    after_relative = after_foot.to_quaternion().inverted() @ after_toe.to_quaternion()
    return {
        "mu": mu,
        "applied_swing_deg": _rotation_error_deg(Matrix.Identity(4), correction),
        "correction_twist_deg": twist,
        "toe_relative_rotation_error_deg": math.degrees(
            base_relative.rotation_difference(after_relative).angle
        ),
    }


def _positions(mesh, arm, *, evaluated: bool) -> list[Vector]:
    transform = arm.matrix_world.inverted() @ mesh.matrix_world
    if not evaluated:
        return [transform @ vertex.co for vertex in mesh.data.vertices]
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_object = mesh.evaluated_get(depsgraph)
    temporary = evaluated_object.to_mesh(
        preserve_all_data_layers=True, depsgraph=depsgraph
    )
    try:
        return [transform @ vertex.co for vertex in temporary.vertices]
    finally:
        evaluated_object.to_mesh_clear()


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    at = q * (len(ordered) - 1)
    lo = int(math.floor(at))
    hi = int(math.ceil(at))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - at) + ordered[hi] * (at - lo)


def _triangle_area(points: list[Vector], tri: tuple[int, int, int]) -> float:
    a, b, c = (points[index] for index in tri)
    return 0.5 * (b - a).cross(c - a).length


def _triangle_normal(points: list[Vector], tri: tuple[int, int, int]) -> Vector | None:
    a, b, c = (points[index] for index in tri)
    normal = (b - a).cross(c - a)
    if normal.length <= _EPS:
        return None
    return normal.normalized()


def _vertex_weight_rows(mesh) -> list[dict[str, float]]:
    group_names = {group.index: group.name for group in mesh.vertex_groups}
    rows = []
    for vertex in mesh.data.vertices:
        rows.append({
            group_names[item.group]: float(item.weight)
            for item in vertex.groups if item.group in group_names
        })
    return rows


def _clearance(points: list[Vector], foot_indices: list[int], calf_indices: list[int],
               scale: float) -> tuple[float, float]:
    tree = KDTree(len(calf_indices))
    for slot, index in enumerate(calf_indices):
        tree.insert(points[index], slot)
    tree.balance()
    distances = [tree.find(points[index])[2] / scale for index in foot_indices]
    return min(distances), _quantile(distances, 0.01)


@dataclass
class SurfaceMetrics:
    measurable: bool
    status: str
    roi_vertex_count: int = 0
    roi_triangle_count: int = 0
    roi_edge_count: int = 0
    foot_clearance_vertex_count: int = 0
    calf_clearance_vertex_count: int = 0
    area_ratio_min: float | None = None
    area_ratio_p01: float | None = None
    edge_ratio_min: float | None = None
    edge_ratio_max: float | None = None
    log_edge_strain_p99: float | None = None
    clearance_min_leg_ratio: float | None = None
    clearance_p01_leg_ratio: float | None = None
    dihedral_change_p99_deg: float | None = None
    new_sharp_fold_count: int = 0


class AnkleSwingReport:
    """Non-invasive report extension around the frozen parent's dataclass."""

    def __init__(self, parent, selection: dict):
        self.parent = parent
        self.ankle_swing_selection = selection

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def as_dict(self) -> dict:
        payload = self.parent.as_dict()
        payload["ankle_swing_selection"] = self.ankle_swing_selection
        return payload


def _surface_metrics(mesh, arm, rt, dst_table: dict[str, str], side: str,
                     rest_positions: list[Vector], posed_positions: list[Vector],
                     policy: dict) -> SurfaceMetrics:
    measure = policy["measurement"]
    inv = arm.matrix_world.inverted()
    knee = inv @ rt._rest_world(arm, dst_table[f"leg.{side}"]).translation
    ankle = inv @ rt._rest_world(arm, dst_table[f"foot.{side}"]).translation
    toe = inv @ rt._rest_world(arm, dst_table[f"toe.{side}"]).translation
    leg_length = (ankle - knee).length
    foot_length = (toe - ankle).length
    if leg_length <= _EPS or foot_length <= _EPS:
        return SurfaceMetrics(False, "target_rest_edges_degenerate")
    shin_to_knee = (knee - ankle).normalized()
    foot_forward = (toe - ankle).normalized()
    names = {
        "leg": dst_table[f"leg.{side}"],
        "foot": dst_table[f"foot.{side}"],
        "toe": dst_table[f"toe.{side}"],
    }
    weights = _vertex_weight_rows(mesh)
    radius = max(
        measure["roi_radius_leg_ratio"] * leg_length,
        measure["roi_radius_foot_ratio"] * foot_length,
    )
    roi: set[int] = set()
    foot_vertices: list[int] = []
    calf_vertices: list[int] = []
    for index, (point, row) in enumerate(zip(rest_positions, weights)):
        w_leg = row.get(names["leg"], 0.0)
        w_foot = row.get(names["foot"], 0.0)
        w_toe = row.get(names["toe"], 0.0)
        if w_leg + w_foot + w_toe >= measure["minimum_limb_weight"]:
            if (point - ankle).length <= radius:
                roi.add(index)
        foot_projection = (point - ankle).dot(foot_forward) / foot_length
        if w_foot + w_toe >= measure["foot_weight_min"] and (
            foot_projection >= measure["foot_distal_min_ratio"]
        ):
            foot_vertices.append(index)
        calf_projection = (point - ankle).dot(shin_to_knee) / leg_length
        if w_leg >= measure["calf_weight_min"] and (
            measure["calf_proximal_min_ratio"] <= calf_projection
            <= measure["calf_proximal_max_ratio"]
        ):
            calf_vertices.append(index)

    mesh.data.calc_loop_triangles()
    triangles = [tuple(tri.vertices) for tri in mesh.data.loop_triangles]
    roi_triangles = [tri for tri in triangles if sum(v in roi for v in tri) >= 2]
    roi_edges = [
        (edge.vertices[0], edge.vertices[1]) for edge in mesh.data.edges
        if edge.vertices[0] in roi and edge.vertices[1] in roi
    ]
    minimum = int(measure["minimum_clearance_vertices"])
    if (
        len(roi) < int(measure["minimum_roi_vertices"])
        or len(roi_triangles) < int(measure["minimum_roi_triangles"])
        or len(roi_edges) < int(measure["minimum_roi_triangles"])
        or len(foot_vertices) < minimum
        or len(calf_vertices) < minimum
    ):
        return SurfaceMetrics(
            False, "ankle_mesh_roi_unmeasurable", len(roi), len(roi_triangles),
            len(roi_edges), len(foot_vertices), len(calf_vertices)
        )

    area_ratios = []
    for tri in roi_triangles:
        rest_area = _triangle_area(rest_positions, tri)
        posed_area = _triangle_area(posed_positions, tri)
        if rest_area > _EPS:
            area_ratios.append(posed_area / rest_area)
    edge_ratios = []
    log_strain = []
    for a, b in roi_edges:
        rest_length = (rest_positions[a] - rest_positions[b]).length
        posed_length = (posed_positions[a] - posed_positions[b]).length
        if rest_length > _EPS and posed_length > _EPS:
            ratio = posed_length / rest_length
            edge_ratios.append(ratio)
            log_strain.append(abs(math.log(ratio)))
    if not area_ratios or not edge_ratios:
        return SurfaceMetrics(False, "ankle_surface_distributions_empty")

    edge_to_triangles: dict[tuple[int, int], list[int]] = {}
    for tri_index, tri in enumerate(roi_triangles):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_to_triangles.setdefault(tuple(sorted((a, b))), []).append(tri_index)
    dihedral_changes = []
    new_sharp = 0
    rest_normals = [_triangle_normal(rest_positions, tri) for tri in roi_triangles]
    pose_normals = [_triangle_normal(posed_positions, tri) for tri in roi_triangles]
    for adjacent in edge_to_triangles.values():
        if len(adjacent) != 2:
            continue
        first, second = adjacent
        rn1, rn2 = rest_normals[first], rest_normals[second]
        pn1, pn2 = pose_normals[first], pose_normals[second]
        if None in (rn1, rn2, pn1, pn2):
            continue
        rest_angle = math.degrees(math.acos(_clamp(rn1.dot(rn2), -1.0, 1.0)))
        pose_angle = math.degrees(math.acos(_clamp(pn1.dot(pn2), -1.0, 1.0)))
        dihedral_changes.append(abs(pose_angle - rest_angle))
        if rest_angle < 100.0 and pose_angle > 130.0:
            new_sharp += 1
    clearance_min, clearance_p01 = _clearance(
        posed_positions, foot_vertices, calf_vertices, leg_length
    )
    return SurfaceMetrics(
        True,
        "ok",
        len(roi),
        len(roi_triangles),
        len(roi_edges),
        len(foot_vertices),
        len(calf_vertices),
        min(area_ratios),
        _quantile(area_ratios, 0.01),
        min(edge_ratios),
        max(edge_ratios),
        _quantile(log_strain, 0.99),
        clearance_min,
        clearance_p01,
        _quantile(dihedral_changes, 0.99) if dihedral_changes else 0.0,
        new_sharp,
    )


def _activation(geometry: SwingGeometry, surface: SurfaceMetrics,
                policy: dict) -> tuple[bool, list[str]]:
    if not geometry.measurable or not surface.measurable:
        return False, ["UNMEASURABLE"]
    limits = policy["activation"]
    reasons = []
    if geometry.parent_bend_error_deg < limits["bend_error_min_deg"]:
        reasons.append("BEND_ERROR_BELOW_ACTIVATION")
    if abs(geometry.parent_rest_relative_bend_deg) < limits["output_rest_motion_min_deg"]:
        reasons.append("OUTPUT_MOTION_BELOW_ACTIVATION")
    mesh_symptoms = []
    if surface.clearance_p01_leg_ratio <= limits["clearance_p01_max_leg_ratio"]:
        mesh_symptoms.append("LOW_FOOT_CALF_CLEARANCE")
    if surface.area_ratio_p01 < limits["area_ratio_p01_min"]:
        mesh_symptoms.append("ANKLE_AREA_COMPRESSION")
    if surface.edge_ratio_min < limits["edge_ratio_min"]:
        mesh_symptoms.append("ANKLE_EDGE_COMPRESSION")
    if surface.log_edge_strain_p99 > limits["log_edge_strain_p99_max"]:
        mesh_symptoms.append("ANKLE_EDGE_STRAIN")
    if surface.new_sharp_fold_count > 0:
        mesh_symptoms.append("NEW_SHARP_FOLD")
    if abs(geometry.parent_rest_relative_bend_deg) >= limits["extreme_output_rest_motion_deg"]:
        mesh_symptoms.append("EXTREME_TARGET_REST_RELATIVE_MOTION")
    if not mesh_symptoms:
        reasons.append("NO_MESH_OR_EXTREME_MOTION_SYMPTOM")
    return not reasons, reasons + mesh_symptoms


def _surface_nonregression(candidate: SurfaceMetrics, baseline: SurfaceMetrics,
                           policy: dict) -> tuple[bool, list[str], dict]:
    if not candidate.measurable or not baseline.measurable:
        return False, ["UNMEASURABLE_SURFACE"], {}
    gate = policy["eligibility"]
    reasons = []
    if candidate.area_ratio_p01 < baseline.area_ratio_p01 - gate["area_ratio_p01_regression"]:
        reasons.append("AREA_P01_REGRESSION")
    if candidate.edge_ratio_min < baseline.edge_ratio_min - gate["edge_ratio_min_regression"]:
        reasons.append("EDGE_MIN_REGRESSION")
    if candidate.log_edge_strain_p99 > (
        baseline.log_edge_strain_p99 + gate["log_edge_strain_p99_regression"]
    ):
        reasons.append("LOG_EDGE_STRAIN_REGRESSION")
    if candidate.clearance_p01_leg_ratio < (
        baseline.clearance_p01_leg_ratio - gate["clearance_p01_regression_leg_ratio"]
    ):
        reasons.append("CLEARANCE_REGRESSION")
    if candidate.dihedral_change_p99_deg > (
        baseline.dihedral_change_p99_deg + gate["dihedral_change_p99_regression_deg"]
    ):
        reasons.append("DIHEDRAL_REGRESSION")
    if candidate.new_sharp_fold_count > baseline.new_sharp_fold_count:
        reasons.append("NEW_SHARP_FOLD_REGRESSION")
    improvements = {
        "clearance_p01_leg_ratio": (
            candidate.clearance_p01_leg_ratio - baseline.clearance_p01_leg_ratio
        ),
        "area_ratio_p01": candidate.area_ratio_p01 - baseline.area_ratio_p01,
        "log_edge_strain_p99": (
            baseline.log_edge_strain_p99 - candidate.log_edge_strain_p99
        ),
    }
    practical = (
        improvements["clearance_p01_leg_ratio"]
        >= gate["minimum_clearance_improvement_leg_ratio"]
        or improvements["area_ratio_p01"] >= gate["minimum_area_p01_improvement"]
        or improvements["log_edge_strain_p99"]
        >= gate["minimum_log_strain_improvement"]
    )
    if not practical:
        reasons.append("NO_PRACTICAL_MESH_IMPROVEMENT")
    return not reasons, reasons, improvements


def _pose_snapshot(rt, arm, dst_table: dict[str, str]) -> dict[str, Matrix]:
    return {
        canonical: rt._pose_world(arm, name).copy()
        for canonical, name in dst_table.items() if name in arm.pose.bones
    }


def _proximal_error(before: dict[str, Matrix], after: dict[str, Matrix]) -> float:
    excluded = {"foot.L", "toe.L", "foot.R", "toe.R"}
    common = (set(before) & set(after)) - excluded
    return max((_matrix_error(before[name], after[name]) for name in common), default=0.0)


def _run_parent(rt, src_arm, dst_arm, *, src_profile, dst_profile, mirror,
                apply_root_translation, palm_mu, output_mode, frame):
    _reset_pose(dst_arm)
    report = rt.ConvertReport(output_mode=output_mode, frame=frame)
    return rt.retarget(
        src_arm,
        dst_arm,
        src_profile=src_profile,
        dst_profile=dst_profile,
        mirror=mirror,
        apply_root_translation=apply_root_translation,
        palm_roll_mu=palm_mu,
        report=report,
    )


def _candidate_row(rt, bone_map, src_arm, dst_arm, mesh, rest_positions,
                   policy, resolved_src, resolved_dst, mirror, output_mode, frame,
                   apply_root_translation, palm_mu, mus, baseline_surfaces=None,
                   activation=None) -> tuple[dict, object]:
    report = _run_parent(
        rt, src_arm, dst_arm, src_profile=resolved_src, dst_profile=resolved_dst,
        mirror=mirror, apply_root_translation=apply_root_translation,
        palm_mu=palm_mu, output_mode=output_mode, frame=frame,
    )
    row = {"mu": {side: float(mus[side]) for side in _SIDES}, "parent_ok": report.ok}
    if not report.ok:
        row.update({"eligible": False, "eligibility_reasons": ["PARENT_FAILED"]})
        return row, report
    src_table = bone_map.PROFILES[report.src_profile]
    dst_table = bone_map.PROFILES[report.dst_profile]
    before = _pose_snapshot(rt, dst_arm, dst_table)
    geometries = {}
    contexts = {}
    for side in _SIDES:
        geometry, context = _swing_geometry(
            rt, bone_map, src_arm, dst_arm, src_table, dst_table, side, mirror
        )
        geometries[side], contexts[side] = geometry, context
    applied = {}
    for side in _SIDES:
        mu = float(mus[side])
        if mu > 0.0 and contexts[side] is not None:
            applied[side] = _apply_side_swing(
                rt, dst_arm, dst_table, side, contexts[side], mu
            )
        else:
            applied[side] = {
                "mu": mu,
                "applied_swing_deg": 0.0,
                "correction_twist_deg": 0.0,
                "toe_relative_rotation_error_deg": 0.0,
            }
    after = _pose_snapshot(rt, dst_arm, dst_table)
    posed_positions = _positions(mesh, dst_arm, evaluated=True)
    surfaces = {
        side: _surface_metrics(
            mesh, dst_arm, rt, dst_table, side, rest_positions, posed_positions, policy
        ) for side in _SIDES
    }
    for side in _SIDES:
        if geometries[side].measurable:
            knee = _target_point(rt, dst_arm, dst_table, f"leg.{side}", rest=False)
            ankle = _target_point(rt, dst_arm, dst_table, f"foot.{side}", rest=False)
            toe = _target_point(rt, dst_arm, dst_table, f"toe.{side}", rest=False)
            bend = _angle_deg(ankle - knee, toe - ankle)
            applied[side]["result_bend_deg"] = bend
            applied[side]["result_bend_error_deg"] = abs(
                bend - geometries[side].desired_bend_deg
            )
    row.update({
        "geometry": {side: asdict(geometries[side]) for side in _SIDES},
        "applied": applied,
        "surface": {side: asdict(surfaces[side]) for side in _SIDES},
        "proximal_matrix_error": _proximal_error(before, after),
    })
    if baseline_surfaces is None or activation is None:
        row.update({"eligible": True, "eligibility_reasons": ["EXACT_PARENT_BASELINE"]})
        return row, report

    gate = policy["eligibility"]
    reasons = []
    practical = {}
    changed = False
    for side in _SIDES:
        mu = float(mus[side])
        if mu <= 0.0:
            continue
        changed = True
        if not activation[side]["active"]:
            reasons.append(f"{side}:NORMAL_ANKLE_PROTECTED")
            continue
        geometry = geometries[side]
        if not geometry.measurable:
            reasons.append(f"{side}:GEOMETRY_UNMEASURABLE")
            continue
        improvement = geometry.parent_bend_error_deg - applied[side]["result_bend_error_deg"]
        applied[side]["bend_improvement_deg"] = improvement
        if improvement < gate["minimum_bend_improvement_deg"]:
            reasons.append(f"{side}:BEND_NOT_IMPROVED")
        if abs(applied[side]["correction_twist_deg"]) > gate["maximum_correction_twist_deg"]:
            reasons.append(f"{side}:TWIST_CHANGED")
        if applied[side]["toe_relative_rotation_error_deg"] > gate["maximum_toe_relative_rotation_error_deg"]:
            reasons.append(f"{side}:TOE_RELATIVE_ROTATION_CHANGED")
        surface_ok, surface_reasons, improvements = _surface_nonregression(
            surfaces[side], baseline_surfaces[side], policy
        )
        practical[side] = improvements
        if not surface_ok:
            reasons.extend(f"{side}:{reason}" for reason in surface_reasons)
    if not changed:
        reasons.append("NON_BASELINE_CANDIDATE_DID_NOT_CHANGE")
    if row["proximal_matrix_error"] > gate["maximum_proximal_matrix_error"]:
        reasons.append("PROXIMAL_MATRIX_REGRESSION")
    row["mesh_improvements"] = practical
    row["eligible"] = not reasons
    row["eligibility_reasons"] = reasons
    return row, report


def convert_safe(
    *,
    bvh_path: str,
    character_fbx: str,
    out_path: str,
    frame: int = 0,
    mirror: bool = False,
    output_mode: str = "rigged_rest",
    apply_root_translation: bool = False,
    embed_textures: bool = True,
    src_profile: str | None = None,
    dst_profile: str | None = None,
    policy_path: str | os.PathLike[str] = POLICY_PATH,
):
    """Evaluate swing candidates and export a selected or exact-parent FBX.

    The function never withholds an FBX because of this QA selector.  Any
    uncertainty selects ``mu=0`` on both ankles, which is the exact V3.2.1
    parent including the approved palm-roll ``mu=0.5``.
    """
    started = time.time()
    policy, policy_hash = load_policy(policy_path)
    rt, bone_map = _load_parent()
    if output_mode not in {"rigged_rest", "static_mesh"}:
        raise ValueError("ankle swing QA supports rigged_rest/static_mesh only")
    rt.reset_scene()
    dst_arm, meshes = rt.import_character(character_fbx)
    src_arm = rt.import_bvh(bvh_path, frame=frame)
    dst_names = [bone.name for bone in dst_arm.data.bones]
    src_names = [bone.name for bone in src_arm.data.bones]
    resolved_src = src_profile or bone_map.resolve_profile(src_names)
    resolved_dst = dst_profile or bone_map.resolve_profile(dst_names)
    palm_mu = float(policy["parent_palm_roll_mu"])

    # Multiple meshes are safe but not automatically measurable as one surface.
    # Preserve availability by exporting the exact parent rather than guessing.
    measurable_mesh = meshes[0] if len(meshes) == 1 else None
    rest_positions = (
        _positions(measurable_mesh, dst_arm, evaluated=False)
        if measurable_mesh is not None else None
    )
    baseline_mus = {side: 0.0 for side in _SIDES}
    if measurable_mesh is None:
        report = _run_parent(
            rt, src_arm, dst_arm, src_profile=resolved_src, dst_profile=resolved_dst,
            mirror=mirror, apply_root_translation=apply_root_translation,
            palm_mu=palm_mu, output_mode=output_mode, frame=frame,
        )
        rows = []
        activation = {side: {"active": False, "reasons": ["MESH_COUNT_NOT_ONE"]}
                      for side in _SIDES}
        selected_mus = baseline_mus
    else:
        baseline, report = _candidate_row(
            rt, bone_map, src_arm, dst_arm, measurable_mesh, rest_positions,
            policy, resolved_src, resolved_dst, mirror, output_mode, frame,
            apply_root_translation, palm_mu, baseline_mus,
        )
        rows = [baseline]
        baseline_surfaces = {
            side: SurfaceMetrics(**baseline["surface"][side]) for side in _SIDES
        }
        activation = {}
        for side in _SIDES:
            geometry = SwingGeometry(**baseline["geometry"][side])
            active, reasons = _activation(
                geometry, baseline_surfaces[side], policy
            )
            activation[side] = {"active": active, "reasons": reasons}

        ladders = [
            policy["candidate_mu"] if activation[side]["active"] else [0.0]
            for side in _SIDES
        ]
        for left, right in itertools.product(*ladders):
            mus = {"L": float(left), "R": float(right)}
            if mus == baseline_mus:
                continue
            row, _ = _candidate_row(
                rt, bone_map, src_arm, dst_arm, measurable_mesh, rest_positions,
                policy, resolved_src, resolved_dst, mirror, output_mode, frame,
                apply_root_translation, palm_mu, mus,
                baseline_surfaces=baseline_surfaces, activation=activation,
            )
            rows.append(row)
        eligible = [row for row in rows if row.get("eligible")]
        selected = min(
            eligible,
            key=lambda row: (
                sum(
                    row.get("applied", {}).get(side, {}).get(
                        "result_bend_error_deg",
                        row.get("geometry", {}).get(side, {}).get(
                            "parent_bend_error_deg", 0.0
                        ) or 0.0,
                    )
                    for side in _SIDES if activation[side]["active"]
                ),
                sum(row["mu"].values()),
                row["mu"]["L"],
                row["mu"]["R"],
            ),
        )
        selected_mus = selected["mu"]
        report = _run_parent(
            rt, src_arm, dst_arm, src_profile=resolved_src, dst_profile=resolved_dst,
            mirror=mirror, apply_root_translation=apply_root_translation,
            palm_mu=palm_mu, output_mode=output_mode, frame=frame,
        )
        if report.ok:
            src_table = bone_map.PROFILES[report.src_profile]
            dst_table = bone_map.PROFILES[report.dst_profile]
            for side in _SIDES:
                geometry, context = _swing_geometry(
                    rt, bone_map, src_arm, dst_arm, src_table, dst_table, side, mirror
                )
                if selected_mus[side] > 0.0:
                    if not geometry.measurable or context is None:
                        raise RuntimeError("selected swing became unmeasurable on final rerun")
                    _apply_side_swing(
                        rt, dst_arm, dst_table, side, context, selected_mus[side]
                    )

    if not report.ok:
        return report
    report.pose_fidelity_rmse = rt.pose_fidelity_rmse(
        src_arm, dst_arm, report.src_profile, report.dst_profile, mirror=mirror
    )
    report.skeleton_baseline_rmse = rt.skeleton_baseline_rmse(
        src_arm, dst_arm, report.src_profile, report.dst_profile
    )
    report.pose_fidelity_delta = (
        report.pose_fidelity_rmse - report.skeleton_baseline_rmse
    )
    selection = {
        "selector": "V3.2.2_ANKLE_FLEXION_SWING_ACTUAL_MESH",
        "status": policy["status"],
        "policy_sha256": policy_hash,
        "parent_variant": policy["parent_variant"],
        "parent_retarget_sha256": policy["parent_retarget_sha256"],
        "parent_palm_roll_mu": palm_mu,
        "activation": activation,
        "candidates": rows,
        "selected_mu": selected_mus,
        "fallback_to_exact_parent": selected_mus == baseline_mus,
        "changed_bones_only": ["foot.L", "toe.L", "foot.R", "toe.R"],
        "final_application_count": 1,
        "availability_contract": "uncertain_or_failed_selector_exports_exact_parent",
    }

    selected_positions = (
        _positions(measurable_mesh, dst_arm, evaluated=True)
        if measurable_mesh is not None else None
    )
    bpy.data.objects.remove(src_arm, do_unlink=True)
    rt.apply_output_mode(dst_arm, meshes, output_mode)
    if measurable_mesh is not None and selected_positions is not None:
        baked_positions = _positions(measurable_mesh, dst_arm, evaluated=False)
        selection["post_bake_max_vertex_error"] = max(
            (a - b).length for a, b in zip(selected_positions, baked_positions)
        )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    rt.export_fbx(
        out_path,
        embed_textures=embed_textures,
        bake_anim=False,
    )
    report.warnings.append(f"ankle_swing_qa_elapsed_sec={time.time() - started:.2f}")
    return AnkleSwingReport(report, selection)


def math_self_test() -> dict:
    """Pure geometry controls for the rest-relative bend equation."""
    shin = Vector((0.0, 1.0, 0.0))
    base = Vector((math.sin(math.radians(140.0)), math.cos(math.radians(140.0)), 0.0))
    desired_deg = 55.0 + (140.0 - 76.0)
    perpendicular = (base - shin * base.dot(shin)).normalized()
    desired = (
        math.cos(math.radians(desired_deg)) * shin
        + math.sin(math.radians(desired_deg)) * perpendicular
    ).normalized()
    rt, _ = _load_parent()
    full, why = rt._min_rotation(base, desired)
    if full is None:
        raise RuntimeError(why)
    result = {
        "desired_bend_deg": desired_deg,
        "recovered_bend_deg": _angle_deg(shin, full.to_3x3() @ base),
        "full_correction_deg": _rotation_error_deg(Matrix.Identity(4), full),
        "twist_deg": _signed_twist_deg(full, base),
        "mu0_matrix_error": _matrix_error(_scaled_rotation(full, 0.0), Matrix.Identity(4)),
        "mirror_angle_error_deg": abs(
            _angle_deg(Vector((-shin.x, shin.y, shin.z)),
                       Vector((-(full.to_3x3() @ base).x,
                               (full.to_3x3() @ base).y,
                               (full.to_3x3() @ base).z)))
            - desired_deg
        ),
    }
    policy, _ = load_policy()
    base_surface = SurfaceMetrics(
        True, "ok", 100, 100, 100, 20, 20,
        0.4, 0.7, 0.6, 1.4, 0.4, 0.04, 0.06, 8.0, 0,
    )
    identical_ok, identical_reasons, _ = _surface_nonregression(
        base_surface, base_surface, policy
    )
    improved_surface = SurfaceMetrics(**asdict(base_surface))
    improved_surface.clearance_p01_leg_ratio += 0.01
    improved_ok, improved_reasons, _ = _surface_nonregression(
        improved_surface, base_surface, policy
    )
    regressed_surface = SurfaceMetrics(**asdict(base_surface))
    regressed_surface.area_ratio_p01 -= 0.1
    regressed_ok, regressed_reasons, _ = _surface_nonregression(
        regressed_surface, base_surface, policy
    )
    normal_geometry = SwingGeometry(
        True, "ok", parent_bend_error_deg=0.2,
        parent_rest_relative_bend_deg=6.0,
    )
    normal_active, _ = _activation(normal_geometry, improved_surface, policy)
    issue_geometry = SwingGeometry(
        True, "ok", parent_bend_error_deg=20.0,
        parent_rest_relative_bend_deg=85.0,
    )
    issue_active, _ = _activation(issue_geometry, base_surface, policy)
    result["selector_controls"] = {
        "identical_surface_rejected": (
            not identical_ok and "NO_PRACTICAL_MESH_IMPROVEMENT" in identical_reasons
        ),
        "improved_surface_accepted": improved_ok and not improved_reasons,
        "regressed_surface_rejected": (
            not regressed_ok and "AREA_P01_REGRESSION" in regressed_reasons
        ),
        "normal_geometry_inactive": not normal_active,
        "issue_geometry_active": issue_active,
    }
    # Blender mathutils stores these vectors/matrices as float32.  The 5e-5°
    # bound is only a pure-math equality tolerance; it is not a product gate.
    math_tolerance_deg = 5.0e-5
    result["math_tolerance_deg"] = math_tolerance_deg
    result["passed"] = (
        abs(result["recovered_bend_deg"] - desired_deg) <= math_tolerance_deg
        and abs(result["twist_deg"]) <= 1.0e-5
        and result["mu0_matrix_error"] == 0.0
        and result["mirror_angle_error_deg"] <= math_tolerance_deg
        and all(result["selector_controls"].values())
    )
    if not result["passed"]:
        raise RuntimeError(f"ankle swing math self-test failed: {result}")
    return result
