"""V3.2 기준 대비 palm-roll 후보의 실제 wrist/hand surface 상대 지표.

임계는 아직 동결하지 않았으므로 이 도구는 PASS를 만들지 않는다. 실제 deform group과
armature hierarchy로 ROI를 구성하고, 수치와 측정 가능 여부만 기록한다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from converter import bone_map  # noqa: E402


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    rows = sorted(values)
    at = (len(rows) - 1) * q
    lo, hi = math.floor(at), math.ceil(at)
    if lo == hi:
        return rows[lo]
    return rows[lo] * (hi - at) + rows[hi] * (at - lo)


def triangle_area(points: list[Vector], tri: tuple[int, int, int]) -> float:
    a, b, c = (points[index] for index in tri)
    return 0.5 * (b - a).cross(c - a).length


def self_intersections(points: list[Vector], triangles: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    tree = BVHTree.FromPolygons(points, triangles, all_triangles=True, epsilon=1e-8)
    out = set()
    for left, right in tree.overlap(tree):
        if left == right or set(triangles[left]).intersection(triangles[right]):
            continue
        out.add((min(left, right), max(left, right)))
    return out


def snapshot(path: str) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(
        filepath=path,
        ignore_leaf_bones=True,
        automatic_bone_orientation=False,
    )
    arms = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(arms) != 1 or len(meshes) != 1:
        raise RuntimeError(f"expected one armature/mesh, got {len(arms)}/{len(meshes)}")
    arm, mesh = arms[0], meshes[0]
    profile_name = bone_map.resolve_profile([bone.name for bone in arm.data.bones])
    table = bone_map.PROFILES[profile_name]
    group_by_index = {group.index: group.name for group in mesh.vertex_groups}

    side_groups: dict[str, set[str]] = {}
    for side in ("L", "R"):
        hand_name = table[f"hand.{side}"]
        forearm_name = table[f"forearm.{side}"]
        hand_bone = arm.data.bones[hand_name]
        descendants = {child.name for child in hand_bone.children_recursive}
        side_groups[side] = {forearm_name, hand_name, *descendants}

    side_vertices = {"L": set(), "R": set()}
    for vertex in mesh.data.vertices:
        weighted = {
            group_by_index.get(item.group)
            for item in vertex.groups
            if item.weight > 1e-6
        }
        for side in ("L", "R"):
            if weighted.intersection(side_groups[side]):
                side_vertices[side].add(vertex.index)

    mesh.data.calc_loop_triangles()
    triangles = [tuple(tri.vertices) for tri in mesh.data.loop_triangles]
    points = [mesh.matrix_world @ vertex.co for vertex in mesh.data.vertices]
    return {
        "profile": profile_name,
        "points": points,
        "triangles": triangles,
        "side_vertices": side_vertices,
        "side_groups": {side: sorted(groups) for side, groups in side_groups.items()},
    }


def side_metrics(base: dict, candidate: dict, side: str) -> dict:
    vertices = base["side_vertices"][side]
    triangles = base["triangles"]
    edges = {
        tuple(sorted(edge))
        for tri in triangles
        for edge in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
        if tri[0] in vertices or tri[1] in vertices or tri[2] in vertices
    }
    roi_triangles = {
        index for index, tri in enumerate(triangles)
        if any(vertex in vertices for vertex in tri)
    }
    edge_log_strain = []
    compressed_edges = 0
    for a, b in edges:
        rest = (base["points"][a] - base["points"][b]).length
        posed = (candidate["points"][a] - candidate["points"][b]).length
        if rest <= 1e-12:
            continue
        ratio = posed / rest
        edge_log_strain.append(abs(math.log(max(ratio, 1e-12))))
        compressed_edges += ratio < 0.5
    area_ratios = []
    for tri_index in roi_triangles:
        rest = triangle_area(base["points"], triangles[tri_index])
        posed = triangle_area(candidate["points"], triangles[tri_index])
        if rest > 1e-12:
            area_ratios.append(posed / rest)
    return {
        "roi_vertex_count": len(vertices),
        "roi_triangle_count": len(roi_triangles),
        "edge_count": len(edges),
        "edge_log_strain_p99": percentile(edge_log_strain, 0.99),
        "edge_log_strain_max": max(edge_log_strain, default=None),
        "compressed_edge_ratio_under_0_5": (
            compressed_edges / len(edges) if edges else None
        ),
        "triangle_area_ratio_p01": percentile(area_ratios, 0.01),
        "triangle_area_ratio_min": min(area_ratios, default=None),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args(argv)
    baseline = snapshot(args.baseline)
    candidate = snapshot(args.candidate)
    measurable = (
        baseline["profile"] == candidate["profile"]
        and baseline["triangles"] == candidate["triangles"]
        and len(baseline["points"]) == len(candidate["points"])
        and all(baseline["side_vertices"][s] == candidate["side_vertices"][s] for s in ("L", "R"))
        and all(baseline["side_vertices"][s] for s in ("L", "R"))
    )
    payload = {
        "baseline": str(Path(args.baseline).resolve()),
        "candidate": str(Path(args.candidate).resolve()),
        "measurable": measurable,
        "verdict": "MEASURED_NOT_GATED" if measurable else "UNMEASURABLE",
        "reason": None if measurable else "profile/topology/weight-derived ROI mismatch or empty ROI",
        "profile": baseline["profile"],
        "roi_group_names": baseline["side_groups"],
    }
    if measurable:
        roi_all = baseline["side_vertices"]["L"] | baseline["side_vertices"]["R"]
        outside = [
            (candidate["points"][index] - baseline["points"][index]).length
            for index in range(len(baseline["points"])) if index not in roi_all
        ]
        rest_pairs = self_intersections(baseline["points"], baseline["triangles"])
        posed_pairs = self_intersections(candidate["points"], candidate["triangles"])
        roi_triangles = {
            index for index, tri in enumerate(baseline["triangles"])
            if any(vertex in roi_all for vertex in tri)
        }
        new_roi_intersections = {
            pair for pair in posed_pairs - rest_pairs
            if pair[0] in roi_triangles and pair[1] in roi_triangles
        }
        payload.update({
            "outside_roi_max_displacement": max(outside, default=0.0),
            "new_nonadjacent_self_intersections_in_roi": len(new_roi_intersections),
            "baseline_self_intersections_all": len(rest_pairs),
            "candidate_self_intersections_all": len(posed_pairs),
            "sides": {
                side: side_metrics(baseline, candidate, side) for side in ("L", "R")
            },
        })
    output = Path(args.json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PALM_MESH_DELTA=" + json.dumps(payload, ensure_ascii=False))
    return 0 if measurable else 2


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    raise SystemExit(main(argv))
