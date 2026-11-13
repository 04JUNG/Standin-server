"""Separate registration, LBS/retarget, arm, surface and FBX reimport evidence.

Run inside the pinned Blender. All original assets, stage arrays and diagnostic
FBXs remain in the local run directory. The JSON report contains hashes and
small numeric witnesses; it is not a CSP acceptance or overall quality pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def world_normals(obj):
    normal = np.array([n.vector[:] for n in obj.data.corner_normals])
    normal = normal @ np.linalg.inv(np.asarray(obj.matrix_world)[:3, :3])
    return normal / np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-30)


def angle_error(a, b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), (a*b).sum(1)))


def weights(obj):
    return [sorted((obj.vertex_groups[g.group].name, g.weight) for g in v.groups)
            for v in obj.data.vertices]


def frames(arm, posed):
    return {b.name: np.array(arm.matrix_world @ (b.matrix if posed else b.matrix_local))
            for b in (arm.pose.bones if posed else arm.data.bones)}


def independent_lbs(rest, influences, before, after):
    transform = {name: after[name] @ np.linalg.inv(value) for name, value in before.items()}
    predicted = np.zeros_like(rest)
    total = np.zeros(len(rest))
    by_bone = {}
    for index, row in enumerate(influences):
        for name, amount in row:
            by_bone.setdefault(name, []).append((index, amount))
    homogeneous = np.column_stack((rest, np.ones(len(rest))))
    for name, rows in by_bone.items():
        ids, amount = np.array(rows).T
        ids = ids.astype(int)
        predicted[ids] += (homogeneous[ids] @ transform[name].T)[:, :3] * amount[:, None]
        total[ids] += amount
    predicted /= total[:, None]
    return predicted, transform


def verify(runtime, directory, poses, baseline=None):
    sys.path.insert(0, str(runtime))
    from converter import retarget as rt
    from converter.local import inspection, surface, arm_transfer
    from converter.local.contract import versions
    run = json.loads((directory / "report.json").read_text())
    assert run["allPassed"]
    assert sha(runtime / "runtime-manifest.json") == run["packageManifestSha256"]
    result = {"versions": versions(), "runtimeManifestSha256": run["packageManifestSha256"],
              "cspImport": "unverified", "overallDeformationQuality": "not qualified", "models": []}
    for mi, model in enumerate(run["models"]):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        arm, meshes = inspection.import_fbx(directory / f"model-{mi}-registered.fbx")
        assert len(meshes) == 1, "QA adapter requires the approved single-mesh fixtures"
        obj = meshes[0]
        registered_normals = world_normals(obj)
        neutral = np.array(inspection.surface(meshes)[obj.name])
        influences = weights(obj)
        hierarchy = {b.name: b.parent.name if b.parent else None for b in arm.data.bones}
        polygons = [tuple(p.vertices) for p in obj.data.polygons]
        source_check = {"sourceSha256": model["sourceHash"]}
        if model["inputName"].endswith(".cs3c"):
            from converter.local.cs3c import character_data
            source = directory / "store/models" / f"actual-model-{mi}" / "original.cs3c"
            assert sha(source) == model["sourceHash"]
            decoded = character_data(source)
            assert len(decoded["meshes"]) == 1
            raw = decoded["meshes"][0]
            basis = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
            ids = np.array([loop.vertex_index for loop in obj.data.loops])
            source_normal = raw["normals"] @ basis.T
            source_normal /= np.linalg.norm(source_normal, axis=1)[:, None]
            angles = angle_error(source_normal[ids], registered_normals)
            pos = np.linalg.norm(raw["positions"] @ basis.T * .01 - neutral, axis=1)
            assert pos.max() < 2e-5 and angles.max() < .1
            source_check.update({"decodedToRegisteredMaxPositionErrorM": float(pos.max()),
                "decodedToRegisteredNormalErrorP50P95MaxDeg": np.percentile(angles, [50, 95, 100]).tolist()})
        records = []
        for pi, pose in enumerate(poses):
            assert sha(pose) == model["poses"][pi]["poseHash"]
            bpy.ops.wm.read_factory_settings(use_empty=True)
            arm, meshes = inspection.import_fbx(directory / f"model-{mi}-registered.fbx")
            obj = meshes[0]
            character, _ = inspection.inspect(arm, meshes)
            reference = surface.prepare_reference(obj.data)
            neutral_by_mesh = inspection.surface(meshes)
            rest = np.asarray(neutral_by_mesh[obj.name])
            rt._apply_object_transform(arm)
            rest_frames = frames(arm, False)
            src = rt.import_bvh(str(pose), frame=0)
            report = rt.retarget(src, arm, dst_profile=character["profile"], apply_root_translation=False)
            assert report.ok
            solver = np.asarray(inspection.surface(meshes)[obj.name])
            predicted, transform = independent_lbs(rest, influences, rest_frames, frames(arm, True))
            lbs_error = float(np.linalg.norm(predicted - solver, axis=1).max())
            assert lbs_error < 2e-5
            arm_report = arm_transfer.apply(src, arm, meshes, neutral_by_mesh, report)
            after_arm = np.asarray(inspection.surface(meshes)[obj.name])
            bpy.data.objects.remove(src, do_unlink=True)
            rt.apply_output_mode(arm, meshes, "rigged_rest")
            baked = np.asarray(inspection.surface(meshes)[obj.name])
            assert np.linalg.norm(after_arm - baked, axis=1).max() < 2e-5
            pre_surface_normals = world_normals(obj)
            # Keep a real material diagnostic FBX for visual review of the
            # correction. Do not replace the packaged host's final export.
            stage_path = directory / f"stage-{mi}-{pi}-before-surface.fbx"
            rt.export_fbx(str(stage_path), embed_textures=False)
            surface_report = surface.apply_to_meshes(meshes, neutral_by_mesh, {obj.name: reference})
            corrected = np.asarray(inspection.surface(meshes)[obj.name])
            post_surface_normals = world_normals(obj)
            np.savez_compressed(directory / f"stages-{mi}-{pi}.npz", rest=rest, solver=solver,
                arm=after_arm, baked=baked, corrected=corrected,
                preSurfaceNormals=pre_surface_normals, correctedNormals=post_surface_normals,
                triangles=reference.triangles, anchors=reference.anchors)
            faces = reference.triangles
            graph = surface.Neighborhood(faces, rest)
            rest_lengths = surface.lengths(rest[graph.b] - rest[graph.a])
            rest_area, normals = surface.triangle_geometry(rest, faces)
            i, j = graph.face_pairs.T
            alignment = (normals[i]*normals[j]).sum(1)
            smooth = alignment > np.cos(np.pi/6)
            graph.fixed |= reference.anchors
            graph.fixed[faces[graph.face_pairs[alignment < np.cos(np.pi/3)]].ravel()] = True
            support = surface.damage_support(graph, rest, baked, faces, rest_lengths, rest_area, alignment)
            outside = support == 0
            outside_delta = float(np.linalg.norm(corrected[outside] - baked[outside], axis=1).max())
            assert outside_delta == 0.
            loops = np.array([loop.vertex_index for loop in obj.data.loops])
            outside_normal_error = float(angle_error(pre_surface_normals, post_surface_normals)[outside[loops]].max())
            stage_metrics = {key: surface.metrics(value, faces, graph, rest_lengths, rest_area, smooth)
                             for key, value in [("solver", solver), ("arm", after_arm), ("surface", corrected)]}
            ratios = surface.lengths(solver[graph.b] - solver[graph.a]) / rest_lengths
            k = int(np.argmax(ratios))
            va, vb = int(graph.a[k]), int(graph.b[k])
            wa, wb = dict(influences[va]), dict(influences[vb])
            uniform, gradient = np.zeros(3), np.zeros(3)
            for name in wa.keys() | wb.keys():
                w0, w1 = wa.get(name, 0), wb.get(name, 0)
                uniform += .5*(w0+w1) * (transform[name][:3, :3] @ (rest[vb]-rest[va]))
                gradient += (w1-w0) * (transform[name] @ np.r_[.5*(rest[va]+rest[vb]), 1])[:3]
            witness = {"vertices": [va, vb], "restLengthM": float(rest_lengths[k]),
                "solverLengthM": float(np.linalg.norm(solver[vb]-solver[va])),
                "correctedLengthM": float(np.linalg.norm(corrected[vb]-corrected[va])),
                "averageWeightEdgeTermM": float(np.linalg.norm(uniform)),
                "weightGradientTermM": float(np.linalg.norm(gradient)),
                "decompositionResidualM": float(np.linalg.norm(uniform+gradient - (solver[vb]-solver[va])))}
            final_path = directory / model["poses"][pi]["file"]
            bpy.ops.wm.read_factory_settings(use_empty=True)
            arm, meshes = inspection.import_fbx(final_path)
            exported = np.asarray(inspection.surface(meshes)[meshes[0].name])
            delta = float(np.linalg.norm(exported-corrected, axis=1).max())
            normal_delta = float(angle_error(post_surface_normals, world_normals(meshes[0])).max())
            assert delta < 2e-5
            assert normal_delta < .2
            assert weights(meshes[0]) == influences
            assert [tuple(p.vertices) for p in meshes[0].data.polygons] == polygons
            assert all(arm.data.bones.get(name) is not None and
                       (arm.data.bones[name].parent.name if arm.data.bones[name].parent else None) == parent
                       for name, parent in hierarchy.items())
            row = {"poseSha256": sha(pose), "exportSha256": sha(final_path),
                "independentLbsMaxErrorM": lbs_error, "armTransfer": arm_report,
                "surfaceCorrection": surface_report, "stageMetrics": stage_metrics,
                "solverWorstEdgeWitness": witness, "outsideSupportVertices": int(outside.sum()),
                "outsideSupportMaxPositionDeltaM": outside_delta,
                "outsideSupportMaxNormalDeltaDeg": outside_normal_error,
                "exportToRegeneratedSurfaceMaxErrorM": delta,
                "exportToRegeneratedWorldNormalMaxErrorDeg": normal_delta,
                "exactPolygonsWeightsOriginalHierarchy": True}
            if baseline and (baseline / f"model-{mi}-pose-{pi}.fbx").is_file():
                old_path = baseline / f"model-{mi}-pose-{pi}.fbx"
                bpy.ops.wm.read_factory_settings(use_empty=True)
                old_arm, old_meshes = inspection.import_fbx(old_path)
                old = np.asarray(inspection.surface(old_meshes)[old_meshes[0].name])
                row["baseline"] = {"sha256": sha(old_path),
                    "metrics": surface.metrics(old, faces, graph, rest_lengths, rest_area, smooth),
                    "maxPositionDeltaM": float(np.linalg.norm(exported-old, axis=1).max())}
            records.append(row)
            print("FOLLOWUP_GEOMETRY", mi, pi, delta, flush=True)
        result["models"].append({"source": model["inputName"], "registration": source_check, "poses": records})
        (directory / "geometry-followup.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pose", type=Path, action="append", required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    verify(args.runtime.resolve(), args.run_dir.resolve(), [p.resolve() for p in args.pose], args.baseline)
