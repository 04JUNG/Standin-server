"""Private diagnostic: couple proximal frame transport with bounded surface repair.

The parent frame rotates about its original joint head while every descendant
world matrix is restored. Authored weights, hand frames and bone heads remain
unchanged. This is deliberately outside the packaged converter and never picks
an export for the product. Run with pinned Blender and permitted input paths.
"""
import argparse, json, sys, hashlib
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from converter import retarget as rt
from converter.local import inspection, surface, arm_transfer
from converter.local.pose_quality import measure


def run(registered, pose, output):
    output.mkdir(exist_ok=False, parents=True)
    # Separate parent-only, child-only and coupled parent-frame candidates.
    trials = [('baseline', [])]
    for role in ['hips', 'spine']:
        for axis in ['X', 'Y', 'Z']:
            for degrees in [-3, 3]: trials.append((f'{role}-{axis}-{degrees}', [(role, axis, degrees)]))
    trials += [('hips-spine-counter', [('hips','X',3),('spine','X',-3)]),
               ('hips-spine-counter-reverse', [('hips','X',-3),('spine','X',3)])]
    rows = []
    baseline = {}
    for label, rotations in trials:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        arm, meshes = inspection.import_fbx(registered)
        character, _ = inspection.inspect(arm, meshes)
        neutral = inspection.surface(meshes)
        references = {m.name: surface.prepare_reference(m.data) for m in meshes}
        rt._apply_object_transform(arm)
        src = rt.import_bvh(str(pose), frame=0)
        report = rt.retarget(src, arm, dst_profile=character['profile'], apply_root_translation=False)
        assert report.ok
        arm_transfer.apply(src, arm, meshes, neutral, report)
        bpy.data.objects.remove(src, do_unlink=True)
        matrices = {p.name: p.matrix.copy() for p in arm.pose.bones}
        heads = {p.name: p.head.copy() for p in arm.pose.bones}
        bones = sorted(arm.pose.bones, key=lambda p: len(p.parent_recursive))
        targets = {name:m.copy() for name,m in matrices.items()}
        for role, axis, degrees in rotations:
            name = character['mapping'][role]
            head = heads[name]
            rotation = Matrix.Rotation(np.radians(degrees),4,axis)
            targets[name] = Matrix.Translation(head) @ rotation @ Matrix.Translation(-head) @ targets[name]
        if rotations:
            for p in bones: p.matrix = targets[p.name]; bpy.context.view_layer.update()
        max_head = max((p.head-heads[p.name]).length for p in bones)
        # Endpoint restoration is measured, not assumed from matrix assignment.
        if max_head > 2e-5:
            rows.append({'trial':label,'rejected':'joint_head_drift','maxHeadDriftM':max_head}); continue
        posed = inspection.surface(meshes)
        rt.apply_output_mode(arm, meshes, "rigged_rest")
        repair_report = surface.apply_to_meshes(meshes, neutral, references)
        final = inspection.surface(meshes)
        metrics = []
        for obj in meshes:
            ref = references[obj.name]
            rest = np.asarray(neutral[obj.name]); raw = np.asarray(posed[obj.name])
            corrected = np.asarray(final[obj.name])
            repair = repair_report["meshes"][len(metrics)]
            graph = surface.Neighborhood(ref.triangles,rest)
            lengths = surface.lengths(rest[graph.b]-rest[graph.a])
            area,normals=surface.triangle_geometry(rest,ref.triangles)
            i,j=graph.face_pairs.T;smooth=(normals[i]*normals[j]).sum(1)>np.cos(np.pi/6)
            shape=surface.metrics(corrected,ref.triangles,graph,lengths,area,smooth)
            row={'raw':measure(rest,raw,ref.triangles),'corrected':measure(rest,corrected,ref.triangles),
                 'shape':shape,'surfaceApplied':repair['applied']}
            if label=='baseline':baseline[obj.name]=shape
            row['metricRegressions']=[k for k,v in shape.items() if v>baseline[obj.name][k]+1e-6]
            metrics.append(row)
            np.savez_compressed(output/(label+'-'+str(len(metrics))+'.npz'),rest=rest,posed=raw,corrected=corrected,triangles=ref.triangles)
        rows.append({'trial':label,'rotations':rotations,'maxHeadDriftM':max_head,'meshes':metrics,
                     'adopted':False,'qualification':'diagnostic only; contact/export/CSP gate not reached'})
        (output/'report.json').write_text(json.dumps({'registeredSha256':hashlib.sha256(registered.read_bytes()).hexdigest(),
            'poseSha256':hashlib.sha256(pose.read_bytes()).hexdigest(),'trials':rows,'productChanged':False},indent=2))
        print('PARENT_TRANSPORT',label,[m['corrected']['maxStretch'] for m in metrics],flush=True)
    (output/"report.json").write_text(json.dumps({"registeredSha256":hashlib.sha256(registered.read_bytes()).hexdigest(), "poseSha256":hashlib.sha256(pose.read_bytes()).hexdigest(), "trials":rows,"productChanged":False},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registered',type=Path,required=True);p.add_argument('--pose',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);run(a.registered,a.pose,a.output)
