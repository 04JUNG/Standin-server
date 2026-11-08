"""Prepare one source FBX and validate the two separately run pilot inputs."""

import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.refine_v5.ab_blender import apply_rotations, bone_world, sha, verify_plan, weighted_mesh
from experiments.refine_v5.ab_native_audit import RoundtripError, export_checked, frozen_contacts, measure_contact, snapshot, strict_skin, triage, write
from experiments.refine_v5.audit_iterative_native import load
from experiments.refine_v5.target_safety import skin_nonregression
from experiments.refine_v5.part_quality import segment_cos

def prepare(root, prior, review):
    source_row = next(x for x in json.loads((review/'comparison.json').read_text()) if x['case_id']=='029')
    source = (review/source_row['fbx']).resolve()
    if sha(source) != source_row['sha256']:
        raise ValueError('reviewed FBX changed')
    case = next(c for c in json.loads((prior/'manifest.json').read_text())['cases'] if c['case_id']=='029')
    arm, meshes = load(source)
    by = bone_world(arm)
    names = list(by)
    verts, faces, weights = weighted_mesh(meshes, names)
    case['bones'] = [dict(b, matrix=by[b['name']].tolist()) for b in case['bones']]
    case.update(source=str(source), source_sha256=source_row['sha256'])
    root.mkdir(parents=True, exist_ok=False)
    write(root/'input.json', case)
    np.savez_compressed(root/'base.npz', vertices=verts, faces=faces, weights=weights, names=names)


def export(root, mode):
    output = root/mode
    if (output/'audit.json').exists():
        raise FileExistsError('FBX outcome is already frozen')
    case = json.loads((root/'input.json').read_text())
    filename = f"{case['case_id']}-part-goals-head-limbs.fbx"
    source = Path(case['source'])
    bound = json.loads((output/'plan.json').read_text())
    if sha(root/'input.json') != bound['input_sha256'] or sha(source) != case['source_sha256']:
        raise ValueError('single-goal inputs changed')
    plan = bound['result']
    arm, meshes = load(source)
    world_by_name = bone_world(arm)
    names = list(world_by_name)
    world = np.array(list(world_by_name.values()))
    verts, faces, weights = weighted_mesh(meshes, names)
    if plan is None:
        shutil.copyfile(source, output/filename)
        np.savez_compressed(output/'final.npz', vertices=verts, faces=faces)
        write(output/'audit.json', dict(accepted=False, reason='no_supported_goal', source_sha256=sha(source),
             output_sha256=sha(source), bones=[dict(name=n,matrix=m.tolist()) for n,m in world_by_name.items()]))
        return
    if plan['source_sha256'] != case['source_sha256']:
        raise ValueError('plan source differs')
    active = set(plan['edits_deg'])
    permitted = {'RightUpLeg':110.,'RightLeg':65.,'RightFoot':72.,'LeftArm':54.,'LeftForeArm':53.,'RightArm':54.,'RightForeArm':53.}
    if not active or active - permitted.keys():
        raise ValueError('unexpected edited joints')
    if any(np.asarray(r).shape != (3,) or not np.isfinite(r).all() or np.linalg.norm(r) > permitted[n]+1e-5
           for n,r in plan['edits_deg'].items()):
        raise ValueError('edit outside the pilot profile')
    measure_waist = plan['goal'].get('left_arm', {}).get('clearance_m') is not None
    frozen_patches = frozen_contacts(verts, faces, weights, names, world, [('Left','waist')]) if measure_waist else []

    def left_waist(v, f):
        if not frozen_patches:
            return None
        value = measure_contact(v, f, frozen_patches)[0]
        hand, _, waist = frozen_patches[0][2]
        tree = lambda idx: BVHTree.FromPolygons(v.tolist(), f[idx].tolist(), all_triangles=True)
        return dict(value, triangle_crossings=len(tree(hand).overlap(tree(waist))))

    def bends(bones):
        groups = dict(right_knee=('RightUpLeg','RightLeg','RightFoot'), left_elbow=('LeftArm','LeftForeArm','LeftHand'))
        return {name: float(np.degrees(np.arccos(segment_cos([np.asarray(bones[b])[:3,3] for b in chain]))))
                for name,chain in groups.items()}

    before = snapshot(arm)
    old_skin = triage(arm)
    old_waist = left_waist(verts, faces)
    parents = before['parents']
    descendants = set(active)
    for bone in case['bones']:
        if parents[bone['name']] in descendants:
            descendants.add(bone['name'])
    protected = set(names)-descendants
    apply_rotations(arm, plan['edits_deg'])
    verify_plan(arm, plan['expected_heads'])
    posed = bone_world(arm)
    if any(not np.allclose(posed[n], before['matrices'][n], atol=1e-5, rtol=0) for n in protected):
        raise ValueError('protected pose changed')
    with TemporaryDirectory(dir=root) as temp:
        stage_file = Path(temp)/'candidate.fbx'
        try:
            arm, v, f, _, cloud, frame = export_checked(stage_file, arm, meshes, names, before, load)
        except RoundtripError as exc:
            shutil.copyfile(source, output/filename)
            np.savez_compressed(output/'final.npz', vertices=verts, faces=faces)
            write(output/'audit.json', dict(accepted=False, reason=str(exc), source_sha256=sha(source),
                  output_sha256=sha(source), bones=[dict(name=n,matrix=m.tolist()) for n,m in world_by_name.items()]))
            return
        actual = bone_world(arm)
        drift = max(np.linalg.norm(actual[n][:3,3]-np.asarray(before['matrices'][n])[:3,3]) for n in protected)
        if drift >= 1e-5:
            raise ValueError('protected pose moved on reload')
        new_skin = triage(arm)
        new_waist = left_waist(v, f)
        safety = (['skin_triage_unavailable'] if 'unavailable' in old_skin or 'unavailable' in new_skin
                  else skin_nonregression(old_skin, new_skin))
        accepted = not safety and (not measure_waist or new_waist['triangle_crossings'] <= old_waist['triangle_crossings'])
        if not accepted:
            print('REGIONAL_REJECTED', safety, new_waist, flush=True)
        if accepted:
            stage_file.replace(output/filename)
        else:
            shutil.copyfile(source, output/filename)
    np.savez_compressed(output/'final.npz', vertices=v if accepted else verts, faces=f if accepted else faces)
    saved_v, saved_f = (v, f) if accepted else (verts, faces)
    full_skin = strict_skin(saved_v, saved_f, case['body_asset_sha256'], full_body=True)
    write(output/'audit.json', dict(accepted=accepted, protected_joint_drift_m=float(drift), cloud_m=cloud, frame_error=frame,
        new_or_worsened_skin=safety, before_skin=old_skin.get('collisions', []), after_skin=(new_skin if accepted else old_skin).get('collisions', []),
        full_skin_status=full_skin['status'], full_skin_certified=full_skin['complete_collision_certificate'],
        before_waist=old_waist, after_waist=new_waist if accepted else old_waist, before_bends=bends(before['matrices']), after_bends=bends(actual if accepted else before['matrices']),
        bones=[dict(name=n, matrix=np.asarray(m).tolist()) for n,m in (actual if accepted else before['matrices']).items()],
        source_sha256=sha(source), output_sha256=sha(output/filename) if accepted else sha(source)))
    print('SINGLE_NATIVE', case['case_id'], accepted, drift, safety, new_waist, bends(actual), flush=True)


if __name__ == '__main__':
    stage, root_arg, *rest = sys.argv[sys.argv.index('--') + 1:]
    root = Path(root_arg)
    if stage == 'prepare':
        prepare(root, *map(Path, rest))
    elif stage == 'export' and rest in (['reviewed'], ['detector']):
        export(root, rest[0])
    else:
        raise ValueError('expected prepare or export with a pilot mode')
