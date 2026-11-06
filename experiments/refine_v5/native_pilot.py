"""Blender-only diagnostic conversion/application. No skin safety certificate."""
import bpy, sys, json, hashlib
from pathlib import Path
from mathutils import Vector
ACTIVE = ['mixamorig:LeftArm', 'mixamorig:LeftForeArm', 'mixamorig:RightArm', 'mixamorig:RightForeArm']

def find(arm, suffix):
    matches = [b for b in arm.pose.bones if b.name.split(':')[-1] == suffix.split(':')[-1]]
    if len(matches) != 1:
        raise ValueError('ambiguous/missing bone ' + suffix)
    return matches[0]

def snapshot(arm):
    rows = {}
    for name in ACTIVE:
        b = find(arm, name)
        h = arm.matrix_world @ b.head
        t = arm.matrix_world @ b.tail
        rows[name] = {'head': list(h), 'tail': list(t), 'direction': list((t - h).normalized()), 'length': (t - h).length}
    hips = find(arm, 'Hips')
    spine = find(arm, 'Spine2')
    return {'arms': rows, 'center': list(arm.matrix_world @ hips.head), 'scale': (arm.matrix_world @ (spine.head - hips.head)).length, 'mesh_checked': False}

def load(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(path))
    arm = next((o for o in bpy.context.scene.objects if o.type == 'ARMATURE'))
    return (arm, [o for o in bpy.context.scene.objects if o.type == 'MESH'])

def main():
    a = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text())
    sys.path.insert(0, a['repo'])
    from converter.convert import convert
    from converter import retarget as rt
    root = Path(a['out'])
    root.mkdir(exist_ok=False, parents=True)
    records = []
    for row in a['cases']:
        case = row['case_id']
        if a['mode'] == 'prepare':
            base = root / (case + '.base.fbx')
            r = convert(bvh_path=row['bvh'], character_fbx=a['body'], out_path=str(base), output_mode='rigged_rest', embed_textures=False)
            if not r.ok:
                raise ValueError('conversion rejected ' + case)
            arm, _ = load(base)
            state = snapshot(arm)
            records.append(dict(case_id=case, base_fbx=str(base), state=state, seed_kind=row['seed_kind'], image=row['image']))
        else:
            arm, meshes = load(row['base_fbx'])
            before = snapshot(arm)
            for name, direction in zip(ACTIVE, row['prediction']):
                b = find(arm, name)
                cur = (arm.matrix_world @ (b.tail - b.head)).normalized()
                target = Vector(direction).normalized()
                delta = cur.rotation_difference(target)
                world = arm.matrix_world @ b.matrix
                position = world.translation.copy()
                world = delta.to_matrix().to_4x4() @ world
                world.translation = position
                b.matrix = arm.matrix_world.inverted() @ world
                bpy.context.view_layer.update()
            planned = snapshot(arm)
            dest = root / (case + '.candidate-not-for-service.fbx')
            rt.apply_output_mode(arm, meshes, 'rigged_rest')
            rt.export_fbx(str(dest), embed_textures=False, bake_anim=False)
            arm, _ = load(dest)
            actual = snapshot(arm)
            records.append(dict(case_id=case, before=before, planned=planned, actual=actual, artifact=str(dest), sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), production_ready=False, mesh_checked=False))
    (root / 'records.json').write_text(json.dumps(records, indent=2))
    print('V5_NATIVE', len(records))
if __name__ == '__main__':
    main()
