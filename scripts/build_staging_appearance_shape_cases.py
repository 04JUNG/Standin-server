"""Export pinned authoring sources with actual slider changes and bent joints (Blender)."""
import hashlib
import json
import os
from pathlib import Path
import sys

import bpy
from mathutils import Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from converter.shape_bake import _load
from converter.shape_spec import resolve_shape, digest_shape


def main():
    root = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
    models = root / 'models'
    registry = json.loads((models / 'shape-bakes.json').read_bytes())
    for row in registry['sources'].values():
        for key in ('blend', 'controller'):
            row[key] = str(models / 'sources' / Path(row[key]).name)
    local = root / 'shape-registry.local.json'
    local.write_text(json.dumps(registry))
    os.environ['CONVERTER_SHAPE_REGISTRY'] = str(local)
    cases = []
    for gender, cid in [('male', 'male-base-v0-8-0'), ('female', 'female-base-v0-4-0')]:
        row = registry['characters'][cid]
        controls = {c['key']: c['default'] for c in row['controls']}
        changes = dict(height=190 if gender == 'male' else 162, shoulders=.55, waist=-.3, muscle=.4, head_scale=1.1)
        specs = {c['key']: c for c in row['controls']}
        for key, value in changes.items():
            if key in specs:
                spec = specs[key]
                value = min(spec['max'], max(spec['min'], value))
                controls[key] = round(spec['min'] + round((value-spec['min'])/spec['step'])*spec['step'], 6)
        raw = dict(schemaVersion='body-shape.v1', controls=controls,
                   shapeSha256=digest_shape(cid, row['asset_sha256'], controls))
        recipe = resolve_shape(cid, row['asset_sha256'], raw)
        module = _load(recipe)
        mesh, rig = module.objects()
        for side, sign in [('Left', 1), ('Right', -1)]:
            for name, axis, angle in [('Arm', (0, 1, 0), sign*55), ('ForeArm', (0, 0, 1), -sign*70),
                                      ('UpLeg', (1, 0, 0), -45), ('Leg', (1, 0, 0), 65)]:
                bone = rig.pose.bones['mixamorig:' + side + name]
                local_axis = bone.bone.matrix_local.to_3x3().inverted() @ Vector(axis)
                bone.rotation_mode = 'QUATERNION'
                bone.rotation_quaternion = Quaternion(local_axis, __import__('math').radians(angle))
        bpy.context.view_layer.update()
        module.correct_pose(check_intersections=True, contacts=True)
        out = root / 'shape-cases' / gender
        out.mkdir(parents=True, exist_ok=True)
        exported = module.export_fixed(str(out), posed=True)
        source = Path(exported) / 'model.fbx'
        assert source.is_file()
        case = dict(name=gender+'-custom-shaped-bent', characterId=cid, source=str(source),
                    statureM=controls['height']/100, width=1.2, shape=raw,
                    shapeSource={k: recipe[k] for k in ('rules', 'blend_sha256', 'controller_sha256')},
                    sourceSha256=hashlib.sha256(source.read_bytes()).hexdigest())
        cases.append(case)
        (root / 'shape-cases.json').write_text(json.dumps(cases, indent=2) + '\n')
        print('SHAPED', cid, raw['shapeSha256'], flush=True)


if __name__ == '__main__':
    main()
