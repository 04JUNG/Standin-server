"""Blender regression gate for pre-bent offsets plus nonzero finger channels.

Synthetic rigs only. Checks the delivered/reimported FBX, including mirroring,
offset-only sources, per-side preset precedence and unchanged body geometry.
"""
import json
import math
from pathlib import Path
import shutil
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from converter import retarget as rt
from converter.shape_bake import import_full_character
from converter.reviewed_source_fingers import bake_source_fingers, bone_names, _palm_frame
from converter.body_material import restore_anatomical_clay

FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
rt.import_character = import_full_character


def segment(degrees):
    a = math.radians(degrees)
    return Vector((0, math.cos(a), math.sin(a))) * .2


def build_target(path):
    rt.reset_scene()
    data = bpy.data.armatures.new('SyntheticHands')
    arm = bpy.data.objects.new('SyntheticHands', data)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    root = data.edit_bones.new('Hips')
    root.head, root.tail = (0, 0, 0), (0, .3, 0)
    for side, sign in [('Left', 1), ('Right', -1)]:
        wrist = data.edit_bones.new(side + 'Hand')
        wrist.head, wrist.tail = (sign, 0, 0), (sign, .4, 0)
        wrist.parent = root
        for i, finger in enumerate(FINGERS):
            head = Vector((sign + (2 - i) * .17, .5, 0))
            parent = wrist
            for j, angle in enumerate((0, 11, 13), 1):
                bone = data.edit_bones.new(f'{side}Hand{finger.title()}{j}')
                bone.head, bone.tail = head, head + segment(angle)
                bone.parent, bone.use_connect = parent, j > 1
                parent, head = bone, bone.tail.copy()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.mesh.primitive_cube_add(size=.2)
    body = bpy.context.object
    body.name = 'BodyControl'
    group = body.vertex_groups.new(name='Hips')
    group.add(list(range(len(body.data.vertices))), 1, 'REPLACE')
    body.modifiers.new(name='Armature', type='ARMATURE').object = arm
    mat = bpy.data.materials.new('Standin anatomical clay')
    mat.use_nodes = True
    body.data.materials.append(mat)
    rt.export_fbx(str(path), embed_textures=False)


def build_bvh(path, *, offset_only=False):
    lines = ['HIERARCHY', 'ROOT Hips', '{', 'OFFSET 0 0 0',
             'CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation']
    channels = [0] * 6
    for side, sign in [('Left', 1), ('Right', -1)]:
        lines += [f'JOINT {side}Hand', '{', f'OFFSET {sign} 0 0',
                  'CHANNELS 3 Zrotation Xrotation Yrotation']
        channels += [0, 0, 0]
        for i, finger in enumerate(FINGERS):
            for j in range(1, 4):
                # Rest bend is 77 degrees. Opening by 50 must yield 27,
                # instead of applying -50 to the almost-straight target.
                offset = Vector(((2 - i) * .17, .5, 0)) if j == 1 else segment((j - 2) * 77)
                lines += [f'JOINT {side}Hand{finger.title()}{j}', '{',
                          'OFFSET ' + ' '.join(str(v) for v in offset),
                          'CHANNELS 3 Zrotation Xrotation Yrotation']
                channels += [0, 0 if offset_only or j == 1 else -50, 0]
            lines += ['End Site', '{', 'OFFSET ' + ' '.join(str(v) for v in segment(154)), '}', '}', '}', '}']
        lines += ['}']
    lines += ['}', 'MOTION', 'Frames: 1', 'Frame Time: 0.033333333', ' '.join(map(str, channels))]
    path.write_text('\n'.join(lines) + '\n')


def snapshot(arm, meshes=()):
    result = {'hands': {}, 'body': {}}
    for side in ('left', 'right'):
        wrist, fingers = bone_names(arm.data.bones.keys(), side)
        frame = (rt._pose_world(arm, wrist).to_quaternion().to_matrix()
                 @ rt._rest_world(arm, wrist).to_quaternion().to_matrix().transposed()
                 @ _palm_frame(arm, wrist, fingers))
        values = {}
        for key, name in fingers.items():
            bone = arm.pose.bones[name]
            child = (key[0], key[1] + 1)
            tip = arm.pose.bones[fingers[child]].head if child in fingers else bone.tail
            values[key] = (frame.transposed() @ arm.matrix_world.to_3x3() @ (tip - bone.head)).normalized()
        result['hands'][side] = values
    for name in ('Hips', 'LeftHand', 'RightHand'):
        result['body'][name] = tuple(v for row in rt._pose_world(arm, name) for v in row)
    result['vertices'] = [tuple(m.matrix_world @ v.co) for m in meshes for v in m.data.vertices]
    return result


def load(path):
    rt.reset_scene()
    return snapshot(*import_full_character(str(path)))


def main():
    out = Path(sys.argv[sys.argv.index('--') + 1])
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'target.fbx'
    build_target(target)
    neutral = load(target)
    rows = []
    for offset_only in (False, True):
        bvh = out / ('offset.bvh' if offset_only else 'mixed.bvh')
        build_bvh(bvh, offset_only=offset_only)
        rt.reset_scene()
        expected = snapshot(rt.import_bvh(str(bvh), frame=0))
        for mirror, override, transfer in [(False, False, True), (True, False, True),
                                           (False, True, True), (False, False, False)]:
            path = out / 'result.fbx'
            shutil.copyfile(target, path)
            receipt = bake_source_fingers(path, bvh, mirror=mirror,
                hands={'left': {'id': 'explicit'}} if override else None,
                transfer_posed_offsets=transfer)
            actual = load(path)
            maximum = 0
            for side, fingers in actual['hands'].items():
                skipped = (override and side == 'left') or (offset_only and not transfer)
                source_side = ('right' if side == 'left' else 'left') if mirror else side
                for key, direction in fingers.items():
                    wanted = (neutral['hands'][side][key] if skipped else expected['hands'][source_side][key]).copy()
                    if mirror and not skipped:
                        wanted.z *= -1
                    maximum = max(maximum, math.degrees(direction.angle(wanted)))
            assert maximum < .1, (offset_only, mirror, override, transfer, maximum)
            for name, matrix in actual['body'].items():
                assert max(abs(a - b) for a, b in zip(matrix, neutral['body'][name])) < 1e-5
            assert len(actual['vertices']) == len(neutral['vertices'])
            assert max(abs(a - b) for va, vb in zip(actual['vertices'], neutral['vertices'])
                       for a, b in zip(va, vb)) < 1e-5
            assert restore_anatomical_clay(path)['changed_materials']
            assert not restore_anatomical_clay(path)['changed_materials']
            rows.append(dict(offset_only=offset_only, mirror=mirror, preset_override=override,
                             transfer_offsets=transfer, max_direction_error_deg=maximum, receipt=receipt))
    (out / 'report.json').write_text(json.dumps(rows, indent=2) + '\n')
    print('SOURCE_FINGERS_VERIFIED', len(rows))


if __name__ == '__main__':
    main()
