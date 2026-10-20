"""Rotate a verified cropped FBX, reimport it, then render the delivered artifact."""
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bpy
import numpy as np
from mathutils import Matrix
from converter import retarget
from converter.bone_map import PROFILES, resolve_profile
from pose_curation.orientation import Orientation
from pose_curation.rendering.scene import render_views
from pose_curation.storage import read_json, sha256, write_json


def landmarks(arm):
    return {b.name: np.array(arm.matrix_world @ b.head) for b in arm.pose.bones}


def run(job):
    if bpy.app.version[:3] != (5, 2, 0):
        raise ValueError('Blender 5.2.0 required')
    if sha256(Path(job['base_fbx'])) != job['base_fbx_sha256']:
        raise ValueError('기준 FBX가 변경되었습니다.')
    orientation = Orientation(**job['angles'])
    directory = Path(job['directory'])
    retarget.reset_scene()
    arm, meshes = retarget.import_character(job['base_fbx'])
    before = landmarks(arm)
    profile = resolve_profile(list(before))
    pivot = arm.matrix_world @ arm.pose.bones[PROFILES[profile]['hips']].head
    rotation = Matrix(orientation.blender_matrix().tolist()).to_4x4()
    transform = Matrix.Translation(pivot) @ rotation @ Matrix.Translation(-pivot)
    # Transform top-level objects once; child meshes inherit their armature transform.
    for obj in list(bpy.context.scene.objects):
        if obj.parent is None:
            obj.matrix_world = transform @ obj.matrix_world
    bpy.context.view_layer.update()
    expected = landmarks(arm)
    output = directory / 'oriented.fbx'
    retarget.export_fbx(str(output), embed_textures=False)
    retarget.reset_scene()
    arm, _ = retarget.import_character(str(output))
    after = landmarks(arm)
    if set(after) != set(before):
        raise ValueError('방향 출력에서 뼈대 이름이 변경되었습니다.')
    error = max(float(np.max(np.abs(after[k] - expected[k]))) for k in expected)
    extent = max(float(np.ptp(np.stack(list(expected.values())), axis=0).max()), 1)
    if error > extent * 1e-5:
        raise ValueError(f'FBX 재가져오기 위치 불일치: {error}')
    images = render_views(directory, 'oriented', resolution=512)
    metadata = {**job['metadata'], 'fbx_validation': {'bones': len(after),
                'max_position_error': error}, 'reference_camera': {'projection': 'orthographic',
                'view': 'front', 'auto_fit': True},
                'csp_import': 'Use a front parallel-projection camera. Existing layer camera is not changed by the file.'}
    write_json(directory / 'orientation.json', metadata)
    files = {'preview': images['front'], 'fbx': output,
             'settings': directory / 'orientation.json'}
    return {'ok': True, 'kind': 'orientation', 'fingerprint': job['fingerprint'],
            'orientation': orientation.public(), 'validation': metadata['fbx_validation'],
            'files': {name: {'file': path.name, 'sha256': sha256(path)} for name, path in files.items()}}


if __name__ == '__main__':
    job = read_json(Path(sys.argv[sys.argv.index('--') + 1]))
    try:
        result = run(job)
    except Exception as exc:
        traceback.print_exc()
        result = {'ok': False, 'error': str(exc)}
    write_json(Path(job['directory']) / 'result.json', result)
