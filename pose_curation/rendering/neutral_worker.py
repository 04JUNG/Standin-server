"""Reference head/bust from the native character, with no retargeted neck pose."""
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bpy
from converter import retarget
from converter.framing import crop_meshes
from pose_curation.rendering.scene import render_views
from pose_curation.storage import read_json, sha256, write_json


def run(job):
    if bpy.app.version[:3] != (5, 2, 0) or job['scope'] not in {'head', 'bust'}:
        raise ValueError('Reference head/bust requires Blender 5.2.0')
    source, directory = Path(job['character']), Path(job['directory'])
    if sha256(source) != job['character_sha256']:
        raise ValueError('기본 캐릭터가 변경되었습니다.')
    retarget.reset_scene()
    arm, meshes = retarget.import_character(str(source))
    if arm.animation_data:
        arm.animation_data_clear()
    for bone in arm.pose.bones:
        bone.matrix_basis.identity()
    bpy.context.view_layer.update()
    crop = crop_meshes(arm, meshes, job['scope'])
    output = directory / 'character.fbx'
    retarget.export_fbx(str(output), embed_textures=False)
    retarget.reset_scene()
    retarget.import_character(str(output))
    images = render_views(directory, 'preview', resolution=512)
    if sha256(source) != job['character_sha256']:
        raise ValueError('생성 중 기본 캐릭터가 변경되었습니다.')
    return {'ok': True, 'fingerprint': job['fingerprint'], 'crop': crop,
            'source_kind': 'native_reference', 'fbx_sha256': sha256(output),
            'thumbnails': {view: {'file': path.name, 'sha256': sha256(path)} for view, path in images.items()}}


if __name__ == '__main__':
    job = read_json(Path(sys.argv[sys.argv.index('--') + 1]))
    try:
        result = run(job)
    except Exception as exc:
        traceback.print_exc()
        result = {'ok': False, 'error': str(exc)}
    write_json(Path(job['directory']) / 'result.json', result)
