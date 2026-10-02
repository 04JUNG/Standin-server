"""Render an explicit diagnostic job without modifying the review manifests."""
import json
from pathlib import Path
import sys
import tempfile
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from converter import retarget
from converter.convert import convert
from pose_curation.rendering.profiles import register
from pose_curation.rendering.fingers import rest_bases, apply_channels
from pose_curation.rendering.anatomy import rest_frames, inspect
from pose_curation.rendering.scene import render_views
from pose_curation.storage import write_json, sha256


def main(job):
    register()
    bases = rest_bases(job['character'])
    # rest_bases leaves the imported unposed character in the scene.
    arm = retarget._armatures()[0]
    frames = rest_frames(arm)
    output = Path(job['output'])
    for pose in job['poses']:
        result = {'pose_id': pose['pose_id'], 'bvh_sha256': sha256(Path(pose['bvh']))}
        try:
            with tempfile.TemporaryDirectory(prefix='pose-anatomy-') as temp:
                fbx = str(Path(temp) / 'converted.fbx')
                report = convert(bvh_path=pose['bvh'], character_fbx=job['character'], out_path=fbx,
                                 src_profile=pose.get('rig_profile', 'mixamo_noprefix'), dst_profile='mixamo', embed_textures=False)
                if not report.ok:
                    raise ValueError('body conversion failed')
                retarget.reset_scene()
                arm, _ = retarget.import_character(fbx)
                result['fingers'] = apply_channels(arm, pose['bvh'], bases)
                result['anatomy'] = inspect(arm, frames)
                if job.get('render', True) or (job.get('render_flagged') and result['anatomy']['flags']):
                    result['images'] = {k: str(v) for k, v in render_views(output, pose['pose_id'], resolution=job.get('resolution', 640)).items()}
                result['ok'] = True
        except Exception as exc:
            result.update(ok=False, error=str(exc))
            traceback.print_exc()
        write_json(output / (pose['pose_id'] + '.json'), result)
        print('ANATOMY', pose['pose_id'], result.get('anatomy', result), flush=True)


if __name__ == '__main__':
    main(json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf-8')))
