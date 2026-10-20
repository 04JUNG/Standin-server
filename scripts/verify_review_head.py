"""Generate a small, repeatable neutral head/bust FBX handoff pack."""
from pathlib import Path
import shutil
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from pose_curation.review.framing import FramedPreviews, NativeReference, ORIENTED_FILES
from pose_curation.orientation import Orientation
from pose_curation.storage import read_json, sha256, write_json


def main():
    destination = PROJECT / 'data/dev/head-direction-20261002'
    destination.mkdir(parents=True, exist_ok=True)
    service = FramedPreviews(PROJECT / 'data/curation')
    pose = NativeReference(service.character, sha256(service.character))
    cases = [('head_front', 'head', Orientation()),
             ('head_down', 'head', Orientation(30,20,-10)),
             ('bust_up', 'bust', Orientation(-35,-15,12))]
    report = {'reference_sha256': pose.content_hash, 'csp_verified': False,
              'automatic_face_recommendation_enabled': False, 'cases': []}
    try:
        for name, scope, angles in cases:
            state = service.wait_ready(pose, scope, orientation=angles)
            row = {'name': name, 'scope': scope, 'orientation': angles.public(), 'files': {}}
            for kind in ORIENTED_FILES:
                source = service.artifact(pose, scope, angles, kind, state['version'])
                target = destination / (name + source.suffix)
                shutil.copyfile(source, target)
                row['files'][kind] = {'file': target.name, 'sha256': sha256(target)}
            row['validation'] = read_json(destination / (name+'.json'))['fbx_validation']
            report['cases'].append(row)
    finally:
        service.close()
    write_json(destination/'verification.json', report)
    (destination/'README.txt').write_text(
        'Standin 기본 두상·흉상 FBX 시험 파일\n\n'
        'head_front: 두상, 정면\nhead_down: 두상, 좌우 30 / 높낮이 20 / 기울기 -10도\n'
        'bust_up: 흉상, 좌우 -35 / 높낮이 -15 / 기울기 12도\n\n'
        '각 FBX를 Blender에서 재가져와 뼈대 위치를 검사했습니다. JPG는 해당 FBX 렌더입니다.\n'
        'CSP의 정면·평행 투영 카메라에서 비교하세요. 기존 레이어 카메라는 바꾸지 않습니다.\n'
        '이 두상 파일의 CSP 검증은 아직 받지 않았습니다.\n'
        '기본 캐릭터의 목·표정·어깨이며 특정 러프를 재현한 파일이 아닙니다.\n'
        '메시만 절단하고 전체 뼈대는 보존합니다. BVH는 포함하지 않습니다.\n'
        '검수 화면: http://127.0.0.1:8765/head\n', encoding='utf-8')
    archive = destination.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as output:
        for path in sorted(destination.iterdir()):
            if path.is_file(): output.write(path,path.name)
    print({'archive':str(archive),'cases':len(cases),'validations':[row['validation'] for row in report['cases']]})


if __name__ == '__main__':
    main()
