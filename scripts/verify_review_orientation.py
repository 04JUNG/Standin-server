"""Generate a self-contained CSP import pack using the review server's exact workers.

Run with the ordinary curation Python environment, not Blender. Creates only
local synthetic/curated pose outputs; does not drive CSP or claim CSP validation.
"""
import argparse
import html
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_curation.orientation import Orientation
from pose_curation.review.catalog import Catalog
from pose_curation.review.framing import FramedPreviews, ORIENTED_FILES
from pose_curation.storage import read_json, write_json

CASES = [('01_full_front', 'full', (0, 0, 0)),
         ('02_full_right45', 'full', (45, 0, 0)),
         ('03_full_combined', 'full', (37, 25, -18)),
         ('04_half_combined', 'half', (37, 25, -18)),
         ('05_bust_combined', 'bust', (37, 25, -18)),
         ('06_head_combined', 'head', (37, 25, -18)),
         ('07_half_opposite', 'half', (-55, -30, 20))]

INSTRUCTIONS = '''Standin — CSP 방향 출력 시험

이 묶음은 실제 출력 파일을 Blender에서 다시 읽고 검증한 결과입니다.
CSP에서의 가져오기 결과는 사용자가 확인할 단계이며 아직 검증 완료가 아닙니다.

1. index.html에서 각 파일의 예상 미리보기를 확인합니다.
2. CSP의 시험용 새 문서에서 파일 > 가져오기 > 3D 데이터로 .fbx를 엽니다.
3. 오브젝트 도구에서 카메라를 정면으로 맞추고, 지원되는 버전에서는
   보조 도구 상세 > 카메라 > 투영 방법을 평행 투영으로 설정합니다.
   자동 화면 맞춤으로 크기·위치는 다를 수 있으므로 방향과 실루엣을 비교합니다.
4. FBX에서 좌우·상하·기울기가 미리보기와 같은지 확인합니다.
   결과를 알려주실 때 파일명, CSP 버전, 실제 화면을 함께 알려주세요.

파일 의미
- oriented.fbx: 선택 범위의 메시와 전신 뼈대에 방향을 적용한 모델.
- preview.jpg: 실제 FBX를 다시 읽어 정면·정사영에서 렌더링한 예상 화면.
- orientation.json: 각도, 원본 해시, FBX 검증 결과.

각도 적용 출력은 FBX만 제공합니다. CSP에서 BVH 각도가 유지되지 않는 사용자
시험 결과를 반영했습니다. BVH는 검수 서버에서 원본 자세로 내려받을 수 있습니다.

좌우(yaw) +는 오른쪽으로 이동한 카메라 시점, 높낮이(pitch) +는 위에서 내려다봄,
기울기(roll) +는 화면 시계 방향입니다. 모델 전체에 적용하므로 바닥에 대해
모델이 기울어질 수 있습니다. 원본 관절 자세와 손가락은 유지합니다.
카메라 자동 설정·원근 전달·러프 자동 각도 추정은 이번 파일에 포함되지 않습니다.
모든 각도를 확인할 때 동일한 CSP 기준 카메라를 사용하세요.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose-key', default='340d8b16a0fac1d035ca9659')
    parser.add_argument('--output', type=Path, default=Path('data/dev/csp-fbx-orientation-20261002'))
    args = parser.parse_args()
    pose = Catalog(Path('data'), Path('data/curation')).get(args.pose_key)
    if pose is None:
        raise ValueError('Unknown pose key')
    service = FramedPreviews(Path('data/curation'))
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cards, results = [], []
    try:
        for label, scope, angles in CASES:
            orientation = Orientation(*angles)
            state = service.wait_ready(pose, scope, orientation=orientation)
            assert state['status'] == 'ready', state
            target = root / label
            target.mkdir(exist_ok=True)
            for kind in ORIENTED_FILES:
                source = service.artifact(pose, scope, orientation, kind, state['version'])
                shutil.copyfile(source, target / ('preview.jpg' if kind == 'preview' else source.name))
            metadata = read_json(target / 'orientation.json')
            assert metadata['angle_export_format'] == 'fbx'
            results.append({'case': label, **metadata})
            cards.append(f'<article><h2>{html.escape(label)}</h2><p>yaw {angles[0]}° · pitch {angles[1]}° · roll {angles[2]}°</p>'
                         f'<img src="{label}/preview.jpg" alt="{scope}"><p><a href="{label}/oriented.fbx">FBX</a></p></article>')
            print('VERIFIED', label, flush=True)
    finally:
        service.close()
    (root / 'README.txt').write_text(INSTRUCTIONS, encoding='utf-8-sig')
    write_json(root / 'verification.json', {'csp_verified': False, 'cases': results})
    from PIL import Image, ImageDraw
    sheet = Image.new('RGB', (4 * 280, 2 * 320), '#f5f6f2')
    draw = ImageDraw.Draw(sheet)
    for index, (label, scope, angles) in enumerate(CASES):
        x, y = (index % 4) * 280, (index // 4) * 320
        with Image.open(root / label / 'preview.jpg') as original:
            sheet.paste(original.resize((256, 256)), (x + 12, y + 12))
        draw.text((x + 12, y + 275), label, fill='#233d33')
        draw.text((x + 12, y + 293), f'yaw {angles[0]} / pitch {angles[1]} / roll {angles[2]}', fill='#233d33')
    sheet.save(root / 'contact-sheet.jpg', quality=92)
    (root / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>CSP 방향 시험</title>'
        '<style>body{font:16px system-ui;background:#f5f6f2;padding:30px;color:#233d33}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}article{background:white;padding:20px;border-radius:14px}h2{font-size:18px}img{width:100%;max-width:512px}a{color:#286551}</style>'
        '<h1>CSP 방향 출력 시험 · 7가지</h1><p>실제 FBX 재가져오기 검증 완료 · CSP 가져오기 확인 대기</p>'
        '<p>먼저 <a href="README.txt">확인 방법</a>을 읽어주세요. 각도 출력은 FBX로 제공합니다.</p><main>'
        + ''.join(cards) + '</main>', encoding='utf-8')
    archive = root.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                output.write(path, path.relative_to(root))
    print('PACK', archive, flush=True)


if __name__ == '__main__':
    main()
