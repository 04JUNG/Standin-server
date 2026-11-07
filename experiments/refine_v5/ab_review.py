"""Compare the saved FBX results in a common body-relative view."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from experiments.refine_v5.ab_common import load_json
from experiments.refine_v5.learning_data import frame
from experiments.refine_v5.render_review import render

METHODS = (
    ('B0', 'Original B0'),
    ('ortho_fixed', 'Orthographic control'),
    ('perspective', 'Perspective fit'),
    ('openai', 'OpenAI rotation'),
    ('GT', 'Ground-truth FBX'),
)
VIEWS = {'front': 0, 'three_quarter': 30, 'side': 90, 'back': 180}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    cases = load_json(args.root / 'manifest.json')['cases']
    audit = {(row['case_id'], row['method']): row for row in load_json(args.root / 'native/audit.json')}
    metrics = {(row['case_id'], row['method']): row for row in load_json(args.root / 'comparison.json')['rows']}
    cell_w, cell_h = 230, 295
    canvas = Image.new('RGB', (cell_w * 6, 60 + cell_h * len(cases)), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 15)
    except OSError:
        font = ImageFont.load_default()
    for col, label in enumerate(['Input rough'] + [item[1] for item in METHODS]):
        draw.text((col * cell_w + 9, 8), label, fill='#202020', font=font)
    draw.text((9, 35), 'Saved mesh in a body-relative view. Per-case headings show mean arm direction error against GT. GT camera is unknown.',
              fill='#404040', font=font)
    for row_index, case in enumerate(cases):
        case_id = case['case_id']
        y = 60 + row_index * cell_h
        with Image.open(case['image']) as image:
            rough = image.convert('RGB')
        rough.thumbnail((cell_w - 12, cell_h - 40))
        canvas.paste(rough, ((cell_w - rough.width) // 2, y + 26))
        draw.text((9, y + 4), case_id + '  ' + case['split'], fill='#202020', font=font)
        yaw = VIEWS[case['seed_view']]
        camera = Rotation.from_euler('y', yaw, degrees=True).as_matrix()
        meshes = []
        for method, _ in METHODS:
            bones = audit[case_id, method]['bones']
            _, center, scale, basis = frame(bones)
            with np.load(args.root / 'native' / f'{case_id}-{method}.npz') as mesh:
                vertices = (mesh['vertices'] - center) @ basis / scale
                meshes.append((vertices, mesh['faces']))
        xy = np.concatenate([(v @ camera.T)[:, :2] * [1, -1] for v, _ in meshes])
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        size = (cell_w, cell_h - 28)
        scale = min((size[0] - 22) / max(1e-5, hi[0] - lo[0]), (size[1] - 16) / max(1e-5, hi[1] - lo[1]))
        offset = np.asarray(size) / 2 - (lo + hi) * scale / 2
        for col, ((method, _), (vertices, faces)) in enumerate(zip(METHODS, meshes, strict=True), 1):
            image = render(vertices, faces, size=size, yaw=yaw, framing=(scale, offset))
            canvas.paste(image, (col * cell_w, y + 25))
            if method != 'GT':
                row = metrics[case_id, method]
                caption = f"{row['arm_mean_deg']:.1f}°"
                if method == 'perspective' and row['camera_status'] != 'fitted':
                    caption += '  (ortho fallback)'
                draw.text((col * cell_w + 9, y + 4), caption, fill='#202020', font=font)
    canvas.save(args.root / 'comparison-review.png')


if __name__ == '__main__':
    main()
