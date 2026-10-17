"""Controlled handedness smoke test, not search accuracy or a model change."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.library import _synthetic_pose
from scripts.experiment_z_rendered import render_pose, save_json, sha256
from standin_eval.z_diagnostics import infer_pose
from standin_eval.perspective_search import normalize_batch
from standin_eval.pose_geometry_diagnostic import body_label_errors


def asymmetric_front_pose():
    # Subject's left is +X on screen when frontal; renderer face points -Z.
    p = _synthetic_pose('standing').copy()
    p[:, 0] *= -1
    p[7] = [.34, .65, 0]
    p[9] = [.32, .95, 0]
    p[8] = [-.30, .25, -.10]
    p[10] = [-.40, .0, -.20]
    return p


def run(a):
    from rtmlib import RTMPose
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'images').mkdir()
    save_json(a.out/'protocol.json', {
        'purpose': 'mechanical/representation smoke test; one deliberately asymmetric scaffold, not a benchmark',
        'conditions': {'yaw': [0, 60, 180, 240], 'style': ['solid', 'rough'], 'pitch': 0, 'distance': None},
        'input': 'RGB, same frozen Human-Art wrapper, foreground crop',
        'arms': ['upright only', 'confidence-selected four quarter turns'],
        'rules': 'score>=0.3 body mask; >=6 body joints and shoulder/hip anchors; no GT-selected rotation or correction',
        'labels': 'COCO17 anatomical left/right; no automatic swapping',
        'official_schema': 'https://github.com/open-mmlab/mmpose/blob/main/configs/_base_/datasets/humanart.py',
        'model_sha256': sha256(a.model),
        'code_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in [
            Path(__file__), ROOT/'scripts/experiment_z_rendered.py', ROOT/'standin_eval/z_diagnostics.py']}})
    model = RTMPose(str(a.model), model_input_size=(192, 256), backend='onnxruntime', device='cpu')
    assert model.to_openpose is False
    records = []
    for yaw in (0, 60, 180, 240):
        for style in ('solid', 'rough'):
            name = f'{style}_{yaw:03}.png'
            image, gt = render_pose(asymmetric_front_pose(), yaw, style=style, seed=20261022)
            image.save(a.out/'images'/name)
            row = {'image': name, 'image_sha256': sha256(a.out/'images'/name), 'yaw': yaw,
                   'style': style, 'gt_xy': gt.tolist(), 'arms': {}}
            for arm, turns in [('upright', (0,)), ('four_turns', (0, 1, 2, 3))]:
                pred = infer_pose(np.asarray(image), model, crop=True, turns=turns, is_3d=False)
                mask = np.asarray(pred['scores']) >= .3
                valid = bool(mask[5:].sum() >= 6 and mask[[5, 6, 11, 12]].all())
                rec = {'prediction': pred, 'searchable': valid, 'valid_body': int(mask[5:].sum())}
                if valid:
                    direct, swapped = body_label_errors(normalize_batch(pred['keypoints2d']), normalize_batch(gt), mask)
                    rec.update(direct_xy_error=direct, swapped_xy_error=swapped,
                               label_preference='direct' if direct <= swapped else 'swapped')
                row['arms'][arm] = rec
            records.append(row)
            save_json(a.out/'results.json', {'warning': 'Diagnostic only, no search ranking or fixed-label correction adopted', 'records': records})
            print(name, {k: {x:v for x,v in r.items() if x != 'prediction'} for k,r in row['arms'].items()}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())
