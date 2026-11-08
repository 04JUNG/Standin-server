"""Review the same saved pose with reviewed and detector-based part goals."""

import argparse
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw, ImageFont

from experiments.refine_v5.ab_common import load_json, save_json
from experiments.refine_v5.depth_ten_report import review_images
from experiments.refine_v5.learning_data import digest


LABELS = {'B0': 'Part goals + head', 'reviewed': 'Reviewed goals', 'detector': 'Detector + reviewed intent',
          'GT': 'Ground truth'}


def bundle(root, prior, out):
    case = load_json(root / 'input.json')
    cid = case['case_id']
    source = Path(case['source'])
    gt = next(row for row in load_json(prior / 'native/audit.json')
              if row['case_id'] == cid and row['method'] == 'GT')
    if digest(source) != case['source_sha256'] or digest(gt['path']) != gt['sha256']:
        raise ValueError('comparison references changed')
    observed = load_json(root / 'observation.json')
    out.mkdir(exist_ok=False)
    (out / 'images').mkdir()
    case['cameras'].update({method: case['camera'] for method in LABELS})
    case['cameras']['ortho_fixed'] = case['camera']
    audits = {mode: load_json(root / mode / 'audit.json') for mode in ('reviewed', 'detector')}
    native = {(cid, 'B0'): {'bones': case['bones']}, (cid, 'GT'): gt,
              **{(cid, mode): audit for mode, audit in audits.items()}}
    # Reuse the front/side renderer with its prior label area blanked below.
    rows = [dict(case_id=cid, method=mode, chain_error_deg=0.,
                 status='reference', camera_status=None) for mode in LABELS if mode != 'GT']
    with TemporaryDirectory() as tmp:
        staging = Path(tmp) / 'native'
        staging.mkdir()
        paths = {'B0': root / 'base.npz', 'GT': prior / 'native' / f'{cid}-GT.npz',
                 **{mode: root / mode / 'final.npz' for mode in audits}}
        for mode, path in paths.items():
            (staging / f'{cid}-{mode}.npz').symlink_to(path.resolve())
        review_images(staging.parent, out, [case], rows, native, labels=LABELS, short=LABELS)
    metrics = {}
    for mode, audit in audits.items():
        plan = load_json(root / mode / 'plan.json')['result']
        path = root / mode / f'{cid}-part-goals-head-limbs.fbx'
        if digest(path) != audit['output_sha256']:
            raise ValueError('saved FBX changed')
        target = out / f'{cid}-{mode}.fbx'
        shutil.copyfile(path, target)
        metrics[mode] = dict(accepted=audit['accepted'], fbx=target.name, edited_bones=list(plan['edits_deg']) if plan else [],
                             audit={key: value for key, value in audit.items() if key != 'bones'})
    with Image.open(out / 'overview.png') as original:
        comparison = original.convert('RGB')
    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 13)
    except OSError:
        font = ImageFont.load_default()
    descriptions = [audits['reviewed'].get('before_bends'), *(a.get('after_bends') for a in audits.values())]
    for col, bends in enumerate(descriptions, 1):
        x = col * 245
        draw.rectangle((x + 4, 24, x + 241, 41), fill='white')
        caption = ('R knee %.1f° | L elbow %.1f°' % (bends['right_knee'], bends['left_elbow'])
                   if bends else 'Source fallback')
        draw.text((x + 8, 25), caption, fill='#333333', font=font)
    for path in (out / '029-comparison.png', out / 'overview.png', out / 'images' / f'{cid}.png'):
        comparison.save(path)
    save_json(out / 'comparison.json', dict(case_id=cid, observation=observed, results=metrics))
    print({key: data['audit'].get('after_bends') for key, data in metrics.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'prior', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    bundle(**vars(parser.parse_args()))
