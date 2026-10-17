"""Paired representation diagnostic, not a rough reconstruction or search method."""
import argparse
import fcntl
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_z_rendered import render_pose, save_json, sha256
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_humanart_z import read_rgb
from standin_eval.mixed_z import corrected_rtmw
from standin_eval.z_diagnostics import undo_quarter_turn, foreground_bbox
from standin_eval.depth_correspondence import map_depth
from standin_eval.dual_label_search import LEFT_RIGHT
from standin_eval.ordinal_depth import PAIRS
from standin_eval.perspective_search import Camera, camera_points

CACHE = ROOT/'out/humanart-plus-z-20261012/results.json'
ANSWERS = ROOT/'out/z-diagnosis-20261012-heldout/answers.json'
STYLES = ('rough', 'solid')
MAPPINGS = ('direct', 'xy_aligned')


def read(path):
    return json.loads(path.read_text())


def verify(args):
    protocol = read(args.out/'protocol.json')
    for path, digest in protocol['hashes'].items():
        assert sha256(Path(path)) == digest, path


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=False)
    paths = [CACHE, ANSWERS, args.model, args.library_root/'poses.db', Path(__file__),
             ROOT/'scripts/experiment_z_rendered.py', ROOT/'scripts/experiment_z_axis_search.py',
             ROOT/'scripts/experiment_humanart_z.py', ROOT/'standin_eval/mixed_z.py',
             ROOT/'standin_eval/z_diagnostics.py', ROOT/'standin_eval/depth_correspondence.py',
             ROOT/'standin_eval/dual_label_search.py', ROOT/'standin_eval/ordinal_depth.py',
             ROOT/'standin_eval/perspective_search.py', ROOT/'src/bvh.py',
             ROOT/'standin_eval/z_axis_search.py']
    save_json(args.out/'protocol.json', {
        'stage': 'known-development paired representation diagnostic; no retrieval, no new holdout',
        'sample': 'all original synthetic40, original order, no replacement; no actual rough10 inference',
        'arms': 'original rough vs solid colored shaded ellipsoid; same geometry, yaw, 512px projection',
        'render_check': 'rough seed20261013+i must reproduce pixels exactly; both styles must reproduce gt_projected_xy within1e-6px',
        'preprocess': 'RGB; corrected RTMW288x384 CPU; original rough HumanArt turn and pixel crop FIXED for both arms',
        'human': 'cached original HumanArt XY and mask fixed; never re-estimate on solid',
        'mapping': 'direct and XY-only whole-body RTMW-to-Human LR alignment, same frozen map_depth',
        'depth': 'camera +Z farther; raw RTMW depth /0.5m after hip centering; GT normalized by true 3D torso',
        'eligibility': 'eight frozen limb-anchor pairs; both original Human joints valid; predicted absdelta>=.1; optional both XY errors<=.5 Human torso; GT absdelta>=.1',
        'score': 'semantic GT and secondary observed-Human GT frame; latter chooses global GT LR using Human XY for scoring only; own/common pair counts, fixed GT denominator and abstentions, constant front/back controls, image gains/losses',
        'primary_diagnostic': 'XY-aligned observed-Human frame, common eligible pairs; all other cells descriptive; no tuning or adoption gate',
        'replay': 'all rough raw XY/Z/scores must match original within1e-7 before scoring',
        'failure': 'inference exception/nonfinite recorded, no substitution; scoring abstains for failed arm',
        'limitations': ['same examined40, orthographic yaw only, no strong perspective',
                       'color/fill/shading/background/contours change together, not shading-only',
                       'GT3D used to create controlled stimulus, not a real rough-to-solid solution',
                       'pairs within an image correlated; no generalization or search accuracy claim'],
        'hashes': {str(p.resolve()): sha256(p) for p in paths}})
    cache = read(CACHE)['synthetic']['records']
    answers = read(ANSWERS)
    assert len(cache) == len(answers) == 40
    poses = {p.pose_id: p for p in load_library(args.library_root)}
    manifest, truth, baseline = [], [], []
    for style in STYLES:
        (args.out/style).mkdir()
    for i, (old, answer) in enumerate(zip(cache, answers)):
        assert old['image'] == answer['image'] and not old.get('error')
        assert not old['z_fallback'] and old['human']['turn'] == old['rtmw']['turn']
        assert sha256(Path(old['path'])) == old['sha256'] == answer['sha256']
        pose = poses[answer['pose_id']]
        row = {'image': old['image'], 'human': old['human'], 'mask': old['shared_mask'],
               'turn': old['human']['turn'], 'bbox': old['crop_in_rotated_image'], 'styles': {}}
        original = read_rgb(old['path'])
        assert row['bbox'] == foreground_bbox(np.ascontiguousarray(np.rot90(original, row['turn'])))
        for style in STYLES:
            image, xy = render_pose(pose.joints, answer['yaw'], size=512, style=style, seed=20261013+i)
            assert np.allclose(xy, answer['gt_projected_xy'], atol=1e-6, rtol=0)
            if style == 'rough':
                assert np.array_equal(np.asarray(image), original), 'rough pixels changed'
            path = args.out/style/old['image']
            image.save(path)
            row['styles'][style] = {'path': str(path.resolve()), 'sha256': sha256(path)}
        manifest.append(row)
        baseline.append({'image': old['image'], 'rtmw': old['rtmw']})
        truth.append({**answer, 'camera_z': camera_points(pose.joints, Camera(answer['yaw']))[:, 2].tolist()})
    save_json(args.out/'manifest.json', manifest)
    save_json(args.out/'baseline.json', baseline)
    save_json(args.out/'answers.json', truth)
    verify(args)
    save_json(args.out/'prepared.json', {p: sha256(args.out/p) for p in ('manifest.json', 'baseline.json', 'answers.json')})
    print('prepared 40 matched pairs; rough pixels and projected GT reproduced', flush=True)


def fixed_infer(rgb, model, turn, bbox):
    rotated = np.ascontiguousarray(np.rot90(rgb, turn))
    k3, scores, _, k2 = model(rotated, bboxes=[bbox])
    points = undo_quarter_turn(k2[0, :17], turn, rgb.shape[1], rgb.shape[0])
    z = np.asarray(k3[0, :17, 2], dtype=float)
    conf = np.asarray(scores[0, :17], dtype=float)
    if not all(np.isfinite(v).all() for v in (points, z, conf)):
        raise ValueError('nonfinite model output')
    return {'keypoints2d': points.tolist(), 'z_meters': z.tolist(), 'scores': conf.tolist()}


def features(row, pred):
    mapped = map_depth(row['human']['keypoints2d'], row['mask'], pred['keypoints2d'],
                       pred['z_meters'], pred['scores'])
    mask = np.asarray(row['mask'], bool)
    for item in mapped['hypotheses'].values():
        z = np.asarray(item['normalized_z'], np.float32)
        delta = z[PAIRS[:, 0]] - z[PAIRS[:, 1]]
        active = mask[PAIRS].all(1) & (abs(delta) >= .1)
        gate = (np.asarray(item['xy_error_torso'])[PAIRS] <= .5).all(1)
        item.update(signs=np.sign(delta).astype(int).tolist(), all=active.tolist(),
                    xy_gate=(active & gate).tolist())
    return mapped


def infer(args):
    verify(args)
    assert not (args.out/'predictions.json').exists()
    for name, digest in read(args.out/'prepared.json').items():
        assert sha256(args.out/name) == digest
    # No answer values or library data are read below. Renderer/GT preparation is separate.
    manifest = read(args.out/'manifest.json')
    baseline = {r['image']: r['rtmw'] for r in read(args.out/'baseline.json')}
    model = corrected_rtmw(args.model)
    predictions = []
    (args.out/'raw').mkdir(exist_ok=False)
    max_replay = 0.
    for row in manifest:
        result = {'image': row['image'], 'styles': {}}
        for style in STYLES:
            entry = row['styles'][style]
            assert sha256(Path(entry['path'])) == entry['sha256']
            try:
                pred = fixed_infer(read_rgb(entry['path']), model, row['turn'], row['bbox'])
            except Exception as error:
                result['styles'][style] = {'error': str(error)}
                continue
            if style == 'rough':
                delta = max(float(np.max(abs(np.asarray(pred[k])-baseline[row['image']][k])))
                            for k in ('keypoints2d', 'z_meters', 'scores'))
                max_replay = max(delta, max_replay)
                assert delta <= 1e-7, ('rough replay mismatch', row['image'], delta)
            result['styles'][style] = {'raw': pred, 'mapped': features(row, pred)}
        predictions.append(result)
        save_json(args.out/'raw'/row['image'].replace('.png', '.json'), result)
        print('inferred', len(predictions), '/40', row['image'], flush=True)
    verify(args)
    save_json(args.out/'predictions.json', predictions)
    save_json(args.out/'predicted.json', {'sha256': sha256(args.out/'predictions.json'),
              'max_rough_replay_error': max_replay, 'answer_values_read': False})


def pair_counts(signs, truth, use):
    return {'n': int(use.sum()), 'correct': int(((signs == truth) & use).sum()),
            'always_front': int(((truth < 0) & use).sum()), 'always_back': int(((truth > 0) & use).sum())}


def evaluate(args):
    verify(args)
    assert not (args.out/'results.json').exists()
    assert sha256(args.out/'predictions.json') == read(args.out/'predicted.json')['sha256']
    for name, digest in read(args.out/'prepared.json').items():
        assert sha256(args.out/name) == digest
    manifest = read(args.out/'manifest.json'); predictions = read(args.out/'predictions.json')
    answers = read(args.out/'answers.json'); cells = []
    desc = {s: {'images': 40, 'failed': 0, 'swaps': 0, 'xy_errors': [], 'body_score_ge03': 0,
                'xy_gate_pairs': 0} for s in STYLES}
    for row, pred, answer in zip(manifest, predictions, answers):
        assert row['image'] == pred['image'] == answer['image']
        mask = np.asarray(row['mask'], bool); body = np.flatnonzero(mask[5:])+5
        h = np.asarray(row['human']['keypoints2d']); gt = np.asarray(answer['gt_projected_xy'])
        swap = bool(np.linalg.norm(h[body]-gt[LEFT_RIGHT][body], axis=1).mean() <
                    np.linalg.norm(h[body]-gt[body], axis=1).mean())
        for style in STYLES:
            value = pred['styles'][style]
            if 'error' in value:
                desc[style]['failed'] += 1
                continue
            m = value['mapped']; desc[style]['swaps'] += int(m['swapped'])
            desc[style]['xy_errors'].append(min(m['mean_direct'], m['mean_swapped']))
            desc[style]['body_score_ge03'] += int((np.asarray(value['raw']['scores'])[5:] >= .3).sum())
            desc[style]['xy_gate_pairs'] += sum(m['hypotheses']['xy_aligned']['xy_gate'])
        for frame in ('semantic', 'observed_human'):
            truth = np.asarray(answer['camera_z'])
            if frame == 'observed_human' and swap:
                truth = truth[LEFT_RIGHT]
            delta = truth[PAIRS[:, 0]]-truth[PAIRS[:, 1]]
            truth_sign = np.sign(delta)
            gt_valid = (abs(delta) >= .1) & mask[PAIRS].all(1)
            for mapping in MAPPINGS:
                for gate in ('all', 'xy_gate'):
                    eligible, signs = {}, {}
                    for style in STYLES:
                        value = pred['styles'][style]
                        item = value.get('mapped', {}).get('hypotheses', {}).get(mapping)
                        signs[style] = np.asarray(item['signs']) if item else np.zeros(8)
                        eligible[style] = gt_valid & (np.asarray(item[gate], bool) if item else np.zeros(8, bool))
                    common = eligible['rough'] & eligible['solid']
                    correct = {s: signs[s] == truth_sign for s in STYLES}
                    cell = {'image': row['image'], 'frame': frame, 'mapping': mapping, 'gate': gate,
                            'gt_n': int(gt_valid.sum()),
                            'own': {s: pair_counts(signs[s], truth_sign, eligible[s]) for s in STYLES},
                            'common': {s: pair_counts(signs[s], truth_sign, common) for s in STYLES},
                            'gains': int((common & ~correct['rough'] & correct['solid']).sum()),
                            'losses': int((common & correct['rough'] & ~correct['solid']).sum())}
                    cells.append(cell)
    aggregates = []
    for frame in ('semantic', 'observed_human'):
        for mapping in MAPPINGS:
            for gate in ('all', 'xy_gate'):
                cc = [r for r in cells if (r['frame'], r['mapping'], r['gate']) == (frame, mapping, gate)]
                group = {k: cc[0][k] for k in ('frame', 'mapping', 'gate')}
                group.update({k: sum(r[k] for r in cc) for k in ('gt_n', 'gains', 'losses')})
                group.update({scope: {s: {k: sum(r[scope][s][k] for r in cc)
                                           for k in ('n', 'correct', 'always_front', 'always_back')}
                                       for s in STYLES} for scope in ('own', 'common')})
                group.update(images_better=sum(r['gains'] > r['losses'] for r in cc),
                             images_worse=sum(r['gains'] < r['losses'] for r in cc),
                             images_tied=sum(r['gains'] == r['losses'] for r in cc))
                aggregates.append(group)
    for value in desc.values():
        value['median_xy_error_human_torso'] = float(np.median(value.pop('xy_errors'))) if value['failed'] < 40 else None
    save_json(args.out/'results.json', {'diagnostic_only': True, 'retrieval_accuracy': None,
              'description': desc, 'depth_order': aggregates, 'records': cells})
    print(json.dumps({'description': desc, 'primary': [r for r in aggregates if r['frame'] == 'observed_human' and r['mapping'] == 'xy_aligned']}, indent=2))


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'infer', 'evaluate'))
    for name in ('out', 'library-root', 'model'):
        parser.add_argument('--'+name, required=True, type=Path)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with (args.out.parent/'.depth-style.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        globals()[args.stage](args)
