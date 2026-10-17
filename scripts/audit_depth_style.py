"""Independent scalar scoring and input-contract audit for the style diagnostic."""
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_z_rendered import sha256, save_json
from scripts.experiment_z_axis_search import load_library
from scripts.diagnose_depth_style import read, fixed_infer
from scripts.experiment_humanart_z import read_rgb
from standin_eval.mixed_z import corrected_rtmw

SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
PAIRS = [(7, 5), (9, 5), (8, 6), (10, 6), (13, 11), (15, 11), (14, 12), (16, 12)]
STYLES = ('rough', 'solid')


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


def sign(x):
    return int(x > 0)-int(x < 0)


def main(args):
    protocol = read(args.out/'protocol.json')
    for path, digest in protocol['hashes'].items():
        assert sha256(Path(path)) == digest
    for path, digest in read(args.out/'prepared.json').items():
        assert sha256(args.out/path) == digest
    assert sha256(args.out/'predictions.json') == read(args.out/'predicted.json')['sha256']
    manifest = read(args.out/'manifest.json'); predictions = read(args.out/'predictions.json')
    answers = read(args.out/'answers.json'); results = read(args.out/'results.json')
    baseline = read(args.out/'baseline.json')
    source = read(ROOT/'out/humanart-plus-z-20261012/results.json')['synthetic']['records']
    assert len(manifest) == len(predictions) == len(answers) == len(baseline) == len(source) == 40
    poses = {p.pose_id: p for p in load_library(args.library_root)}
    checked = []; maximum_z = 0.; swapped_gt = 0
    descriptions = {s: {'images': 40, 'failed': 0, 'swaps': 0, 'body_score_ge03': 0,
                        'xy_gate_pairs': 0, 'errors': []} for s in STYLES}
    for row, pred, answer, old, replay in zip(manifest, predictions, answers, source, baseline):
        assert row['image'] == pred['image'] == answer['image'] == old['image'] == replay['image']
        assert row['human'] == old['human'] and row['mask'] == old['shared_mask']
        assert row['bbox'] == old['crop_in_rotated_image'] and row['turn'] == old['human']['turn']
        assert replay['rtmw'] == old['rtmw']
        p = poses[answer['pose_id']].joints.astype(float)
        hip = (p[11]+p[12])/2; torso3 = distance((p[5]+p[6])/2, hip)
        radians = math.radians(answer['yaw'])
        z = [(-(v[0]-hip[0])*math.sin(radians)+(v[2]-hip[2])*math.cos(radians))/torso3 for v in p]
        dz = max(abs(a-b) for a, b in zip(z, answer['camera_z']))
        maximum_z = max(dz, maximum_z); assert dz < 1e-6
        h = row['human']['keypoints2d']; gt = answer['gt_projected_xy']; mask = row['mask']
        body = [j for j in range(5, 17) if mask[j]]
        torso = distance([(h[5][i]+h[6][i])/2 for i in range(2)],
                         [(h[11][i]+h[12][i])/2 for i in range(2)])
        gt_swap = sum(distance(h[j], gt[SWAP[j]]) for j in body) < sum(distance(h[j], gt[j]) for j in body)
        swapped_gt += gt_swap
        expected = {}
        for style in STYLES:
            entry = row['styles'][style]
            assert sha256(Path(entry['path'])) == entry['sha256']
            value = pred['styles'][style]
            if 'error' in value:
                descriptions[style]['failed'] += 1
                expected[style] = None
                continue
            raw = value['raw']; r = raw['keypoints2d']
            if style == 'rough':
                for k in ('keypoints2d', 'z_meters', 'scores'):
                    assert np.allclose(raw[k], old['rtmw'][k], atol=1e-7, rtol=0)
            error = [[distance(h[j], r[perm[j]])/torso for j in range(17)] for perm in (list(range(17)), SWAP)]
            means = [sum(e[j] for j in body)/len(body) for e in error]
            swap = means[1] < means[0]; mapped = value['mapped']
            assert swap == mapped['swapped']
            assert abs(means[0]-mapped['mean_direct']) < 1e-10
            assert abs(means[1]-mapped['mean_swapped']) < 1e-10
            descriptions[style]['errors'].append(min(means))
            descriptions[style]['swaps'] += swap
            descriptions[style]['body_score_ge03'] += sum(v >= .3 for v in raw['scores'][5:])
            expected[style] = {}
            for mapping, perm in (('direct', list(range(17))), ('xy_aligned', SWAP if swap else list(range(17)))):
                depth = np.asarray([raw['z_meters'][j] for j in perm], np.float32)
                center = (depth[11]+depth[12])/2
                normalized = (depth-center)/.5
                normalized[~np.asarray(mask, bool)] = 0
                item = mapped['hypotheses'][mapping]
                assert np.array_equal(normalized, item['normalized_z'])
                e = [distance(h[j], r[perm[j]])/torso for j in range(17)]
                assert np.allclose(e, item['xy_error_torso'], atol=1e-10, rtol=0)
                assert item['scores'] == [raw['scores'][j] for j in perm]
                signs, active, gated = [], [], []
                for a, b in PAIRS:
                    diff = normalized[a]-normalized[b]
                    use = bool(mask[a] and mask[b] and abs(diff) >= .1)
                    signs.append(sign(float(diff))); active.append(use)
                    gated.append(bool(use and e[a] <= .5 and e[b] <= .5))
                assert (signs, active, gated) == (item['signs'], item['all'], item['xy_gate'])
                expected[style][mapping] = {'signs': signs, 'all': active, 'xy_gate': gated}
            descriptions[style]['xy_gate_pairs'] += sum(expected[style]['xy_aligned']['xy_gate'])
        for frame in ('semantic', 'observed_human'):
            # Use saved float32 camera values after independently verifying their geometry.
            truth = answer['camera_z']
            if frame == 'observed_human' and gt_swap:
                truth = [truth[j] for j in SWAP]
            for mapping in ('direct', 'xy_aligned'):
                for gate in ('all', 'xy_gate'):
                    cell = {'image': row['image'], 'frame': frame, 'mapping': mapping,
                            'gate': gate, 'gt_n': 0, 'gains': 0, 'losses': 0}
                    for scope in ('own', 'common'):
                        cell[scope] = {s: dict(n=0, correct=0, always_front=0, always_back=0) for s in STYLES}
                    for k, (a, b) in enumerate(PAIRS):
                        delta = truth[a]-truth[b]; truth_sign = sign(delta)
                        gt_ok = bool(mask[a] and mask[b] and abs(delta) >= .1)
                        cell['gt_n'] += gt_ok
                        use, correct = {}, {}
                        for style in STYLES:
                            item = expected[style][mapping] if expected[style] else None
                            use[style] = bool(gt_ok and item and item[gate][k])
                            correct[style] = bool(item and item['signs'][k] == truth_sign)
                        common = all(use.values())
                        for scope in ('own', 'common'):
                            for style in STYLES:
                                if (use[style] if scope == 'own' else common):
                                    cell[scope][style]['n'] += 1
                                    cell[scope][style]['correct'] += correct[style]
                                    cell[scope][style]['always_front'] += truth_sign < 0
                                    cell[scope][style]['always_back'] += truth_sign > 0
                        cell['gains'] += bool(common and not correct['rough'] and correct['solid'])
                        cell['losses'] += bool(common and correct['rough'] and not correct['solid'])
                    checked.append(cell)
    assert checked == results['records'] and len(checked) == 320
    for group in results['depth_order']:
        cells = [c for c in checked if all(c[k] == group[k] for k in ('frame', 'mapping', 'gate'))]
        assert len(cells) == 40
        for key in ('gt_n', 'gains', 'losses'):
            assert group[key] == sum(c[key] for c in cells)
        for scope in ('own', 'common'):
            for style in STYLES:
                for key in ('n', 'correct', 'always_front', 'always_back'):
                    assert group[scope][style][key] == sum(c[scope][style][key] for c in cells)
        assert group['images_better'] == sum(c['gains'] > c['losses'] for c in cells)
        assert group['images_worse'] == sum(c['gains'] < c['losses'] for c in cells)
        assert group['images_tied'] == sum(c['gains'] == c['losses'] for c in cells)
    for style, desc in descriptions.items():
        values = desc.pop('errors')
        desc['median_xy_error_human_torso'] = float(np.median(values)) if values else None
        assert desc == results['description'][style]
    model = corrected_rtmw(args.model)
    for row, pred in zip(manifest[:2], predictions[:2]):
        for style in STYLES:
            replay = fixed_infer(read_rgb(row['styles'][style]['path']), model, row['turn'], row['bbox'])
            assert replay == pred['styles'][style]['raw']
    assert results['retrieval_accuracy'] is None and results['diagnostic_only']
    output = {'images': 40, 'inferences': 80, 'independent_scored_cells': len(checked),
              'independent_camera_z_max_error': maximum_z, 'oracle_human_gt_swaps': swapped_gt,
              'replayed_inferences': 4, 'human_xy_mask_crop_turn_unchanged': True,
              'hashes': {p: sha256(args.out/p) for p in ('protocol.json', 'predictions.json', 'results.json')},
              'auditor_sha256': sha256(Path(__file__))}
    save_json(args.out/'audit.json', output)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('out', 'library-root', 'model'):
        p.add_argument('--'+name, type=Path, required=True)
    main(p.parse_args())
