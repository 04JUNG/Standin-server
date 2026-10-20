import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_body_observations import diagnose, summarize
from standin_eval.dual_label_search import LEFT_RIGHT


def fixture():
    gt = np.array([[(-1.) ** j, float(j // 2)] for j in range(17)])
    return gt, np.ones(17)


def test_global_labels_and_scale_invariance():
    gt, scores = fixture()
    a = diagnose(gt[LEFT_RIGHT], scores, gt)
    b = diagnose(3 * gt[LEFT_RIGHT] + 21, scores, 3 * gt + 21)
    assert a['oracle_global_swap'] and b['oracle_global_swap']
    assert all(j['global_error'] == 0 for j in a['joints'])
    assert [j['direct_error'] for j in a['joints']] == [j['direct_error'] for j in b['joints']]


def test_local_pair_not_whole_body_and_low_score_retained():
    gt, scores = fixture()
    xy = gt.copy(); xy[[9, 10]] = xy[[10, 9]]
    a = diagnose(xy, scores, gt)
    assert not a['oracle_global_swap']
    assert [p['part'] for p in a['pairs'] if p['material_reduction']] == ['wrists']
    scores[9] = .2
    b = diagnose(xy, scores, gt)
    assert len(b['joints']) == 12 and b['valid_body_joints'] == 11
    assert 'wrists' not in [p['part'] for p in b['pairs']]


def test_bins_and_failure_denominator():
    gt, scores = fixture(); scores[5:9] = [.2, .3, .5, .7]
    row = diagnose(gt, scores, gt)
    row.update(searchable=False, top5_hit=False)
    s = summarize([row, {'searchable': False, 'top5_hit': False}])
    assert s['n'] == 2 and s['searchable'] == 0
    assert [s['by_score'][b]['n'] for b in ('lt03', '03_to_05', '05_to_07', 'ge07')] == [1, 1, 1, 9]
    assert s['by_search_outcome']['miss']['n'] == 0


if __name__ == '__main__':
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_'):
            fn(); print('PASS', name)
