"""Read pinned geometry, audit paired mirror conventions without reranking."""
import argparse
from collections import Counter
import fcntl
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.experiment_z_rendered import save_json, sha256
from standin_eval.preview_mirror_diagnostic import pair_metrics


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    sources = ('scripts/diagnose_preview_mirror.py', 'standin_eval/preview_mirror_diagnostic.py',
               'scripts/experiment_z_rendered.py', 'src/bvh.py')
    protocol = {'purpose': 'geometry-only preview mirror convention diagnostic; no query or rankings changed',
                'predeclared': True, 'selection': 'every suffix _mirror with an exact unsuffixed source ID in each pinned snapshot',
                'hypotheses': 'hip-centered reflection across X or Z; with or without all anatomical left-right indices swapped',
                'tolerance': 1e-4, 'error': 'maximum of all17 joint distances divided by source torso length',
                'limits': ['torso front is a renderer convention, not rig-calibrated actual face direction',
                           'non-axis matches may involve other transforms; no bug attribution from suffix alone',
                           'COCO head-position collapse cannot preserve native BVH head rotation',
                           'no semantic accuracy or new holdout; frozen renderer/scoring unchanged'],
                'inputs': {str(p): sha256(p) for p in (a.historical, a.extended)},
                'code': {n: sha256(ROOT/n) for n in sources}}
    save_json(a.out/'protocol.json', protocol)
    summary = {}; verified = 0
    # A separate matrix/Frobenius check verifies each scalar max residual.
    permutation = np.eye(17)[[0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]]
    for label, path in (('historical1981', a.historical), ('extended1995', a.extended)):
        with np.load(path, allow_pickle=False) as d:
            ids = d['pose_ids'].tolist(); gallery = d['joints'].astype(float)
        lookup = dict(zip(ids, gallery)); records = []; missing = []
        for pid in ids:
            if not pid.endswith('_mirror'): continue
            source = pid[:-7]
            if source not in lookup: missing.append(pid); continue
            p, q = lookup[source], lookup[pid]
            record = {'pose_id': pid, 'source_id': source, **pair_metrics(p, q)}
            pc = p - .5 * (p[11] + p[12]); qc = q - .5 * (q[11] + q[12])
            scale = np.sqrt(np.sum((.5 * (pc[5] + pc[6]))**2))
            for r in record['hypotheses']:
                axis = 0 if r['reflection_axis'] == 'X' else 2
                matrix = np.eye(3); matrix[axis,axis] = -1
                transformed = (permutation @ pc if r['anatomical_lr_swapped'] else pc) @ matrix
                residual = np.sqrt(np.sum((transformed - qc)**2, axis=1)).max() / scale
                assert abs(residual - r['max_joint_error_torso_units']) < 1e-12
                if r['matches'] and r['reflected_front_dot_preview_front'] is not None:
                    expected = 1 if r['anatomical_lr_swapped'] else -1
                    assert abs(r['reflected_front_dot_preview_front'] - expected) < 1e-6
                verified += 1
            records.append(record)
        head_collapsed = np.max(np.linalg.norm(gallery[:,:5] - gallery[:,0:1], axis=2), axis=1) < 1e-8
        save_json(a.out/(label+'-pairs.json'), records)
        summary[label] = {'gallery_count': len(ids), 'paired_mirrors': len(records),
                          'unpaired_mirror_ids': missing,
                          'classifications': dict(Counter(r['classification'] for r in records)),
                          'head_landmarks_coincident_count': int(head_collapsed.sum())}
    save_json(a.out/'results.json', {'snapshots': summary, 'independently_checked_residuals': verified,
                                    'search_accuracy': None, 'renderer_changed': False,
                                    'interpretation': 'matching reflection conventions alone do not establish native rig facing correctness'})
    for p,h in protocol['inputs'].items(): assert sha256(Path(p)) == h
    for n,h in protocol['code'].items(): assert sha256(ROOT/n) == h
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for n in ('historical','extended','out'): parser.add_argument('--'+n, type=Path, required=True)
    a = parser.parse_args()
    with (ROOT/'out/.perspective-rough-review.lock').open('a+') as rough, (ROOT/'out/.joint-shape.lock').open('a+') as shape, (ROOT/'out/.first-rank-confirm.lock').open('a+') as fresh:
        for lock in (rough,shape,fresh): fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(a)
