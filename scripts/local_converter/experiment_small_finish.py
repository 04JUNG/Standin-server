"""Small finishing pass on rejected skin candidates; diagnostic only."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from converter.local import surface as s

def run(base, motion, out):
    out.mkdir(parents=True, exist_ok=False)
    reports = json.loads((motion / 'motion-fit-results.json').read_text())
    rows = []
    for record in reports:
        mi, pi = (record['model'], record['pose'])
        valid = [v for v in record['variants'] if 'metrics' in v]
        chosen = min(valid, key=lambda v: v['metrics']['maxStretch'])
        raw = np.load(base / f'stages-{mi}-{pi}.npz')
        rest, f = (raw['rest'], raw['triangles'])
        p = np.load(motion / f"{mi}-{pi}-{chosen['label']}.npz")['points']
        target, report = s.correct(rest, p, f, raw['anchors'])
        scale = float(np.ptp(rest, axis=0).max())
        g = s.Neighborhood(f, rest)
        rl = s.lengths(rest[g.b] - rest[g.a])
        area, n = s.triangle_geometry(rest, f)
        i, j = g.face_pairs.T
        sm = (n[i] * n[j]).sum(1) > np.cos(np.pi / 6)
        metrics = lambda q: s.metrics(q, f, g, rl, area, sm)
        base_metrics = metrics(raw['corrected'])
        delta = target - p
        delta *= np.minimum(1.0, 0.003 * scale / np.maximum(s.lengths(delta), 1e-30))[:, None]
        variants = []
        for strength in (0.25, 0.5, 1.0):
            q = p + delta * strength
            m = metrics(q)
            variants.append(dict(strength=strength, maxFinishMoveM=float(s.lengths(q - p).max()), metrics=m, regressedV10=[k for k in base_metrics if m[k] > base_metrics[k] + 1e-06]))
            np.savez_compressed(out / f'{mi}-{pi}-{strength}.npz', points=q)
        rows.append(dict(model=mi, pose=pi, diagnosticInputOnly=True, stage2Candidate=chosen['label'], stage2Accepted=not chosen['regressedV10'] and (not chosen['newCollapsedFaces']), surfaceSelector=report, v10=base_metrics, variants=variants))
        (out / 'finish-results.json').write_text(json.dumps(rows, indent=2))
        print('FINISH_STAGE', mi, pi, report['applied'], [(v['strength'], v['regressedV10']) for v in variants], flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--motion', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    run(a.base, a.motion, a.out)
