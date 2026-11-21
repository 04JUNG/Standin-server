"""Reproduce the 37 rejected female-CMU surface candidates from a local stage NPZ.

Diagnostic only. No production thresholds, weights, bones or materials change.
Use stages-1-1.npz from verify_followup_geometry.py with a new output folder.
Candidate coordinates stay local; JSON records every regression and collapse.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from converter.local import surface as s
from converter.local.surface import lengths

def main(stages, output):
    D = output
    D.mkdir(parents=True, exist_ok=False)
    raw = np.load(stages)
    d = {'rest': raw['rest'], 'baked': raw['baked'], 'surface': raw['corrected'], 'faces': raw['triangles'], 'pinned': raw['anchors']}
    r, p, f = (d['rest'], d['baked'], d['faces'])
    g = s.Neighborhood(f, r)
    rl = s.lengths(r[g.b] - r[g.a])
    ra, n = s.triangle_geometry(r, f)
    a, b = g.face_pairs.T
    al = (n[a] * n[b]).sum(1)
    sm = al > np.cos(np.pi / 6)
    g.fixed |= d['pinned']
    crease = g.face_pairs[al < np.cos(np.pi / 3)]
    g.fixed[f[crease].ravel()] = True
    support = s.damage_support(g, r, p, f, rl, ra, al)
    g.fixed |= support == 0
    before = s.metrics(p, f, g, rl, ra, sm)
    base = s.metrics(d['surface'], f, g, rl, ra, sm)
    scale = float(2 * s.lengths(r - r.mean(0)).max())
    ratio = s.lengths(p[g.b] - p[g.a]) / rl
    severe = np.zeros(len(r))
    bad = ratio > 4
    severe[g.a[bad]] = 1
    severe[g.b[bad]] = 1
    for amount in (0.75, 0.5, 0.25):
        adj = (severe[g.a] > 0) | (severe[g.b] > 0)
        ids = np.r_[g.a[adj], g.b[adj]]
        severe[ids] = np.maximum(severe[ids], amount)
    steps = int(np.clip((0.055 * scale / np.median(rl)) ** 2, 20, 140))
    ref = g.smooth(r, steps)
    low = g.smooth(p, steps)
    rot = s.rotations(g, ref, low)
    detail = np.einsum('nij,nj->ni', rot, r - ref)
    delta = low + detail - p
    rows = []
    for capratio in (0.021, 0.04, 0.08):
        cap = np.maximum(0.021 * scale * support, capratio * scale * severe)
        cap[g.fixed] = 0
        for mode in ('strain', 'detail', 'fair'):
            cor = delta * np.minimum(1.0, cap / np.maximum(s.lengths(delta), 1e-30))[:, None]
            seed = p.copy() if mode == 'strain' else p + cor
            if mode == 'fair':
                seed += np.clip(g.smooth(s.lengths(delta)[:, None], 8) / (0.003 * scale), 0, 1) * (g.smooth(seed, max(12, steps // 3)) - seed)
            x = s.limit_strain(seed, p, g, rl, cap)
            for blend in (1.0, 0.75, 0.5):
                q = p + blend * (x - p)
                m = s.metrics(q, f, g, rl, ra, sm)
                area, _ = s.triangle_geometry(q, f)
                newcollapse = int(((area < 0.01 * ra) & (s.triangle_geometry(p, f)[0] >= 0.01 * ra)).sum())
                name = f'{capratio}-{mode}-{blend}'
                row = dict(name=name, metrics=m, newCollapse=newcollapse, regressed=[k for k in before if m[k] > before[k] + 1e-06], vsBaseline=[k for k in base if m[k] > base[k] + 1e-06], maxMoveM=float(s.lengths(q - p).max()), moved=int((s.lengths(q - p) > 1e-10).sum()))
                rows.append(row)
                print(json.dumps(row), flush=True)
                np.savez_compressed(D / f'candidate-{name}.npz', vertices=q)
    (D / 'experiment.json').write_text(json.dumps(dict(before=before, baseline=base, rows=rows), indent=2))
    active = ~g.fixed[g.a] | ~g.fixed[g.b]
    i, j = (g.a[active], g.b[active])
    nr = len(r)
    degree = np.bincount(np.r_[i, j], minlength=nr).astype(float)
    edge = r[i] - r[j]

    def accum(values):
        return np.column_stack([np.bincount(i, v, nr) + np.bincount(j, v, nr) for v in values.T])

    def signed(values):
        return np.column_stack([np.bincount(i, v, nr) - np.bincount(j, v, nr) for v in values.T])

    def relax(tether, capratio):
        cap = np.maximum(0.021 * scale * support, capratio * scale * severe)
        cap[g.fixed] = 0
        anchor = degree * tether / np.maximum(support, 0.02) ** 2
        x = p.copy()
        for _ in range(18):
            rotation = s.rotations(g, r, x)
            goal = np.einsum('eij,ej->ei', (rotation[i] + rotation[j]) * 0.5, edge)
            rhs = signed(goal)
            for _ in range(16):
                neigh = np.column_stack([np.bincount(i, x[j, c], nr) + np.bincount(j, x[i, c], nr) for c in range(3)])
                proposed = (neigh + rhs + anchor[:, None] * p) / np.maximum((degree + anchor)[:, None], 1e-30)
                offset = proposed - p
                offset *= np.minimum(1.0, cap / np.maximum(lengths(offset), 1e-30))[:, None]
                x = 0.5 * x + 0.5 * (p + offset)
                x[g.fixed] = p[g.fixed]
        return x
    rows = []
    for tether in (0.01, 0.04, 0.15):
        for capratio in (0.04, 0.08):
            q = relax(tether, capratio)
            m = s.metrics(q, f, g, rl, ra, sm)
            ar, _ = s.triangle_geometry(q, f)
            nc = int(((ar < 0.01 * ra) & (s.triangle_geometry(p, f)[0] >= 0.01 * ra)).sum())
            name = f'arap-{tether}-{capratio}'
            row = dict(name=name, metrics=m, newCollapse=nc, regressed=[k for k in before if m[k] > before[k] + 1e-06], vsBaseline=[k for k in base if m[k] > base[k] + 1e-06], maxMoveM=float(lengths(q - p).max()))
            rows.append(row)
            print(json.dumps(row), flush=True)
            np.savez_compressed(D / f'candidate-{name}.npz', vertices=q)
    (D / 'arap-experiment.json').write_text(json.dumps(rows, indent=2))

    def project(x, upper, lower, cap, iterations):
        a, b = (g.a, g.b)
        mobility = (~g.fixed[a]).astype(float) + ~g.fixed[b]
        x = x.copy()
        for _ in range(iterations):
            vec = x[b] - x[a]
            le = s.lengths(vec)
            excess = le - np.clip(le, lower * rl, upper * rl)
            bad = (abs(excess) > rl * 1e-05) & (mobility > 0)
            i, j = (a[bad], b[bad])
            st = vec[bad] * (excess[bad] / np.maximum(le[bad], 1e-30) / mobility[bad])[:, None]
            move = np.column_stack([np.bincount(i, col, len(x)) - np.bincount(j, col, len(x)) for col in st.T])
            cnt = np.bincount(np.r_[i, j], minlength=len(x))
            x += 0.7 * move / np.maximum(cnt[:, None], 1)
            de = x - p
            x = p + de * np.minimum(1.0, cap / np.maximum(s.lengths(de), 1e-30))[:, None]
            x[g.fixed] = p[g.fixed]
        return x
    rows = []
    for top in (1.7, 1.5):
        for capratio in (0.04, 0.08):
            cap = np.maximum(0.021 * scale * support, capratio * scale * severe)
            cap[g.fixed] = 0
            cor = delta * np.minimum(1.0, cap / np.maximum(s.lengths(delta), 1e-30))[:, None]
            seed = p + cor
            seed += np.clip(g.smooth(s.lengths(delta)[:, None], 8) / (0.003 * scale), 0, 1) * (g.smooth(seed, max(12, steps // 3)) - seed)
            q = project(seed, np.minimum(top, np.maximum(ratio, 1)), np.maximum(0.27, np.minimum(ratio, 1)), cap, 300)
            m = s.metrics(q, f, g, rl, ra, sm)
            ar, _ = s.triangle_geometry(q, f)
            nc = int(((ar < 0.01 * ra) & (s.triangle_geometry(p, f)[0] >= 0.01 * ra)).sum())
            name = f'band-{top}-{capratio}'
            row = dict(name=name, metrics=m, newCollapse=nc, regressed=[k for k in before if m[k] > before[k] + 1e-06], vsBaseline=[k for k in base if m[k] > base[k] + 1e-06], maxMoveM=float(s.lengths(q - p).max()))
            rows.append(row)
            print(json.dumps(row), flush=True)
            np.savez_compressed(D / f'candidate-{name}.npz', vertices=q)
    (D / 'band-experiment.json').write_text(json.dumps(rows, indent=2))
    all_rows = []
    for name in ('experiment.json', 'arap-experiment.json', 'band-experiment.json'):
        data = json.loads((D / name).read_text())
        all_rows.extend(data['rows'] if isinstance(data, dict) else data)
    summary = {'inputStagesSha256': hashlib.sha256(stages.read_bytes()).hexdigest(), 'candidateCount': len(all_rows), 'automaticallyApplied': False, 'nonRegressingCandidates': [row['name'] for row in all_rows if not row['regressed'] and (not row['newCollapse'])]}
    (D / 'experiment-summary.json').write_text(json.dumps(summary, indent=2))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.stages.resolve(), args.output.resolve())
