"""Offline rest-detail-preserving local rigid fit of the skin motion field.

This is an experimental least-squares fit, not an implementation/qualification
of the Direct Delta Mush paper. Source vertices/weights and rig are read-only.
"""
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from converter.local import surface as s

def fit(graph, rest, posed, radius):
    r = rest - rest.mean(0)
    p = posed - posed.mean(0)
    field = np.column_stack((r, p, (p[:, :, None] * r[:, None, :]).reshape(-1, 9)))
    avg = graph.diffuse(field, radius)
    a, b = (avg[:, :3], avg[:, 3:6])
    cov = avg[:, 6:].reshape(-1, 3, 3) - b[:, :, None] * a[:, None, :]
    u, sigma, vt = np.linalg.svd(cov)
    u[:, :, -1] *= np.sign(np.linalg.det(u @ vt))[:, None]
    rotation = u @ vt
    return b + np.einsum('nij,nj->ni', rotation, r - a) + posed.mean(0)

def run(base, out, geometric_pins=True, female_cmu_only=False):
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for mi in (1,) if female_cmu_only else range(3):
        for pi in (1,) if female_cmu_only else range(2):
            path = base / f'stages-{mi}-{pi}.npz'
            data = np.load(path)
            r, p, f = (data['rest'], data['baked'], data['triangles'])
            g = s.Neighborhood(f, r)
            rl = s.lengths(r[g.b] - r[g.a])
            ra, n = s.triangle_geometry(r, f)
            i, j = g.face_pairs.T
            al = (n[i] * n[j]).sum(1)
            sm = al > np.cos(np.pi / 6)
            g.fixed |= data['anchors']
            if geometric_pins:
                g.fixed[f[g.face_pairs[al < np.cos(np.pi / 3)]].ravel()] = True
            support = s.damage_support(g, r, p, f, rl, ra, al)
            met = lambda q: s.metrics(q, f, g, rl, ra, sm)
            before = met(p)
            baseline = met(data['corrected'])
            scale = float(np.ptp(r, axis=0).max())
            variants = []
            for radius in (0.01, 0.03, 0.06):
                try:
                    target = fit(g, r, p, radius * scale)
                except (ArithmeticError, np.linalg.LinAlgError) as e:
                    variants.append({'label': str(radius), 'error': type(e).__name__})
                    continue
                for strength in (0.25, 0.5, 1.0):
                    delta = (target - p) * support[:, None] * strength
                    cap = 0.08 * scale * support
                    delta *= np.minimum(1.0, cap / np.maximum(s.lengths(delta), 1e-30))[:, None]
                    q = p + delta
                    q[support == 0] = p[support == 0]
                    m = met(q)
                    areas, _ = s.triangle_geometry(q, f)
                    newcollapse = int(((areas < 0.01 * ra) & (s.triangle_geometry(p, f)[0] >= 0.01 * ra)).sum())
                    label = f'{radius}-{strength}'
                    variants.append(dict(label=label, metrics=m, newCollapsedFaces=newcollapse, outsideMaxMoveM=float(s.lengths(q[support == 0] - p[support == 0]).max()), maxMoveM=float(s.lengths(q - p).max()), regressedRaw=[k for k in before if m[k] > before[k] + 1e-06], regressedV10=[k for k in baseline if m[k] > baseline[k] + 1e-06]))
                    np.savez_compressed(out / f'{mi}-{pi}-{label}.npz', points=q)
            rows.append(dict(model=mi, pose=pi, inputStagesSha256=hashlib.sha256(path.read_bytes()).hexdigest(), before=before, v10=baseline, variants=variants))
            (out / 'motion-fit-results.json').write_text(json.dumps(rows, indent=2))
            print('LOCAL_MOTION_STAGE', mi, pi, [(v['label'], round(v.get('metrics', {}).get('maxStretch', 0), 2), v.get('regressedV10')) for v in variants], flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--allow-geometric-crease-motion', action='store_true')
    p.add_argument('--female-cmu-only', action='store_true')
    a = p.parse_args()
    run(a.base, a.out, not a.allow_geometric_crease_motion, a.female_cmu_only)
