"""Build a version-agnostic live-preview asset (`standin-shape-runtime.v2`, "SSR2").

Run with Blender's Python (pip bpy works) — the controller itself is executed, so nothing
about a body version is ported by hand:

    python build_sampled_runtime.py --blend <model>.blend --controller standin_body.py \
        --out runtime-<rules>.ssr --report runtime-<rules>.json

For every preset the controller's neutral `compute()` is sampled:
  1. each control alone, on an adaptive knot grid (linear interpolation error <= --axis-tol-mm);
  2. control pairs that interact strongly (screened at their far corners), on a 3x3 grid;
     pairs that do not improve a held-out validation set are pruned.
All samples share one PCA basis (randomized SVD, int16 + per-basis scale). The app evaluates
    position = mean + sum_b coef_b * scale_b * basis_b,
    coef     = base[preset] + sum_axes lerp + sum_pairs bilerp
and never needs the controller code. The header names the authoring source (blend and
controller SHA-256); the catalog/BFF/converter must agree on it or the app falls back to images.

The report records the error against the real controller on validation sets per preset:
  few   : 2-6 random controls moved around the preset (typical editing),
  basic : every basic slider random over its full range,
  all   : every control random over its full range (stress; not a typical edit).
"""
import argparse, hashlib, importlib.util, itertools, json, math, struct, sys, tempfile, time
from pathlib import Path

import numpy as np
from types import SimpleNamespace
from functools import lru_cache

FORMAT = 'standin-shape-runtime.v2'
MAGIC = b'SSR2'
BASIC_GUESS = ['height_cm', 'heads', 'physique', 'muscle', 'muscle_bulk', 'shoulders', 'fat_gain', 'breast_size']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(blend, controller):
    import bpy
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    spec = importlib.util.spec_from_file_location('runtime_controller', str(controller))
    sb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sb)
    sb.register()
    return sb


class Sampler:
    def __init__(self, sb):
        self.sb = sb
        self.mesh, _ = sb.objects()
        self.evals = 0

    def __call__(self, c):
        return self._sample(tuple(sorted(c.items())))

    @lru_cache(maxsize=64)
    def _sample(self, values):
        self.evals += 1
        if self.evals % 32 == 0:
            print('SAMPLING', self.evals, flush=True)
        c = dict(values)
        return (self.sb.preview_compute(c) if hasattr(self.sb, 'preview_compute') else self.sb.compute(c)[0]).astype(np.float64)


def lerp_axis(knots, rows, x):
    j = int(np.clip(np.searchsorted(knots, x) - 1, 0, len(knots) - 2))
    t = (x - knots[j]) / (knots[j + 1] - knots[j])
    t = min(max(t, 0.0), 1.0)
    return rows[j] * (1 - t) + rows[j + 1] * t


def locate(grid, x):
    if x <= grid[0]:
        return 0, 0.0
    if x >= grid[-1]:
        return len(grid) - 2, 1.0
    i = int(np.searchsorted(grid, x) - 1)
    return i, (x - grid[i]) / (grid[i + 1] - grid[i])


def bilerp(ga, gb, table, x, y):
    i, u = locate(ga, x)
    j, v = locate(gb, y)
    return (table[i, j] * (1 - u) * (1 - v) + table[i + 1, j] * u * (1 - v)
            + table[i, j + 1] * (1 - u) * v + table[i + 1, j + 1] * u * v)


def max_mm(a, b):
    return float(np.linalg.norm(a - b, axis=1).max() * 1000)


def sample_preset(V, sb, name, args, rng):
    fields = sb.FIELDS
    keys = list(fields)
    base = sb.normalize(sb.preset_values(name, V.mesh))
    VP = V(base)
    tol = args.axis_tol_mm / 1000
    axes = {}
    for k in keys:
        lo, hi = fields[k][:2]
        knots = sorted(set(np.round(np.linspace(lo, hi, args.axis_knots), 6).tolist()) | {round(base[k], 6)})
        rows = {x: V({**base, k: x}) - VP for x in knots}
        # Refine where linear interpolation misses the controller (smoothsteps, folds).
        for _ in range(args.axis_refine):
            added = False
            ks = sorted(rows)
            for a, b in zip(ks, ks[1:]):
                if b - a < (hi - lo) / 256:
                    continue
                mid = round((a + b) / 2, 6)
                truth = V({**base, k: mid}) - VP
                if np.linalg.norm(truth - (rows[a] + rows[b]) / 2, axis=1).max() > tol:
                    rows[mid] = truth
                    added = True
            if not added:
                break
        ks = sorted(rows)
        axes[k] = (np.array(ks), np.stack([rows[x] for x in ks]))

    def axis_sum(c):
        out = np.zeros_like(VP)
        for k, (kn, rows) in axes.items():
            out += lerp_axis(kn, rows, c[k])
        return out

    far = {k: (fields[k][0] if abs(fields[k][0] - base[k]) > abs(fields[k][1] - base[k]) else fields[k][1]) for k in keys}
    score = {}
    for a, b in itertools.combinations(keys, 2):
        if far[a] == base[a] or far[b] == base[b]:
            continue
        c = {**base, a: far[a], b: far[b]}
        score[(a, b)] = float(np.linalg.norm(V(c) - VP - axis_sum(c), axis=1).max())
    candidates = [p for p in sorted(score, key=lambda p: -score[p]) if score[p] * 1000 > args.pair_min_mm][:args.pair_budget]
    pairs = {}

    def nodes(k, dense):
        lo, hi = fields[k][:2]
        grid = {lo, round(base[k], 6), hi}
        if dense:  # strongest pairs: quartiles too (kinks such as "muscle > 2 strengthens")
            grid |= {round(x, 6) for x in np.linspace(lo, hi, args.pair_dense_nodes)}
        while len(grid) < 3:  # preset at an end: add the midpoint of the widest gap
            g = sorted(grid)
            i = int(np.argmax(np.diff(g)))
            grid.add(round((g[i] + g[i + 1]) / 2, 6))
        return sorted(grid)
    for rank, (a, b) in enumerate(candidates):
        dense = rank < args.pair_dense
        ga, gb = nodes(a, dense), nodes(b, dense)
        table = np.zeros((len(ga), len(gb)) + VP.shape)
        for i, x in enumerate(ga):
            for j, y in enumerate(gb):
                if x == round(base[a], 6) or y == round(base[b], 6):
                    continue
                c = {**base, a: x, b: y}
                table[i, j] = V(c) - VP - axis_sum(c)
        pairs[(a, b)] = (np.array(ga), np.array(gb), table)

    basic = [k for k in BASIC_GUESS if k in fields]
    validation = []
    for mode, count in (('few', args.val_few), ('basic', args.val_basic), ('all', args.val_all)):
        for _ in range(count):
            c = dict(base)
            moved = keys if mode == 'all' else basic if mode == 'basic' else list(rng.choice(keys, int(rng.integers(2, 7)), replace=False))
            for k in moved:
                lo, hi = fields[k][:2]
                x = rng.uniform(lo, hi) if mode != 'few' else np.clip(base[k] + rng.normal(0, .3) * (hi - lo), lo, hi)
                c[k] = float(np.round(x, 3))
            validation.append((mode, c, V(c)))

    # Backward pruning: drop pairs whose correction does not help the held-out set.
    def contrib(p, c):
        ga, gb, table = pairs[p]
        return bilerp(ga, gb, table, c[p[0]], c[p[1]])
    preds = [VP + axis_sum(c) for _, c, _ in validation]
    contribs = {p: [contrib(p, c) for _, c, _ in validation] for p in pairs}
    full = [preds[i] + sum(contribs[p][i] for p in pairs) for i in range(len(validation))]
    weight = {'few': 1.0, 'basic': 1.0, 'all': 0.25}

    def objective(current):
        per = {}
        for i, (mode, _, truth) in enumerate(validation):
            per.setdefault(mode, []).append(np.linalg.norm(current[i] - truth, axis=1).max())
        return sum(weight[m] * float(np.mean(v)) for m, v in per.items())
    kept = set(pairs)
    best = objective(full)
    improved = True
    while improved and kept:
        improved = False
        for p in sorted(kept, key=lambda p: score[p]):
            trial = [full[i] - contribs[p][i] for i in range(len(validation))]
            value = objective(trial)
            if value < best - 1e-7:
                best, full = value, trial
                kept.discard(p)
                improved = True
    pairs = {p: pairs[p] for p in candidates if p in kept}
    validation = [(m, c, truth.astype(np.float32)) for m, c, truth in validation]
    return {'name': name, 'values': base, 'VP': VP, 'axes': axes, 'pairs': pairs,
            'validation': validation, 'screened': len(score), 'candidates': len(candidates)}


def randomized_basis(rows_iter_factory, dim, rank, oversample=32, power=1, seed=0):
    """Row-space basis of the stacked sample matrix without holding it in memory.

    Peak memory stays around one preset's float32 rows plus a few dim x rank matrices,
    so two builds (male + female) can run side by side on an 8 GB machine.
    """
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((dim, rank + oversample)).astype(np.float32)
    Y = np.concatenate([X @ omega for X in rows_iter_factory()])
    print('BASIS_STAGE random_projection', flush=True)
    for _ in range(power):
        Q, _ = np.linalg.qr(Y)
        Z = np.zeros((dim, Q.shape[1]), dtype=np.float32)
        start = 0
        for X in rows_iter_factory():
            Z += X.T @ Q[start:start + len(X)]
            start += len(X)
        Q2, _ = np.linalg.qr(Z)
        print('BASIS_STAGE power_qr', flush=True)
        Y = np.concatenate([X @ Q2.astype(np.float32) for X in rows_iter_factory()])
    Q, _ = np.linalg.qr(Y)
    B = np.zeros((Q.shape[1], dim), dtype=np.float32)
    start = 0
    for X in rows_iter_factory():
        B += Q[start:start + len(X)].T @ X
        start += len(X)
    print('BASIS_STAGE compressed_matrix', flush=True)
    # B has only ~124 rows and 168k columns. Diagonalize its small Gram
    # matrix rather than asking LAPACK to copy the entire wide matrix.
    gram = B @ B.T
    eigen, left = np.linalg.eigh(gram.astype(np.float64))
    order = np.argsort(eigen)[::-1]
    s = np.sqrt(np.maximum(eigen[order], 1e-20))
    vt = (left[:, order].T @ B) / s[:, None]
    print('BASIS_STAGE complete', flush=True)
    return vt, s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--blend', required=True)
    ap.add_argument('--controller', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--presets', help='comma list; default all controller presets')
    ap.add_argument('--axis-knots', type=int, default=9)
    ap.add_argument('--axis-refine', type=int, default=2)
    ap.add_argument('--axis-tol-mm', type=float, default=0.3)
    ap.add_argument('--pair-budget', type=int, default=60)
    ap.add_argument('--pair-min-mm', type=float, default=1.0)
    ap.add_argument('--pair-dense', type=int, default=20, help='strongest pairs sampled on a denser grid')
    ap.add_argument('--pair-dense-nodes', type=int, default=5)
    ap.add_argument('--val-few', type=int, default=16)
    ap.add_argument('--val-basic', type=int, default=10)
    ap.add_argument('--val-all', type=int, default=6)
    ap.add_argument('--basis', type=int, default=0, help='fixed basis size; 0 = smallest meeting --basis-tol-mm')
    ap.add_argument('--basis-tol-mm', type=float, default=0.25)
    ap.add_argument('--max-basis', type=int, default=224)
    ap.add_argument('--seed', type=int, default=20261016)
    ap.add_argument('--selection-cut-ratio', type=float, default=2.0)
    ap.add_argument('--state-dir', help='keep per-preset samples here (default <out>.state); reruns resume')
    ap.add_argument('--metadata', help='Compress completed samples with host NumPy, without Blender')
    ap.add_argument('--export-metadata', help='Export Blender metadata, then exit without resampling')
    args = ap.parse_args(argv)
    t0 = time.time()
    source = {'blendSha256': sha(args.blend), 'controllerSha256': sha(args.controller)}
    if args.metadata:
        info = json.loads(Path(args.metadata).read_text())
        if info['source'] != source: raise ValueError('metadata source mismatch')
        rules, nv, names, controls, asset = (info[k] for k in ('rules','nv','names','controls','asset'))
        faces = np.load(str(args.metadata)+'.faces.npy')
        sb, V = None, SimpleNamespace(evals=0)
    else:
        sb = load(args.blend, args.controller)
        rules = sb.VERSION
        V = Sampler(sb)
        mesh = V.mesh
        preview = sb.preview_mesh() if hasattr(sb, 'preview_mesh') else mesh
        polys = [p.vertices[:] for p in preview.data.polygons]
        if any(len(p) != 3 for p in polys): raise SystemExit('runtime faces must be triangles')
        faces = np.array(polys, dtype=np.uint16)
        nv = len(preview.data.vertices)
        names = args.presets.split(',') if args.presets else list(sb.PRESETS)
        controls = {k: {'min': float(v[0]), 'max': float(v[1]), 'default': float(sb.defaults(mesh)[k])} for k,v in sb.FIELDS.items()}
        asset = str(mesh['asset_identity'])
        if args.export_metadata:
            Path(args.export_metadata).write_text(json.dumps(dict(source=source,rules=rules,nv=nv,names=names,controls=controls,asset=asset)))
            np.save(args.export_metadata+'.faces.npy',faces)
            print('METADATA_READY',flush=True)
            return
    if nv > 65535:
        raise SystemExit('too many vertices for u16 faces')
    rng = np.random.default_rng(args.seed)  # fixture vertex choice only; presets use their own streams
    # Sampling is the slow part (about an hour per body on 2 vCPU). Each preset's samples are
    # kept in --state-dir so a rerun (crash, OOM, a tweak to PCA settings) resumes from disk.
    state = Path(args.state_dir or (args.out + '.state'))
    state.mkdir(parents=True, exist_ok=True)
    stamp = {'rules': rules, 'source': source, 'axis_knots': args.axis_knots, 'axis_refine': args.axis_refine,
             'axis_tol_mm': args.axis_tol_mm, 'pair_budget': args.pair_budget, 'pair_min_mm': args.pair_min_mm,
             'pair_dense': args.pair_dense, 'pair_dense_nodes': args.pair_dense_nodes, 'seed': args.seed,
             'val': [args.val_few, args.val_basic, args.val_all]}
    presets = []
    for index, name in enumerate(names):
        folder = state / f'{index:02d}-{name}'
        meta_path = folder / 'meta.json'
        if meta_path.exists() and json.loads(meta_path.read_text()).get('stamp') == stamp:
            meta = json.loads(meta_path.read_text())
            truths = np.load(folder / 'validation.npy')
            p = {'name': name, 'values': meta['values'], 'VP': np.load(folder / 'VP.npy'),
                 'axes': [(k, np.array(kn), n) for k, kn, n in meta['axes']],
                 'pairs': [(a, b, np.array(ga), np.array(gb)) for a, b, ga, gb in meta['pairs']],
                 'validation': [(m, c, truths[i]) for i, (m, c) in enumerate(meta['validation'])],
                 'evals': meta['evals'], 'seconds': meta['seconds'], 'candidates': meta['candidates'],
                 'rows_file': folder / 'rows.npy'}
            presets.append(p)
            print('RESUMED', name, flush=True)
            continue
        ts = time.time()
        if sb is None: raise ValueError('Matching completed samples required for host compression')
        before = V.evals
        p = sample_preset(V, sb, name, args, np.random.default_rng([args.seed, index]))
        p['evals'] = V.evals - before
        p['seconds'] = round(time.time() - ts, 1)
        # Keep only layout in memory; the sample rows go to disk (float32) for the PCA passes.
        rows = [r for _, (_, rr) in p['axes'].items() for r in rr]
        rows += [r for (_, _, table) in p['pairs'].values() for r in table.reshape(-1, nv, 3)]
        folder.mkdir(exist_ok=True)
        np.save(folder / 'rows.npy', np.stack([r.reshape(-1) for r in rows]).astype(np.float32))
        np.save(folder / 'VP.npy', p['VP'])
        np.save(folder / 'validation.npy', np.stack([truth for _, _, truth in p['validation']]))
        p['rows_file'] = folder / 'rows.npy'
        p['axes'] = [(k, kn, len(rr)) for k, (kn, rr) in p['axes'].items()]
        p['pairs'] = [(a, b, ga, gb) for (a, b), (ga, gb, _) in p['pairs'].items()]
        meta_path.write_text(json.dumps({
            'stamp': stamp, 'values': p['values'], 'evals': p['evals'], 'seconds': p['seconds'], 'candidates': p['candidates'],
            'axes': [(k, [float(x) for x in kn], n) for k, kn, n in p['axes']],
            'pairs': [(a, b, [float(x) for x in ga], [float(x) for x in gb]) for a, b, ga, gb in p['pairs']],
            'validation': [(m, c) for m, c, _ in p['validation']]}))
        presets.append(p)
        print('SAMPLED', name, 'evals', p['evals'], 'pairs', len(p['pairs']), 'of', p['candidates'], 's', p['seconds'], flush=True)

    mean = np.mean([p['VP'] for p in presets], axis=0)
    bodies = np.stack([(p['VP'] - mean).reshape(-1) for p in presets]).astype(np.float32)

    def rows_iter():
        yield bodies
        for p in presets:
            yield np.load(p['rows_file'], mmap_mode='r').astype(np.float32)
    dim = nv * 3
    vt, singular = randomized_basis(rows_iter, dim, args.max_basis, seed=args.seed)

    def reconstruct_error(K):
        # float32 and row chunks: a full float64 residual of one preset is >1 GB.
        worst = 0.0
        B = vt[:K].astype(np.float32)
        for X in rows_iter():
            for s in range(0, len(X), 256):
                x = np.asarray(X[s:s + 256], dtype=np.float32)
                R = x - (x @ B.T) @ B
                worst = max(worst, float(np.linalg.norm(R.reshape(len(x), nv, 3), axis=2).max()))
        return worst * 1000
    if args.basis:
        K = args.basis
    else:
        K = None
        for trial in (64, 96, 128, 160, 192, args.max_basis):
            if trial > args.max_basis:
                continue
            err = reconstruct_error(trial)
            print('BASIS', trial, 'max_row_error_mm', round(err, 3), flush=True)
            if err <= args.basis_tol_mm:
                K = trial
                break
        K = K or args.max_basis
    basis = vt[:K]
    scale = np.abs(basis).max(axis=1) / 32767.0
    scale[scale == 0] = 1.0
    basis_q = np.round(basis / scale[:, None]).astype(np.int16)
    basis_deq = basis_q.astype(np.float64) * scale[:, None]

    # Coefficient table and model layout (project with the quantized basis the app uses).
    # Quantized orthonormal rows have a well-conditioned small Gram matrix.
    # Avoid an SVD of a 168k-by-92 matrix on the authoring workstation.
    proj = np.linalg.solve(basis_deq @ basis_deq.T, basis_deq)
    coef_rows, models = [], {}
    def add(vec):
        coef_rows.append(proj @ vec.reshape(-1))
        return len(coef_rows) - 1
    for idx, p in enumerate(presets):
        model = {'base': add(bodies[idx]), 'axes': [], 'pairs': []}
        stored = np.load(p['rows_file'], mmap_mode='r')
        cursor = 0
        for k, kn, n in p['axes']:
            offset = len(coef_rows)
            for r in stored[cursor:cursor + n]:
                add(np.asarray(r, dtype=np.float64))
            cursor += n
            model['axes'].append({'key': k, 'knots': [float(x) for x in kn], 'offset': offset})
        for a, b, ga, gb in p['pairs']:
            offset = len(coef_rows)
            for r in stored[cursor:cursor + len(ga) * len(gb)]:
                add(np.asarray(r, dtype=np.float64))
            cursor += len(ga) * len(gb)
            model['pairs'].append({'a': a, 'b': b, 'ga': [float(x) for x in ga], 'gb': [float(x) for x in gb], 'offset': offset})
        models[p['name']] = model
    coef = np.stack(coef_rows).astype(np.float32)

    header = {'format': FORMAT, 'rules': rules, 'source': source, 'asset': asset,
              'vertexCount': nv, 'faceCount': len(faces), 'basisCount': K, 'coefCount': len(coef),
              'controls': controls, 'aliases': {'height': 'height_cm'},
              'presets': {p['name']: {k: float(v) for k, v in p['values'].items()} for p in presets},
              'models': models,
              # Blend nearby preset models (same rule in the app and check_sampled_runtime.py).
              'selection': {'mode': 'idw2', 'cutRatio': args.selection_cut_ratio}}
    blobs = [('faces', 'uint16', faces.reshape(-1)), ('mean', 'float32', mean.reshape(-1).astype(np.float32)),
             ('basisScale', 'float32', scale.astype(np.float32)), ('coef', 'float32', coef.reshape(-1)),
             ('basis', 'int16', basis_q.reshape(-1))]
    write_ssr2(Path(args.out), header, blobs)
    # Measure exactly what ships: read the file back and evaluate with the app's rule.
    sys.path.insert(0, str(Path(__file__).parent))
    from check_sampled_runtime import make_evaluator, read_ssr2
    shipped, arrays = read_ssr2(args.out)
    evaluate = make_evaluator(shipped, arrays, K)
    quality = {}
    for p in presets:
        per = {}
        for mode, c, truth in p['validation']:
            per.setdefault(mode, []).append(max_mm(evaluate(p['name'], c), truth))
        exact = max_mm(evaluate(p['name'], p['values']), p['VP'])
        quality[p['name']] = {'preset_mm': round(exact, 3), **{m: {'max_mm': round(max(v), 2), 'median_mm': round(float(np.median(v)), 2), 'n': len(v)} for m, v in per.items()},
                              'evals': p['evals'], 'pairs': len(p['pairs']), 'pair_candidates': p['candidates']}
        print('QUALITY', p['name'], json.dumps(quality[p['name']]), flush=True)
    header['quality'] = {'basisMaxRowErrorMm': round(reconstruct_error(K), 3), 'presets': quality,
                         'note': 'validation set also used to prune pairs; see check_sampled_runtime.py for an independent set'}
    write_ssr2(Path(args.out), header, blobs)
    report = {'rules': rules, 'source': source, 'basis': K, 'bytes': Path(args.out).stat().st_size,
              'sha256': sha(args.out), 'seconds': round(time.time() - t0), 'evals': V.evals,
              'singular_values_head': [round(float(x), 6) for x in singular[:8]], 'quality': header['quality']}
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('SSR2_READY', args.out, report['bytes'], report['sha256'], 'basis', K, flush=True)



def write_ssr2(path, header, blobs):
    arrays, offset, data = [], 0, []
    for name, dtype, values in blobs:
        raw = np.ascontiguousarray(values).tobytes()
        arrays.append({'name': name, 'dtype': dtype, 'offset': offset, 'length': int(values.size)})
        pad = (-len(raw)) % 4
        data.append(raw + b'\0' * pad)
        offset += len(raw) + pad
    header = {**header, 'arrays': arrays}
    text = json.dumps(header, ensure_ascii=False, separators=(',', ':')).encode()
    text += b' ' * ((-len(text)) % 4)
    path.write_bytes(MAGIC + struct.pack('<I', len(text)) + text + b''.join(data))


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None)
