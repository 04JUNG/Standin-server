"""Re-measure an SSR2 live-preview asset against the real controller (independent test set).

    python check_sampled_runtime.py --runtime runtime.ssr --blend model.blend --controller standin_body.py \
        --report check.json [--truncate-basis 160 --out smaller.ssr]

The builder already reports errors on its own validation set (used to prune pairs). This
draws a fresh test set with another seed, so the numbers are not tuned to it. With
``--truncate-basis`` the PCA basis is cut to its first K components (ordered by singular
value) to fit a size budget; the written file's header ``quality`` is replaced by this
check's numbers.
"""
import argparse, hashlib, importlib.util, json, struct, sys
from pathlib import Path

import numpy as np

BASIC_GUESS = ['height_cm', 'heads', 'physique', 'muscle', 'muscle_bulk', 'shoulders', 'fat_gain', 'breast_size']


def read_ssr2(path):
    raw = Path(path).read_bytes()
    if raw[:4] != b'SSR2':
        raise SystemExit('not an SSR2 runtime')
    size = struct.unpack('<I', raw[4:8])[0]
    header = json.loads(raw[8:8 + size])
    start = 8 + size
    arrays = {}
    for a in header['arrays']:
        dtype = {'float32': np.float32, 'uint16': np.uint16, 'int16': np.int16}[a['dtype']]
        arrays[a['name']] = np.frombuffer(raw, dtype=dtype, count=a['length'], offset=start + a['offset'])
    return header, arrays


def locate(grid, x):
    if x <= grid[0]:
        return 0, 0.0
    if x >= grid[-1]:
        return len(grid) - 2, 1.0
    i = int(np.searchsorted(grid, x) - 1)
    return i, (x - grid[i]) / (grid[i + 1] - grid[i])


def make_evaluator(header, arrays, K):
    nv = header['vertexCount']
    full = header['basisCount']
    basis = arrays['basis'].reshape(full, nv * 3)[:K].astype(np.float64)
    scale = arrays['basisScale'][:K].astype(np.float64)
    coef = arrays['coef'].reshape(header['coefCount'], full)[:, :K].astype(np.float64)
    mean = arrays['mean'].astype(np.float64)

    keys = list(header['controls'])
    ranges = np.array([header['controls'][k]['max'] - header['controls'][k]['min'] for k in keys])
    centers = {n: np.array([v[k] for k in keys]) for n, v in header['presets'].items()}

    def weights(preset, c):
        """Same rule as the app (shapeRuntime/sampled.ts modelWeights)."""
        sel = header.get('selection')
        if not sel:
            return [(preset, 1.0)]
        x = np.array([c[k] for k in keys])
        dist = {n: float((((x - p) / ranges) ** 2).sum()) for n, p in centers.items()}
        dmin = min(dist.values())
        if dmin < 1e-12:
            return [(min(dist, key=dist.get), 1.0)]
        cut = 1 / (sel['cutRatio'] ** 2 * dmin * dmin)
        w = {n: max(0.0, 1 / (d * d) - cut) for n, d in dist.items()}
        total = sum(w.values())
        return [(n, v / total) for n, v in w.items() if v > 0]

    def coefficients(preset, c):
        m = header['models'][preset]
        co = coef[m['base']].copy()
        for axis in m['axes']:
            i, t = locate(axis['knots'], c[axis['key']])
            co += coef[axis['offset'] + i] * (1 - t) + coef[axis['offset'] + i + 1] * t
        for pair in m['pairs']:
            i, u = locate(pair['ga'], c[pair['a']])
            j, v = locate(pair['gb'], c[pair['b']])
            nb = len(pair['gb'])
            at = lambda a, b: pair['offset'] + a * nb + b
            co += (coef[at(i, j)] * (1 - u) * (1 - v) + coef[at(i + 1, j)] * u * (1 - v)
                   + coef[at(i, j + 1)] * (1 - u) * v + coef[at(i + 1, j + 1)] * u * v)
        return co

    def evaluate(preset, c):
        co = sum(w * coefficients(name, c) for name, w in weights(preset, c))
        return (mean + (co * scale) @ basis).reshape(nv, 3)
    return evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runtime', required=True)
    ap.add_argument('--blend', required=True)
    ap.add_argument('--controller', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--control-spec', help='Published slider bounds; verify that subset explicitly')
    ap.add_argument('--truncate-basis', type=int, default=0)
    ap.add_argument('--out')
    ap.add_argument('--few', type=int, default=12)
    ap.add_argument('--basic', type=int, default=8)
    ap.add_argument('--all', type=int, default=4)
    ap.add_argument('--seed', type=int, default=99)
    ap.add_argument('--selection-cut-ratio', type=float, default=0, help='write/enable idw2 model blending (0 = keep header)')
    args = ap.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None)
    header, arrays = read_ssr2(args.runtime)
    published = json.loads(Path(args.control_spec).read_text()) if args.control_spec else header['controls']
    for key, path in (('blendSha256', args.blend), ('controllerSha256', args.controller)):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != header['source'][key]:
            raise SystemExit(f'{path} is not the source this runtime was built from')
    import bpy
    bpy.ops.wm.open_mainfile(filepath=args.blend)
    spec = importlib.util.spec_from_file_location('check_controller', args.controller)
    sb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sb)
    sb.register()
    if sb.VERSION != header['rules']:
        raise SystemExit('controller VERSION differs from runtime rules')
    K = args.truncate_basis or header['basisCount']
    if args.selection_cut_ratio:
        header['selection'] = {'mode': 'idw2', 'cutRatio': args.selection_cut_ratio}
    evaluate = make_evaluator(header, arrays, K)
    rng = np.random.default_rng(args.seed)
    keys = list(sb.FIELDS)
    basic = [k for k in BASIC_GUESS if k in sb.FIELDS]
    quality = {}
    for preset, values in header['presets'].items():
        per = {}
        compute = sb.preview_compute if hasattr(sb, 'preview_compute') else lambda c: sb.compute(c)[0]
        truth0 = compute(sb.normalize(values))
        exact = float(np.linalg.norm(evaluate(preset, values) - truth0, axis=1).max() * 1000)
        for mode, n in (('few', args.few), ('basic', args.basic), ('all', args.all)):
            for _ in range(n):
                c = dict(values)
                moved = keys if mode == 'all' else basic if mode == 'basic' else list(rng.choice(keys, int(rng.integers(2, 7)), replace=False))
                for k in moved:
                    lo, hi = published[k]['min'], published[k]['max']
                    x = rng.uniform(lo, hi) if mode != 'few' else np.clip(values[k] + rng.normal(0, .3) * (hi - lo), lo, hi)
                    c[k] = float(np.round(x, 3))
                truth = compute(sb.normalize(c))
                errors = np.linalg.norm(evaluate(preset, c) - truth, axis=1) * 1000
                error = float(errors.max())
                per.setdefault(mode, []).append(error)
                if error > 5:
                    print('ERROR_CASE', mode, round(error, 2), 'vertex', int(errors.argmax()),
                          json.dumps({k:v for k,v in c.items() if v != values[k]}), flush=True)
        quality[preset] = {'preset_mm': round(exact, 3), **{m: {'max_mm': round(max(v), 2), 'median_mm': round(float(np.median(v)), 2), 'p90_mm': round(float(np.percentile(v, 90)), 2), 'n': len(v)} for m, v in per.items()}}
        print('CHECK', preset, json.dumps(quality[preset]), flush=True)
    report = {'runtime': args.runtime, 'rules': header['rules'], 'basis': K, 'seed': args.seed, 'quality': quality, 'publishedControls': published}
    limits = {'preset_mm': .5, 'few': 5., 'basic': 10., 'all': 20.}
    report['limits_mm'] = limits
    report['passed'] = all(q['preset_mm'] <= limits['preset_mm'] and
        all(q[mode]['max_mm'] <= limits[mode] for mode in ('few', 'basic', 'all')) for q in quality.values())
    if args.out:
        nv = header['vertexCount']
        full = header['basisCount']
        coef = arrays['coef'].reshape(header['coefCount'], full)[:, :K]
        basis = arrays['basis'].reshape(full, nv * 3)[:K]
        new_header = {k: v for k, v in header.items() if k != 'arrays'}
        new_header['basisCount'] = K
        new_header['quality'] = {'independentCheck': {'seed': args.seed, 'presets': quality}}
        sys.path.insert(0, str(Path(__file__).parent))
        from build_sampled_runtime import write_ssr2
        write_ssr2(Path(args.out), new_header, [
            ('faces', 'uint16', arrays['faces']), ('mean', 'float32', arrays['mean']),
            ('basisScale', 'float32', arrays['basisScale'][:K]), ('coef', 'float32', np.ascontiguousarray(coef).reshape(-1)),
            ('basis', 'int16', np.ascontiguousarray(basis).reshape(-1))])
        report['out'] = {'path': args.out, 'bytes': Path(args.out).stat().st_size,
                         'sha256': hashlib.sha256(Path(args.out).read_bytes()).hexdigest()}
        print('WROTE', json.dumps(report['out']), flush=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
