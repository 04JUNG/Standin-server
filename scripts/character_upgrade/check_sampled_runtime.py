"""Re-measure an SSR2 live-preview asset against the real controller (independent test set).

    python check_sampled_runtime.py --runtime runtime.ssr --blend model.blend --controller standin_body.py \
        --report check.json [--truncate-basis 160 --out smaller.ssr]

The builder already reports errors on its own validation set (used to prune pairs). This
draws a fresh test set with another seed, so the numbers are not tuned to it. With
``--truncate-basis`` the PCA basis is cut to its first K components (ordered by singular
value) to fit a size budget; the written file's header ``quality`` is replaced by this
check's numbers.
"""
import argparse, hashlib, importlib.util, itertools, json, math, struct, sys
from decimal import Decimal
from pathlib import Path

import numpy as np

BASIC_GUESS = ['height_cm', 'heads', 'physique', 'muscle', 'muscle_bulk', 'shoulders', 'fat_gain', 'breast_size']


def catalog_controls(path, expected_sha256, header, runtime_identity=None):
    """Test every exposed preset over its actual, hash-pinned slider domain."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('published catalog checksum mismatch')
    source = {'rules': header['rules'], **header['source']}
    presets = {}
    for asset in json.loads(raw)['assets']:
        if asset.get('shapeSource') != source:
            continue
        runtime = asset['shapeRuntime']
        if runtime['format'] != 'standin-shape-runtime.v2' or runtime['rules'] != header['rules']:
            raise ValueError('published runtime format or rules differ')
        if runtime_identity and any(runtime[k] != v for k,v in runtime_identity.items()):
            raise ValueError('published runtime checksum or size differs from tested bytes')
        name = runtime['preset']
        controls = {('height_cm' if c['key'] == 'height' else c['key']):
                    {k: c[k] for k in ('min', 'max', 'step', 'default')} for c in asset['shape']['controls']}
        if (name in presets or set(controls) != set(header['controls'])
                or len(controls) != len(asset['shape']['controls'])):
            raise ValueError('incomplete or duplicate published controls')
        for key, bounds in controls.items():
            allowed = header['controls'][key]
            if not allowed['min'] <= bounds['min'] < bounds['max'] <= allowed['max']:
                raise ValueError('published controls escape runtime domain')
            if (not math.isfinite(bounds['step']) or bounds['step'] <= 0
                    or bounds['default'] != header['presets'][name][key]):
                raise ValueError('published preset defaults or slider step differ')
        presets[name] = controls
    if set(presets) != set(header['presets']):
        raise ValueError('published preset coverage differs from runtime')
    return presets


def published_value(bounds, value):
    """Use the actual public slider grid; decimal rounding must not escape its bounds."""
    lo, hi, step = (Decimal(str(bounds[k])) for k in ('min', 'max', 'step'))
    last = int((hi - lo) // step)
    index = max(0, min(last, int(round((Decimal(str(value)) - lo) / step))))
    return float(lo + index * step)


def verify_geometry(header, arrays, preview):
    nv = len(preview.data.vertices)
    faces = [p.vertices[:] for p in preview.data.polygons]
    if (nv > 65535 or header['vertexCount'] != nv or header['faceCount'] != len(faces)
            or any(len(p) != 3 for p in faces)):
        raise ValueError('runtime vertex/face counts differ from final source')
    expected = np.asarray(faces, dtype=np.uint16).reshape(-1)
    if not np.array_equal(arrays['faces'], expected) or (len(expected) and int(expected.max()) >= nv):
        raise ValueError('runtime indices differ from final source')
    sizes = {'faces': len(faces) * 3, 'mean': nv * 3, 'basisScale': header['basisCount'],
             'basis': header['basisCount'] * nv * 3, 'coef': header['coefCount'] * header['basisCount']}
    if set(arrays) != set(sizes) or any(len(arrays[k]) != n or not np.isfinite(arrays[k]).all() for k,n in sizes.items()):
        raise ValueError('runtime array shape or finite-value check failed')
    return {'vertexCount': nv, 'faceCount': len(faces), 'indicesMatchFinalSource': True,
            'facesSha256': hashlib.sha256(expected.tobytes()).hexdigest()}


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

    selection = header.get('selection')
    if selection and selection.get('mode') not in ('idw2', 'idw2-varying', 'idw2-varying-trilinear', 'idw2-varying-multilinear'):
        raise ValueError('unsupported sampled-runtime model selection')
    if any(m.get('triples') for m in header['models'].values()) and (not selection or selection['mode'] not in ('idw2-varying-trilinear','idw2-varying-multilinear')):
        raise ValueError('three-control terms require explicit trilinear mode')
    if any(m.get('quads') for m in header['models'].values()) and (not selection or selection['mode'] != 'idw2-varying-multilinear'):
        raise ValueError('four-control terms require explicit multilinear mode')
    explicit_keys = selection.get('distanceKeys') if selection else None
    if selection and 'distanceKeys' in selection:
        if (selection['mode'] != 'idw2-varying-multilinear' or not isinstance(explicit_keys, list)
                or not 1 <= len(explicit_keys) <= 16
                or any(not isinstance(key,str) or key not in header['controls'] for key in explicit_keys)
                or len(set(explicit_keys)) != len(explicit_keys) or selection['cutRatio']<=1):
            raise ValueError('invalid explicit selection dimensions')
        centers = {tuple(p[key] for key in explicit_keys) for p in header['presets'].values()}
        if len(centers) != len(header['presets']):
            raise ValueError('selection dimensions do not distinguish presets')
    keys = explicit_keys if explicit_keys is not None else list(header['controls'])
    if selection and selection['mode'] in ('idw2-varying', 'idw2-varying-trilinear', 'idw2-varying-multilinear'):
        # Common offsets cannot distinguish preset centers. They still affect
        # every axis/pair below; only the model-selection distance omits them.
        keys = [k for k in keys if len({v[k] for v in header['presets'].values()}) > 1]
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
        for triple in m.get('triples', []):
            indices, fractions = zip(*(locate(triple[g], c[triple[k]]) for k,g in [('a','ga'),('b','gb'),('c','gc')]))
            for ia in (0,1):
                for ib in (0,1):
                    for ic in (0,1):
                        row=triple['offset']+((indices[0]+ia)*len(triple['gb'])+indices[1]+ib)*len(triple['gc'])+indices[2]+ic
                        weight=(fractions[0] if ia else 1-fractions[0])*(fractions[1] if ib else 1-fractions[1])*(fractions[2] if ic else 1-fractions[2])
                        co += coef[row]*weight
        for quad in m.get('quads', []):
            grids=[quad[k] for k in ('ga','gb','gc','gd')]
            located=[locate(grid,c[quad[key]]) for grid,key in zip(grids,('a','b','c','d'))]
            for corner in itertools.product((0,1),repeat=4):
                row=0;weight=1.
                for axis,upper in enumerate(corner):
                    index,fraction=located[axis];row=row*len(grids[axis])+index+upper
                    weight*=fraction if upper else 1-fraction
                co+=coef[quad['offset']+row]*weight
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
    ap.add_argument('--catalog', help='Published body-models.json with exact source/runtime bindings')
    ap.add_argument('--catalog-sha256', help='Required approval pin with --catalog')
    ap.add_argument('--truncate-basis', type=int, default=0)
    ap.add_argument('--out')
    ap.add_argument('--few', type=int, default=12)
    ap.add_argument('--basic', type=int, default=8)
    ap.add_argument('--all', type=int, default=4)
    ap.add_argument('--seed', type=int, default=99)
    ap.add_argument('--selection-cut-ratio', type=float, default=0, help='write/enable idw2 model blending (0 = keep header)')
    args = ap.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None)
    header, arrays = read_ssr2(args.runtime)
    if args.catalog and (args.control_spec or not args.catalog_sha256):
        ap.error('--catalog requires --catalog-sha256 and excludes --control-spec')
    published = json.loads(Path(args.control_spec).read_text()) if args.control_spec else header['controls']
    runtime_path = Path(args.runtime)
    runtime_identity = None if args.out else {
        'sha256': hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
        'bytes': runtime_path.stat().st_size}
    per_preset = catalog_controls(args.catalog, args.catalog_sha256, header, runtime_identity) if args.catalog else None
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
    geometry = verify_geometry(header, arrays, sb.preview_mesh() if hasattr(sb, 'preview_mesh') else sb.objects()[0])
    K = args.truncate_basis or header['basisCount']
    if args.selection_cut_ratio:
        header['selection'] = {'mode': 'idw2', 'cutRatio': args.selection_cut_ratio}
    evaluate = make_evaluator(header, arrays, K)
    rng = np.random.default_rng(args.seed)
    keys = list(sb.FIELDS)
    basic = [k for k in BASIC_GUESS if k in sb.FIELDS]
    quality = {}
    for preset, values in header['presets'].items():
        bounds = per_preset[preset] if per_preset else published
        per = {}
        compute = sb.preview_compute if hasattr(sb, 'preview_compute') else lambda c: sb.compute(c)[0]
        truth0 = compute(sb.normalize(values))
        exact = float(np.linalg.norm(evaluate(preset, values) - truth0, axis=1).max() * 1000)
        for mode, n in (('few', args.few), ('basic', args.basic), ('all', args.all)):
            for _ in range(n):
                c = dict(values)
                moved = keys if mode == 'all' else basic if mode == 'basic' else list(rng.choice(keys, int(rng.integers(2, 7)), replace=False))
                for k in moved:
                    lo, hi = bounds[k]['min'], bounds[k]['max']
                    x = rng.uniform(lo, hi) if mode != 'few' else np.clip(values[k] + rng.normal(0, .3) * (hi - lo), lo, hi)
                    c[k] = published_value(bounds[k], x) if per_preset else float(np.round(x, 3))
                truth = compute(sb.normalize(c))
                errors = np.linalg.norm(evaluate(preset, c) - truth, axis=1) * 1000
                error = float(errors.max())
                per.setdefault(mode, []).append(error)
                if error > 5:
                    print('ERROR_CASE', mode, round(error, 2), 'vertex', int(errors.argmax()),
                          json.dumps({k:v for k,v in c.items() if v != values[k]}), flush=True)
        # Gate on full precision; rounding near a limit can turn a failure into
        # a pass. These are the measurements of the actual written float arrays.
        quality[preset] = {'preset_mm': exact, **{m: {'max_mm': max(v), 'median_mm': float(np.median(v)), 'p90_mm': float(np.percentile(v, 90)), 'n': len(v)} for m, v in per.items()}}
        print('CHECK', preset, json.dumps(quality[preset]), flush=True)
    report = {'runtime': args.runtime, 'runtimeSha256': hashlib.sha256(Path(args.runtime).read_bytes()).hexdigest(),
              'source': header['source'], 'rules': header['rules'], 'basis': K, 'seed': args.seed,
              'quality': quality, 'publishedControls': published, 'geometry': geometry}
    if per_preset:
        report['publishedControls'] = per_preset
        report['catalogSha256'] = args.catalog_sha256
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
    shipped_path = Path(args.out or args.runtime)
    report['size'] = {'bytes': shipped_path.stat().st_size, 'limitBytes': 32 * 1024 * 1024}
    report['passed'] = report['passed'] and report['size']['bytes'] <= report['size']['limitBytes']
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
