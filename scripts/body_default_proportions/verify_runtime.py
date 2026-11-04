"""Compare actual client TS evaluation with the pinned Blender surface.

evaluate_runtime.mjs must already have emitted positions and resolved metadata.
Approximate preview errors are reported separately from exact FBX tolerances.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.build_shape_release import load_controller
from scripts.body_default_proportions.build import write

ap = argparse.ArgumentParser(__doc__)
ap.add_argument('--release', type=Path, required=True)
args = ap.parse_args(sys.argv[sys.argv.index('--') + 1:])
folder = args.release / 'runtime-qa'
validation = json.loads((args.release / 'validation.json').read_text())
rows = []
for model in (x for x in validation['rows'] if x['label'] == 'candidate'):
    source, sex = model['source'], model['sex']
    c = load_controller(source['blend'], source['controller'], 'runtime_check_' + sex)
    for label in ('default', 'slider'):
        prefix = folder / f'{sex}-{label}'
        requested = json.loads(prefix.with_suffix('.json').read_text())
        actual = json.loads(Path(str(prefix) + '-result.json').read_text())
        if actual['controls'] != requested or actual['source'] != {
            'blendSha256': source['blend_sha256'], 'controllerSha256': source['controller_sha256']
        }:
            raise ValueError('runtime source or controls mismatch')
        expected = c.compute(c.normalize(requested))[0]
        predicted = np.fromfile(prefix.with_suffix('.f32'), dtype='<f4').reshape(-1, 3)
        errors = np.linalg.norm(expected - predicted, axis=1) * 1000
        rows.append({'sex': sex, 'mode': label, 'maxErrorMm': float(errors.max()),
                     'rmsErrorMm': float(np.sqrt(np.mean(errors ** 2))),
                     'sourceMatches': True, 'controlsMatch': True, 'presetAnchor': 'BASE'})
        print('TS_RUNTIME_SURFACE', rows[-1], flush=True)
    c.unregister()
write(folder / 'results.json', {'status': 'measured', 'rows': rows,
    'note': 'Sampled live preview is approximate; final FBX uses the exact controller. No FBX tolerance is applied or relaxed here.'})
