"""Export one source-validated neutral review job, with the real fixed-FBX writer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from converter.shape_bake import _load

ap = argparse.ArgumentParser()
ap.add_argument('--job', type=Path, required=True)
a = ap.parse_args(sys.argv[sys.argv.index('--') + 1:])
job = json.loads(a.job.read_text())
out = a.job.parent
sb = _load(job['recipe'])
mesh, _ = sb.objects()
points = sb.compute(job['recipe']['values'])[0]
np.save(out / 'controller-neutral.npy', points)
bundle = sb.export_fixed(str(out / 'fixed'))
meta = json.loads((bundle / 'manifest.json').read_text())
target = out / 'download.fbx'
shutil.copyfile(bundle / 'model.fbx', target)
result = {'shape_source': {k: job['recipe'][k] for k in ('rules', 'blend_sha256', 'controller_sha256')},
          'controls': job['recipe']['values'], 'shape_sha256': job['recipe']['shape_sha256'],
          'fbx_sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bytes': target.stat().st_size,
          'validation': meta, 'kind': 'neutral-review-export'}
(out / 'result.json').write_text(json.dumps(result, indent=2))
print('QA_NEUTRAL_EXPORTED', target, flush=True)
