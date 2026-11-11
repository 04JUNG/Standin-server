"""Repeatable synthetic source-only benchmark; never certifies target FBX quality.

Run from repository root with PYTHONPATH=. python scripts/eval_refine_v3_synthetic.py.
"""
import argparse
import json, tempfile,time
from pathlib import Path
import numpy as np
from tests.test_refine_v3_solver import _inputs
from src.refine_regional import build_regional_binding
from src.refine_v3 import refine_regional_bvh
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()
masks=[(),(10,),(8,10,14,16),(13,14,15,16),(11,12,13,14,15,16),(8,10,11,12,13,14,15,16),(7,)]
rows=[]
with tempfile.TemporaryDirectory(prefix='refine-v3-bench-') as directory:
 for n,mask in enumerate(masks):
  path=Path(directory)/str(n);path.mkdir()
  source,inputs=_inputs(path,missing=mask)
  start=time.perf_counter();policy=build_regional_binding(**inputs,enabled=True)
  binding_ms=(time.perf_counter()-start)*1000
  durations=[];reasons=[]
  for trial in range(10):
   start=time.perf_counter()
   result=refine_regional_bvh(source,inputs['keypoints'],policy.binding,inputs['provenance'],inputs['ownership'],enabled=True,deadline_seconds=1)
   durations.append((time.perf_counter()-start)*1000);reasons.append(result.reason)
  rows.append(dict(mask=mask,eligible_limbs=policy.limbs,policy_ms=binding_ms,first_ms=durations[0],warm_p50_ms=float(np.percentile(durations[1:],50)),warm_p95_ms=float(np.percentile(durations[1:],95)),reasons=sorted(set(reasons)),mesh_status='mesh_unverified'))
report=dict(kind='synthetic-source-cache-off',runs=70,cold_process_measured=False,actual_target_fbx_quality=False,production_support=False,rows=rows)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report,indent=2))
print(json.dumps(report))
