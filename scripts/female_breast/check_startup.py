"""Check Blender Text Editor startup restores the same shape as imported rules."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from verify import load

ap=argparse.ArgumentParser()
ap.add_argument('--candidate',type=Path,required=True)
args=ap.parse_args(sys.argv[sys.argv.index('--')+1:])
meta=json.loads((args.candidate/'candidate.json').read_text())
source=meta['source']
for name in ['blend','controller']:
    assert hashlib.sha256(Path(source[name]).read_bytes()).hexdigest()==source[name+'_sha256']
api=load(Path(source['blend']),Path(source['controller']),'startup_import')
text=Path(source['controller']).read_text()
rows=[]
for size in [-1.,1.5,2.]:
    mesh,_=api.objects()
    values={**api.defaults(mesh),'breast_size':size}
    api.set_controls(values,correct=False)
    expected=api.compute(api.controls())[0]
    namespace={'__name__':'__main__'}
    exec(compile(text,source['controller'],'exec'),namespace)
    api=SimpleNamespace(**namespace)
    mesh,_=api.objects()
    error=float(np.linalg.norm(api.evaluated(mesh)-expected,axis=1).max()*1000)
    assert api.controls()['breast_size']==size and error<.01
    rows.append({'size':size,'controls_preserved':True,'max_vertex_error_mm':error})
(args.candidate/'startup.json').write_text(json.dumps({'passed':True,'rows':rows},indent=2))
print('BREAST_STARTUP_PASSED',rows,flush=True)
