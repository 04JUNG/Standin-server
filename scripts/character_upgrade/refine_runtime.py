"""Add measured missing stature/limb interactions to an existing SSR2 basis."""
import argparse,sys,json,importlib.util,copy
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from check_sampled_runtime import read_ssr2,make_evaluator
from build_sampled_runtime import write_ssr2
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--gender',required=True)
args=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);r=Path(args.root);g=args.gender
import bpy
bpy.ops.wm.open_mainfile(filepath=str(r/'authoring'/g/'model.blend'))
spec=importlib.util.spec_from_file_location('refine_controller',r/'authoring'/g/'standin_body.py');sb=importlib.util.module_from_spec(spec);spec.loader.exec_module(sb);sb.register()
h,a=read_ssr2(r/(g+'.ssr'));K=h['basisCount'];nv=h['vertexCount'];ev=make_evaluator(copy.deepcopy(h),a,K)
basis=a['basis'].reshape(K,nv*3).astype(np.float64)*a['basisScale'][:,None]
project=np.linalg.solve(basis@basis.T,basis);co=list(a['coef'].reshape(h['coefCount'],K));base=h['presets']['BASE'];m=h['models']['BASE']
for x,y in [('height_cm','arm_length'),('height_cm','leg_length')]:
 if any({p['a'],p['b']}=={x,y} for p in m['pairs']):continue
 ga=sorted({h['controls'][x]['min'],base[x],h['controls'][x]['max']});gb=sorted({h['controls'][y]['min'],base[y],h['controls'][y]['max']})
 offset=len(co)
 for u in ga:
  for v in gb:
   c={**base,x:u,y:v};residual=sb.preview_compute(c)-ev('BASE',c)
   co.append(project@residual.reshape(-1))
 m['pairs'].append({'a':x,'b':y,'ga':ga,'gb':gb,'offset':offset});print('ADDED',x,y,flush=True)
controls=json.loads((r/(g+'-published-controls.json')).read_text())
h['controls']=controls;h['coefCount']=len(co);h.pop('arrays',None);h.pop('quality',None)
write_ssr2(r/(g+'.ssr'),h,[('faces','uint16',a['faces']),('mean','float32',a['mean']),('basisScale','float32',a['basisScale']),('coef','float32',np.array(co,dtype=np.float32).reshape(-1)),('basis','int16',a['basis'])])
print('REFINED',g,flush=True)
