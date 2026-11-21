"""Measure missing shape-control interactions at a fixed training grid."""
import argparse,importlib.util,json,sys
from pathlib import Path
import bpy,numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--approved',required=True);ap.add_argument('--out',required=True);ap.add_argument('--gender',required=True)
args=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);r=Path(args.out);g=args.gender;src=Path(args.approved)/'authoring'/g;p=json.loads((r/g/'probe.json').read_text(encoding='utf-8'));base=p['presets']['BASE']['controls']
bpy.ops.wm.open_mainfile(filepath=str(src/'model.blend'));spec=importlib.util.spec_from_file_location('pair_controller',src/'standin_body.py');sb=importlib.util.module_from_spec(spec);spec.loader.exec_module(sb);sb.register()
pairs=[('physique','neck_length'),('muscle','leg_length'),('muscle','leg_ratio'),('muscle','detail_shoulder'),('muscle','detail_upperarm'),('muscle','shoulders'),('muscle','breast_size'),('muscle','physique')]
if g=='male':pairs=[('heads','detail_thigh'),('heads','detail_upperarm'),('muscle','neck_length'),('muscle_bulk','neck_length'),('physique','detail_upperarm'),('physique','fat_arms'),('height_cm','chest_depth'),('muscle','detail_thigh'),('physique','detail_thigh'),('muscle','detail_back'),('heads','leg_length'),('heads','torso_length'),('heads','chest_depth')]
rows=[];truth=[];cache={};records=[]
prior=r/(g+'-pair-cases.json')
if prior.exists():
 old=json.loads(prior.read_text());rows=old['rows'];truth=list(np.load(r/(g+'-pair-surfaces.npy')));cache={json.dumps(c,sort_keys=True):i for i,c in enumerate(rows)}
for x,y in pairs:
 if x not in p['fields'] or y not in p['fields']:continue
 grids=[]
 for k in (x,y):
  lo,hi=p['fields'][k][:2]
  extra=([-.9,-.8,-.7,-.6,-.3,0,.3,.6,1.] if k=='physique' else [.1,.12,.2,.35,.5,1.,1.5,2.,2.5,3.,3.5,3.75] if k=='muscle' else []) if g=='female' else ([6.4,7.,8.3] if k=='heads' else [1.,1.7] if k=='physique' else [.45,1.4] if k=='muscle' else [-.5,.5] if k.startswith('detail_') or k=='fat_arms' else [])
  grids.append(sorted({float(v) for v in [lo,hi,base[k],(lo+hi)/2]+extra if lo<=v<=hi}))
 ga,gb=grids;indices=[]
 for u in ga:
  for v in gb:
   c={**base,x:u,y:v};key=json.dumps(c,sort_keys=True)
   if key not in cache:
    cache[key]=len(rows);rows.append(c);truth.append(sb.preview_compute(c).astype(np.float32))
   indices.append(cache[key])
 records.append({'a':x,'b':y,'ga':ga,'gb':gb,'indices':indices});print('PAIR_SAMPLED',g,x,y,len(rows),flush=True)
np.save(r/(g+'-pair-surfaces.npy'),np.stack(truth));(r/(g+'-pair-cases.json')).write_text(json.dumps({'rows':rows,'pairs':records}),encoding='utf-8')
