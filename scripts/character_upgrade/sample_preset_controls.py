"""Capture an independent set of real controller surfaces for every preset."""
import argparse,importlib.util,json,sys
from pathlib import Path
import bpy,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from check_sampled_runtime import read_ssr2,BASIC_GUESS
ap=argparse.ArgumentParser();ap.add_argument('--approved',required=True);ap.add_argument('--out',required=True);ap.add_argument('--gender',required=True);ap.add_argument('--seed',type=int,default=271811);ap.add_argument('--only');ap.add_argument('--suffix',default='validation');ap.add_argument('--counts',default='12,8,4')
args=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);root=Path(args.out);g=args.gender;src=Path(args.approved)/'authoring'/g
p=json.loads((root/g/'probe.json').read_text(encoding='utf-8'));old,_=read_ssr2(Path(args.approved)/(g+'.ssr'))
bpy.ops.wm.open_mainfile(filepath=str(src/'model.blend'));spec=importlib.util.spec_from_file_location('sample_controller',src/'standin_body.py');sb=importlib.util.module_from_spec(spec);spec.loader.exec_module(sb);sb.register()
rng=np.random.default_rng(args.seed);cases=[];truths=[];bounds={};keys=list(sb.FIELDS);basic=[k for k in BASIC_GUESS if k in keys]
for preset,entry in p['presets'].items():
    if args.only and preset!=args.only:continue
    values=entry['controls'];bounds[preset]={k:{'min':min(v,old['controls'][k]['min']),'max':max(v,old['controls'][k]['max'])} for k,v in values.items()}
    if g=='male' and preset=='FAT_MAX':
        # Keep the heavy template within a checked stature/proportion envelope.
        bounds[preset]['height_cm']['max']=185.;bounds[preset]['heads']['max']=7.7
    for mode,n in zip(('few','basic','all'),map(int,args.counts.split(','))):
        for _ in range(n):
            c=dict(values);moved=keys if mode=='all' else basic if mode=='basic' else list(rng.choice(keys,int(rng.integers(2,7)),replace=False))
            for k in moved:
                lo,hi=bounds[preset][k]['min'],bounds[preset][k]['max'];x=rng.uniform(lo,hi) if mode!='few' else np.clip(values[k]+rng.normal(0,.3)*(hi-lo),lo,hi);c[k]=float(np.round(x,6))
            cases.append({'preset':preset,'mode':mode,'controls':c});truths.append(sb.preview_compute(sb.normalize(c)).astype(np.float32))
    print('SAMPLED',g,preset,len(cases),flush=True)
np.save(root/(g+'-'+args.suffix+'-surfaces.npy'),np.stack(truths));(root/(g+'-'+args.suffix+'-cases.json')).write_text(json.dumps({'seed':args.seed,'bounds':bounds,'cases':cases},ensure_ascii=False),encoding='utf-8')
print('SAMPLE_COMPLETE',g,len(cases),flush=True)
