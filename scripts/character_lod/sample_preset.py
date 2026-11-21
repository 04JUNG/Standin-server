"""Sample one SSR2 preset into a fresh isolated state directory.

Uses the exact sampler/settings/seeds of runtime.py. Do not copy this state into
a running builder. A caller may import completed state only after all writers
have exited. Existing state directories are rejected to prevent shared writers.
"""
import argparse,importlib.util,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/character_upgrade'))
import build_sampled_runtime as b
p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--registry',required=True);p.add_argument('--gender',required=True);p.add_argument('--state',required=True);p.add_argument('--preset',required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);folder=Path(a.candidate);c=b.load(folder/'model.blend',folder/'controller.py');registry=json.loads(Path(a.registry).read_text());rules='standin-body-0.8.0' if a.gender=='male' else 'standin-female-body-0.4.0';c.PRESETS={v['preset']:(v['preset'],v['defaults']) for cid,v in registry['characters'].items() if cid.startswith(a.gender+'-') and v['rules']==rules};index=list(c.PRESETS).index(a.preset);state=Path(a.state);state.mkdir(parents=True,exist_ok=False);target=state/f'{index:02d}-{a.preset}';target.mkdir(exist_ok=False)
args=SimpleNamespace(axis_knots=5,axis_refine=1,axis_tol_mm=.3,pair_budget=12,pair_min_mm=1.,pair_dense=4,pair_dense_nodes=5,val_few=6,val_basic=4,val_all=4,seed=20261016)
source={'blendSha256':b.sha(folder/'model.blend'),'controllerSha256':b.sha(folder/'controller.py')};stamp={'rules':c.VERSION,'source':source,**{k:getattr(args,k) for k in ('axis_knots','axis_refine','axis_tol_mm','pair_budget','pair_min_mm','pair_dense','pair_dense_nodes','seed')},'val':[6,4,4]}
v=b.Sampler(c);started=time.time();sample=b.sample_preset(v,c,a.preset,args,np.random.default_rng([args.seed,index]));rows=[r for _,(_,rr) in sample['axes'].items() for r in rr];rows += [r for (_,_,table) in sample['pairs'].values() for r in table.reshape(-1,len(sample['VP']),3)];np.save(target/'rows.npy',np.stack([r.reshape(-1) for r in rows]).astype(np.float32));np.save(target/'VP.npy',sample['VP']);np.save(target/'validation.npy',np.stack([truth for _,_,truth in sample['validation']]));meta={'stamp':stamp,'values':sample['values'],'evals':v.evals,'seconds':round(time.time()-started,1),'candidates':sample['candidates'],'axes':[(k,[float(x) for x in knots],len(rr)) for k,(knots,rr) in sample['axes'].items()],'pairs':[(x,y,[float(v) for v in ga],[float(v) for v in gb]) for (x,y),(ga,gb,_) in sample['pairs'].items()],'validation':[(m,controls) for m,controls,_ in sample['validation']]};tmp=target/'meta.complete.json';tmp.write_text(json.dumps(meta));tmp.rename(target/'meta.json');print('PRESET_COMPLETE',a.preset,v.evals,flush=True)
