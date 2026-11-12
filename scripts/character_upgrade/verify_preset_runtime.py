"""Compare source surfaces with the exact SSR2 client interpolation contract."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from check_sampled_runtime import read_ssr2,make_evaluator
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--gender',required=True);ap.add_argument('--independent',action='store_true');args=ap.parse_args();r=Path(args.out);g=args.gender
h,a=read_ssr2(r/(g+'.ssr'));ev=make_evaluator(h,a,h['basisCount']);data=json.loads((r/(g+'-validation-cases.json')).read_text(encoding='utf-8'));truth=np.load(r/(g+'-validation-surfaces.npy'),mmap_mode='r');limits={'preset':.5,'few':10.,'basic':15.,'all':20.};percentile_limits={'few':5.,'basic':10.,'all':15.};report={};failures=[]
for name,values in h['presets'].items():
    e=float(np.linalg.norm(ev(name,values)-np.load(r/g/(name+'.npy')),axis=1).max()*1000);report[name]={'preset':{'max_mm':e,'n':1}}
    if e>limits['preset']:failures.append({'preset':name,'mode':'preset','max_mm':e})
sets=[('calibration',data,truth)];bounds=data['bounds'];excluded=0
if args.independent:
    fresh=json.loads((r/(g+'-independent-cases.json')).read_text());bounds=fresh['bounds'];sets.append(('independent',fresh,np.load(r/(g+'-independent-surfaces.npy'),mmap_mode='r')))
for label,samples,surfaces in sets:
 for i,c in enumerate(samples['cases']):
    if any(v<bounds[c['preset']][k]['min']-1e-8 or v>bounds[c['preset']][k]['max']+1e-8 for k,v in c['controls'].items()):excluded+=1;continue
    err=np.linalg.norm(ev(c['preset'],c['controls'])-surfaces[i],axis=1)*1000;e=float(err.max());p99=float(np.percentile(err,99.9));per=report[c['preset']].setdefault(c['mode'],{'max_mm':0,'p999_mm':0,'n':0});per['max_mm']=max(e,per['max_mm']);per['p999_mm']=max(p99,per['p999_mm']);per['n']+=1
    if e>limits[c['mode']] or p99>percentile_limits[c['mode']]:failures.append({'set':label,'index':i,'preset':c['preset'],'mode':c['mode'],'max_mm':e,'p999_mm':p99,'vertex':int(err.argmax()),'controls':c['controls']})
out={'passed':not failures,'independent':args.independent,'seeds':[x[1]['seed'] for x in sets],'excludedOutsidePublishedBounds':excluded,'limits':limits,'p999Limits':percentile_limits,'quality':report,'failures':failures,'publishedBounds':bounds};(r/(g+'-runtime-qa.json')).write_text(json.dumps(out,indent=2),encoding='utf-8')
print(g,json.dumps(report),flush=True);print('FAILURES',len(failures),json.dumps([{k:v for k,v in f.items() if k!='controls'} for f in failures]),flush=True)
