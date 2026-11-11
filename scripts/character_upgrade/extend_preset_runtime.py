"""Extend a shared, source-pinned SSR2 basis with measured preset residuals."""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from check_sampled_runtime import read_ssr2,make_evaluator
from build_sampled_runtime import write_ssr2

ap=argparse.ArgumentParser();ap.add_argument('--approved',required=True);ap.add_argument('--out',required=True)
args=ap.parse_args();root=Path(args.out)
for gender in ('male','female'):
    p=json.loads((root/gender/'probe.json').read_text(encoding='utf-8'))
    h,a=read_ssr2(Path(args.approved)/(gender+'.ssr'));K=h['basisCount'];nv=h['vertexCount']
    ev=make_evaluator(copy.deepcopy(h),a,K)
    basis=a['basis'].reshape(K,nv*3).astype(np.float64)*a['basisScale'][:,None]
    residuals=np.stack([(np.load(root/gender/(name+'.npy'))-ev('BASE',entry['controls'])).reshape(-1) for name,entry in p['presets'].items()])
    # Add only directions actually missing from the original training basis.
    originalK=K
    while True:
        project=np.linalg.solve(basis@basis.T,basis)
        remainder=residuals-(residuals@project.T)@basis
        errors=np.linalg.norm(remainder.reshape(len(residuals),nv,3),axis=2).max(axis=1)
        if errors.max()<.00045:break
        direction=remainder[errors.argmax()];direction/=np.linalg.norm(direction)
        basis=np.vstack([basis,direction]);K+=1
        if K-originalK>2:raise ValueError('More than two additional directions would exceed runtime budget')
    scales=np.max(abs(basis),axis=1)/32760
    quantized=np.round(basis/scales[:,None]).astype(np.int16)
    basis=quantized.astype(np.float64)*scales[:,None]
    project=np.linalg.solve(basis@basis.T,basis)
    co=list(np.pad(a['coef'].reshape(h['coefCount'],originalK),((0,0),(0,K-originalK))))
    template=copy.deepcopy(h['models']['BASE']);report={}
    pair_path=root/(gender+'-pair-cases.json')
    if pair_path.exists():
        pair_data=json.loads(pair_path.read_text());pair_truth=np.load(root/(gender+'-pair-surfaces.npy'))
        residual_co=np.stack([project@(truth-ev('BASE',c)).reshape(-1) for truth,c in zip(pair_truth,pair_data['rows'])])
        for pair in pair_data['pairs']:
            ga,gb=pair['ga'],pair['gb'];i0=ga.index(h['presets']['BASE'][pair['a']]);j0=gb.index(h['presets']['BASE'][pair['b']])
            table=residual_co[pair['indices']].reshape(len(ga),len(gb),K)
            interaction=table-table[:,j0:j0+1]-table[i0:i0+1,:]+table[i0,j0]
            template['pairs'].append({k:pair[k] for k in ('a','b','ga','gb')}|{'offset':len(co)});co.extend(interaction.reshape(-1,K))
        # Dense single-axis residuals capture local fold-guard transitions as well.
        for key in ('physique','muscle'):
            rows=[(c[key],i) for i,c in enumerate(pair_data['rows']) if all(abs(v-h['presets']['BASE'][k])<1e-9 for k,v in c.items() if k!=key)]
            rows=sorted(dict(rows).items());base_row=next(i for x,i in rows if abs(x-h['presets']['BASE'][key])<1e-9)
            if len(rows)>1:
                template['axes'].append({'key':key,'knots':[x for x,i in rows],'offset':len(co)});co.extend(residual_co[i]-residual_co[base_row] for x,i in rows)
        hg=copy.deepcopy(h);hg['basisCount']=K;hg['coefCount']=len(co);hg['models']['BASE']=template
        ev=make_evaluator(hg,{**a,'basis':quantized.reshape(-1),'basisScale':scales.astype(np.float32),'coef':np.asarray(co,dtype=np.float32).reshape(-1)},K)
    for name,entry in p['presets'].items():
        values=entry['controls'];truth=np.load(root/gender/(name+'.npy'))
        before=ev('BASE',values);residual=(truth-before).reshape(-1);correction=project@residual
        after=before+(correction@basis).reshape(nv,3)
        report[name]={'before_mm':float(np.linalg.norm(truth-before,axis=1).max()*1000),'after_mm':float(np.linalg.norm(truth-after,axis=1).max()*1000)}
        model=copy.deepcopy(template);model['base']=len(co);co.append(co[template['base']]+correction)
        h['models'][name]=model;h['presets'][name]=values
    # Axis grids were trained on these native ranges, including extreme presets.
    for key,field in p['fields'].items():h['controls'][key]={'min':field[0],'max':field[1],'default':h['presets']['BASE'][key]}
    h['selection']={'mode':'idw2','cutRatio':3.0};h['coefCount']=len(co);h['basisCount']=K;h.pop('arrays',None);h.pop('quality',None)
    target=root/(gender+'.ssr')
    write_ssr2(target,h,[('faces','uint16',a['faces']),('mean','float32',a['mean']),('basisScale','float32',scales.astype(np.float32)),('coef','float32',np.asarray(co,dtype=np.float32).reshape(-1)),('basis','int16',quantized.reshape(-1))])
    if target.stat().st_size>32*1024*1024:raise ValueError('runtime exceeds client size limit')
    (root/(gender+'-projection.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(gender,json.dumps(report),flush=True)
