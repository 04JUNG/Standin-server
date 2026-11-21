"""32 MiB SSR2 compression retaining the span of all exact preset coefficients.

Uses only training coefficients. A fresh controller/catalog check is still mandatory.
"""
from pathlib import Path
import argparse,copy,hashlib,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from check_sampled_runtime import read_ssr2,locate
from build_sampled_runtime import write_ssr2
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
p.add_argument('--expected-rules',required=True,help='Pinned authoring rules expected in the input; mismatches fail closed')
p.add_argument('--balance-blocks',action='store_true',help='Give each referenced training axis/pair equal row mass so dense grids cannot suppress rare controls')
p.add_argument('--selection-cut-ratio',type=float,help='0 uses each selected preset model; positive values use the existing SSR2 idw2 rule')
args=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
source=Path(args.input);out=Path(args.output);h,a=read_ssr2(source);oldK=h['basisCount'];nv=h['vertexCount']
assert not out.exists(),'A changed candidate must use a new output path'
assert h['rules']==args.expected_rules,'Input authoring rules differ from the explicit pin'
oldCo=a['coef'].reshape(h['coefCount'],oldK).astype(np.float64)
rows=[];row_weights=[];remapped={}
def block(offset,count):
    key=(offset,count)
    if key not in remapped:
        remapped[key]=len(rows);rows.extend(oldCo[offset:offset+count]);row_weights.extend([1/np.sqrt(count)]*count)
    return remapped[key]
for model in h['models'].values():
    model['base']=block(model['base'],1)
    for axis in model['axes']:axis['offset']=block(axis['offset'],len(axis['knots']))
    for pair in model['pairs']:pair['offset']=block(pair['offset'],len(pair['ga'])*len(pair['gb']))
    for triple in model.get('triples',[]):triple['offset']=block(triple['offset'],len(triple['ga'])*len(triple['gb'])*len(triple['gc']))
C=np.asarray(rows);h['coefCount']=len(C)
presets=[]
for name,values in h['presets'].items():
    model=h['models'][name];co=C[model['base']].copy()
    for axis in model['axes']:
        i,t=locate(axis['knots'],values[axis['key']]);co+=C[axis['offset']+i]*(1-t)+C[axis['offset']+i+1]*t
    for pair in model['pairs']:
        i,u=locate(pair['ga'],values[pair['a']]);j,v=locate(pair['gb'],values[pair['b']]);nb=len(pair['gb'])
        at=lambda x,y:C[pair['offset']+x*nb+y]
        co+=at(i,j)*(1-u)*(1-v)+at(i+1,j)*u*(1-v)+at(i,j+1)*(1-u)*v+at(i+1,j+1)*u*v
    for triple in model.get('triples',[]):
        ix,fr=zip(*(locate(triple[g],values[triple[k]]) for k,g in [('a','ga'),('b','gb'),('c','gc')]))
        for ia in (0,1):
            for ib in (0,1):
                for ic in (0,1):
                    row=triple['offset']+((ix[0]+ia)*len(triple['gb'])+ix[1]+ib)*len(triple['gc'])+ix[2]+ic
                    co+=C[row]*(fr[0] if ia else 1-fr[0])*(fr[1] if ib else 1-fr[1])*(fr[2] if ic else 1-fr[2])
    presets.append(co)
Q=np.linalg.qr(np.asarray(presets).T,mode='reduced')[0]
residual=C-(C@Q)@Q.T
training_residual=residual*np.asarray(row_weights)[:,None] if args.balance_blocks else residual
eigen,vectors=np.linalg.eigh(training_residual.T@training_residual)
# Include the actual expanded axis/pair header and room for the independent
# quality metadata written by the existing checker. The previous 64 KiB estimate
# was smaller than the refined 101 KiB header; that failed attempt is retained.
header_estimate=len(json.dumps({k:v for k,v in h.items() if k not in ['arrays','quality']},ensure_ascii=False,separators=(',',':')).encode())+16384
fixed=h['faceCount']*3*2+nv*3*4+header_estimate
per=4+len(C)*4+nv*3*2
K=min(oldK,(32*1024*1024-fixed)//per)
assert K>Q.shape[1]
V=np.concatenate([Q,vectors[:,np.argsort(eigen)[::-1][:K-Q.shape[1]]]],axis=1)
# Orthogonalize again to remove round-off along the constrained span.
V=np.linalg.qr(V,mode='reduced')[0]
B=a['basis'].reshape(oldK,nv*3).astype(np.float64)*a['basisScale'][:,None]
newB=V.T@B;scale=np.max(np.abs(newB),axis=1)/32767
assert np.all(scale>0)
quantized=np.rint(newB/scale[:,None]).clip(-32767,32767).astype(np.int16)
newCo=(C@V).astype(np.float32)
h['basisCount']=K;h.pop('arrays',None);h.pop('quality',None)
if args.selection_cut_ratio is not None:
    if args.selection_cut_ratio:h['selection']={'mode':'idw2','cutRatio':args.selection_cut_ratio}
    else:h.pop('selection',None)
write_ssr2(out,h,[('faces','uint16',a['faces']),('mean','float32',a['mean']),('basisScale','float32',scale.astype(np.float32)),
    ('coef','float32',newCo.reshape(-1)),('basis','int16',quantized.reshape(-1))])
assert out.stat().st_size<=32*1024*1024
report={'parentSha256':hashlib.sha256(source.read_bytes()).hexdigest(),'runtimeSha256':hashlib.sha256(out.read_bytes()).hexdigest(),
    'bytes':out.stat().st_size,'limitBytes':32*1024*1024,'basisCount':K,'referencedCoefficientRows':len(C),
    'source':h['source'],'rules':h['rules'],'presetConstraintCount':len(presets),'approvalTransferred':False,
    'headerBudgetBytes':header_estimate,'selection':h.get('selection'),
    'method':'Referenced coefficient compaction; exact preset span plus residual training PCA; re-quantized int16 SSR2',
    'freshIndependentControllerCatalogCheckRequired':True}
report['trainingBlockBalanced']=args.balance_blocks
report['compressionAlgorithmSha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
out.with_suffix(out.suffix+'.compression.json').write_text(json.dumps(report,indent=2)+'\n')
print('COMPRESSED_CANDIDATE',json.dumps(report),flush=True)
