"""Training-table simplification with a conservative per-model geometry bound.

Retained knots are original knots. Each independent 1-D interpolation operator
has norm <=1 in the maximum coefficient norm, so dimension errors telescope.
The coefficient L2 bound times maximum vertex basis Frobenius norm bounds every
vertex. Preset grid knots are retained. Independent controller checks still apply.
"""
from pathlib import Path
import argparse,copy,hashlib,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from sampled_table_simplification import simplify_axis
from check_sampled_runtime import read_ssr2
from build_sampled_runtime import write_ssr2
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--expected-rules',required=True);p.add_argument('--model-error-mm',type=float,default=.05);args=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
assert 0<args.model_error_mm<=.05
source=Path(args.input);out=Path(args.output);assert not out.exists();h,a=read_ssr2(source);assert h['rules']==args.expected_rules
K=h['basisCount'];C=a['coef'].reshape(h['coefCount'],K).astype(np.float64);B=a['basis'].reshape(K,h['vertexCount'],3).astype(np.float64)*a['basisScale'][:,None,None]
vertex_bound=float(np.sqrt(np.sum(B*B,axis=(0,2))).max());assert vertex_bound>0;del B
rows=[];remapped={};records=[]
def store(key,values):
 if key not in remapped:remapped[key]=len(rows);rows.extend(values)
 return remapped[key]
for name,model in h['models'].items():
 model['base']=store(('base',model['base']),[C[model['base']]])
 terms=[('axes',x) for x in model['axes']]+[(kind,x) for kind in ['pairs','triples','quads'] for x in model.get(kind,[])]
 budget=args.model_error_mm/max(1,len(terms))
 for kind,term in terms:
  if kind=='axes':keys=[term['key']];gridkeys=['knots']
  else:keys=[term[k] for k in ['a','b','c','d'] if k in term];gridkeys=['ga','gb','gc','gd'][:len(keys)]
  grids=[term[k] for k in gridkeys];shape=tuple(map(len,grids));old_offset=term['offset'];table=C[old_offset:old_offset+int(np.prod(shape))].reshape(*shape,K)
  tolerance=budget/len(keys)/1000/vertex_bound;indices=[simplify_axis(table,i,g,{h['presets'][name][key]},tolerance) for i,(key,g) in enumerate(zip(keys,grids))]
  trimmed=table
  for i,index in enumerate(indices):trimmed=np.take(trimmed,index,axis=i)
  flat=trimmed.reshape(-1,K);term['offset']=store(('term',old_offset,shape,tuple(tuple(x) for x in indices)),flat)
  for key,grid,index in zip(gridkeys,grids,indices):term[key]=[grid[i] for i in index]
  records.append(dict(model=name,kind=kind,keys=keys,before=list(shape),after=list(trimmed.shape[:-1]),certifiedMaxGeometryChangeMm=budget))
  if tuple(trimmed.shape[:-1])!=shape:print('TABLE_SIMPLIFIED',name,keys,list(shape),list(trimmed.shape[:-1]),flush=True)
h['coefCount']=len(rows);h.pop('arrays',None);h.pop('quality',None)
write_ssr2(out,h,[('faces','uint16',a['faces']),('mean','float32',a['mean']),('basisScale','float32',a['basisScale']),('coef','float32',np.asarray(rows,dtype=np.float32).reshape(-1)),('basis','int16',a['basis'])])
report=dict(inputSha256=hashlib.sha256(source.read_bytes()).hexdigest(),runtimeSha256=hashlib.sha256(out.read_bytes()).hexdigest(),beforeRows=len(C),afterRows=len(rows),perModelGeometryErrorBoundMm=args.model_error_mm,maxVertexBasisFrobeniusNorm=vertex_bound,method='Original-knot retained subset; conservative independent-axis interpolation bound; model term budgets sum to declared bound',presetGridKnotsPreserved=True,approvalTransferred=False,records=records)
out.with_suffix('.simplification.json').write_text(json.dumps(report,indent=2)+'\n');print('TABLE_SIMPLIFICATION_COMPLETE',len(C),len(rows),flush=True)
