"""Share immutable skinning operators across residual/Jacobian evaluations.
Algebraically identical linear blend skinning; no joint or weight approximation.
"""
import hashlib,os,json,atexit
from pathlib import Path
from collections import OrderedDict
import numpy as np
from scipy.sparse import csr_matrix
_OPERATORS=OrderedDict();_INVERSES=OrderedDict();STATS={'calls':0,'operator_builds':0,'inverse_builds':0}
def _record():
 if os.environ.get('REFINE_MINUTE_STATE'):
  p=Path(os.environ['REFINE_MINUTE_STATE']);p.mkdir(parents=True,exist_ok=True);(p/('skinning-'+str(os.getpid())+'.json')).write_text(json.dumps(STATS))
atexit.register(_record)
def _key(*arrays):
 h=hashlib.sha256()
 for a in arrays:h.update(str((a.shape,a.dtype.str)).encode());h.update(np.ascontiguousarray(a).data)
 return h.digest()
def skin(v,weights,world,base):
 STATS['calls']+=1
 key=_key(v,weights)
 if key not in _OPERATORS:
  STATS['operator_builds']+=1
  vi,bi=np.nonzero(weights);hom=np.column_stack((v,np.ones(len(v))));data=(weights[vi,bi,None]*hom[vi]).ravel();rows=np.repeat(vi,4);cols=(bi[:,None]*4+np.arange(4)).ravel()
  _OPERATORS[key]=csr_matrix((data,(rows,cols)),shape=(len(v),len(base)*4))
  if len(_OPERATORS)>8:_OPERATORS.popitem(last=False)
 _OPERATORS.move_to_end(key);bk=_key(base)
 if bk not in _INVERSES:
  STATS['inverse_builds']+=1
  _INVERSES[bk]=np.linalg.inv(base)
  if len(_INVERSES)>8:_INVERSES.popitem(last=False)
 transform=(world@_INVERSES[bk])[:,:3,:].transpose(0,2,1).reshape(len(base)*4,3)
 return _OPERATORS[key]@transform
