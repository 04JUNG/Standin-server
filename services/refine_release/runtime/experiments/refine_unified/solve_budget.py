"""Bounded work with honest convergence status; eligibility still needs geometry gates."""
import os,time,json,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares as scipy_least_squares
POLICY={'shoulder':(30,8.0),'leg':(24,2.5),'depth':(60,1.5),'skin':(24,0.8)}
def solve(fun,x0,*args,stage='depth',motion=None,cache_inputs=None,**kwargs):
 start=time.monotonic();history=[];reason=None;hit=False
 root=Path(os.environ['REFINE_MINUTE_STATE']) if os.environ.get('REFINE_MINUTE_STATE') else None
 usedfile=root/'solver-used.json' if root else None
 state=json.loads(usedfile.read_text()) if usedfile and usedfile.exists() else {'seconds':0.,'by_mode':{}}
 mode=os.environ.get('STANDIN_EXPERIMENT_QUALITY','common');used=state['seconds'];mode_used=state.get('by_mode',{}).get(mode,0.)
 n,seconds=POLICY[stage];seconds=min(seconds,max(0.,28.-used),max(0.,14.-mode_used));requested=kwargs.pop('max_nfev',100);n=min(n,requested)
 if seconds<=0:n=1;reason='shared_solver_time_budget'
 cache=None
 if root and cache_inputs:
  root.mkdir(parents=True,exist_ok=True);key=hashlib.sha256()
  for p in cache_inputs:key.update(Path(p).read_bytes())
  key.update(Path(__file__).read_bytes());key.update(np.asarray(x0).tobytes());key.update(repr((stage,n,kwargs.get('bounds'))).encode());cache=root/'solve-cache'/(key.hexdigest()+'.json')
 def callback(intermediate_result):
  nonlocal reason
  q=intermediate_result;sample=dict(nfev=int(q.nfev),cost=float(q.cost),seconds=time.monotonic()-start)
  if motion:sample['motion']=np.asarray(motion(q.x)).tolist()
  history.append(sample)
  if len(history)>=6 and q.nfev>=10:
   prior=history[-6];relative=(prior['cost']-sample['cost'])/max(1.,abs(prior['cost']))
   stable_motion=not motion or np.max(np.linalg.norm(np.array(sample['motion'])-np.array(prior['motion']),axis=-1))<.5
   if 0<=relative<.002 and stable_motion:reason='gain_plateau';raise StopIteration
  if time.monotonic()-start>=seconds:reason='solver_time_budget';raise StopIteration
 if cache and cache.exists():
  from scipy.optimize import OptimizeResult
  saved=json.loads(cache.read_text());x=np.array(saved['x']);f=fun(x)
  if np.allclose(f,np.array(saved['fun']),rtol=1e-10,atol=1e-10):
   r=OptimizeResult(**{**saved,'x':x,'fun':f});hit=True;reason='exact_input_shared_solve'
  else:raise ValueError('Shared solve residual mismatch')
 else:r=None
 if r is None:
  r=scipy_least_squares(fun,x0,*args,max_nfev=n,callback=callback,**kwargs)
  if r.status==0:reason=reason or 'evaluation_budget'
  r.budget_terminated=bool(reason);r.budget_reason=reason
  if cache:
   cache.parent.mkdir(parents=True,exist_ok=True);cache.write_text(json.dumps(dict(x=r.x.tolist(),fun=r.fun.tolist(),success=bool(r.success),status=int(r.status),message=r.message,nfev=r.nfev,njev=r.njev,optimality=float(r.optimality),cost=float(r.cost),budget_terminated=r.budget_terminated,budget_reason=r.budget_reason)))
 elapsed=time.monotonic()-start
 if root:
  root.mkdir(parents=True,exist_ok=True);state['seconds']=used+elapsed;state.setdefault('by_mode',{})[mode]=mode_used+elapsed;usedfile.write_text(json.dumps(state));record=dict(mode=mode,stage=stage,elapsed=elapsed,nfev=r.nfev,converged=bool(r.success),stop_reason=reason,cache_hit=hit,history=history)
  with (root/'solver-events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 return r

def eligible(result):
 return bool(result.success or getattr(result,'budget_terminated',False)) and np.isfinite(result.fun).all() and np.isfinite(result.x).all()
