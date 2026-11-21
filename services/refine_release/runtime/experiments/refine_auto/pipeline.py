"""One-command automatic refinement from a selected 004/006/010 checkpoint.

No case-specific pose conditions. Cached or fresh observations/semantics feed
shoulder preflight, audited mode selection, anchor-aware legs, audit and review.
"""
import argparse,copy,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from policy import lower_available
HERE=Path(__file__).resolve().parent
SHOULDER=HERE.parent/'refine_shoulder_line'
sys.path.insert(0,str(SHOULDER))
from select_result import choose_mode
sys.path.insert(0,str(HERE.parent/'refine_new'))
from core import targets,prepare
import sys,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine_unified'))
from validity import shadow


def read(p):return json.loads(Path(p).read_text())
def write(p,d):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind_inputs(fixture,observation,conditions=None):
 if observation['image_sha256']!=fixture['files']['rough']['sha256']:raise ValueError('Observation image hash mismatch')
 if conditions is not None:
  if conditions.get('source')=='user_review':raise ValueError('Manual conditions forbidden in automatic mode')
  if conditions['image_sha256']!=observation['image_sha256']:raise ValueError('Condition image hash mismatch')

def measured_policy(fit,observation,fixture,assets):
 goal,*_=targets(observation,fixture['image_size']);names,ix,_,_,_=prepare(read(assets/fixture['files']['rig']['path']));w=np.load(fit/'expected-world.npz')['world'];pr=read(fit/'adjustment.json')
 if not all(n in goal for n in ['LeftArm','RightArm']):return dict(upper_body_fit_verified=False,reason='missing_shoulder_observation',lower_body_may_run=True,whole_upper_body_correctness_certified=False)
 xy=((w[:,:3,3]-pr['origin'])@np.array(pr['camera']))[:,:2]*[pr['pixel_scale'],-pr['pixel_scale']]+pr['offset'];s=xy[[ix['LeftArm'],ix['RightArm']]];g=np.array([goal['LeftArm'],goal['RightArm']]);errors=np.linalg.norm(s-g,axis=1);vector=float(np.linalg.norm((s[0]-s[1])-(g[0]-g[1])));center=float(np.linalg.norm(s.mean(0)-g.mean(0)));ready=bool(max(errors)<=4 and vector<=5 and center<=3)
 return dict(upper_body_fit_verified=ready,reason='shoulder_line_within_tolerance' if ready else 'shoulder_line_unresolved',endpoint_errors=errors.tolist(),vector_error=vector,center_error=center,lower_body_may_run=True,whole_upper_body_correctness_certified=False)

def command(args,log,events,label):
 t=time.monotonic()
 env=os.environ.copy()
 if label=='skeleton':env['POSE_CANARY_STAGE']='shadow'
 with log.open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,env=env)
 events.append(dict(stage=label,returncode=r.returncode,seconds=time.monotonic()-t,log=str(log)))
 return r.returncode==0

def neutral_conditions(obs,obspath):
 return dict(source='no_semantic_evidence',image_sha256=obs['image_sha256'],observation_sha256=digest(obspath),near=None,far=None,relative_order_enabled=False,bent_knees=[],straight_knees=[],unobserved_joints=[],foot_evidence={},local_occlusions=[],semantic_status='unavailable_geometry_only')

def normalize_checkpoint(case_dir):
 fit=case_dir/'fit';pr=read(fit/'adjustment.json');m=read(fit/'metrics.json');fx=read(case_dir/'fixture.json')
 defaults=dict(files=fx['files'],geometry_accepted=False,rejection_reasons=[],bound_hit_bones=[],regional_acceptance=[],old_rmse=m.get('rmse_px',0),orientation_stage={},lens=dict(selected='orthographic',inverse_distance=0,pixel_scale=pr['pixel_scale'],offset=pr['offset']))
 for k,v in defaults.items():m.setdefault(k,v)
 write(fit/'metrics.json',m)
 if not (fit/'camera.json').exists():write(fit/'camera.json',dict(camera=pr['camera'],origin=pr['origin'],pixel_scale=pr['pixel_scale'],offset=pr['offset'],projection='orthographic',inverse_distance=pr.get('inverse_distance',0)))

def archive_generated_exports(root,out):
 root.resolve().relative_to(out.resolve())
 history=root/'audit-history'/str(time.time_ns());moved=[]
 for row in read(root/'results.json'):
  fit=root/row['case_id']/'fit'
  for name in [row['case_id']+'-adjusted.fbx',row['case_id']+'-adjusted.blend',row['case_id']+'-adjusted.blend1','export-check.json']:
   src=fit/name
   if src.exists():
    dst=history/row['case_id']/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(src),str(dst));moved.append(str(dst))
 for name in ['audit-cases','audit.json','status.json']:
  src=root/name
  if src.exists():
   history.mkdir(parents=True,exist_ok=True);shutil.move(str(src),str(history/name));moved.append(str(history/name))
 return moved

def audit(root,baseline,a,events,label):
 archive_generated_exports(root,a.out)
 ok=command([sys.executable,str(SHOULDER/'audit.py'),'--run',str(root),'--baseline',str(baseline),'--assets',str(a.assets),'--blender',a.blender],root/'audit-command.log',events,label)
 return ok and (root/'audit.json').exists() and all(x['export_passed'] for x in read(root/'audit.json'))

def run_case(c,a):
 root=a.out/'cases'/c;root.mkdir(parents=True);events=[];stages={};fx=read(a.baseline/c/'fixture.json');baseline=root/'baseline';shutil.copytree(a.baseline/c,baseline/c);normalize_checkpoint(baseline/c);current=baseline/c
 try:
  # Bind actual image and rig assets, rather than trusting filenames or case labels.
  for key in ['rough','rig','mesh','source']:
   item=fx['files'][key]
   if digest(a.assets/item['path'])!=item['sha256']:raise ValueError('Asset hash mismatch: '+key)
  if read(baseline/c/'fit/camera.json').get('inverse_distance',0)!=0:raise ValueError('Perspective checkpoint preserved: this stage currently supports orthographic projection only')
  obspath=root/'observation.json'
  if a.observations and (a.observations/f'{c}.json').exists():shutil.copy2(a.observations/f'{c}.json',obspath);stages['skeleton']='bound_cache'
  else:
   if not a.extractor_checkout or not a.model_manifest:raise ValueError('Configure extractor checkout/model manifest or provide bound observations')
   ok=command([sys.executable,str(HERE.parent/'refine_new/extract_develop.py'),'--checkout',str(a.extractor_checkout),'--model-manifest',str(a.model_manifest),'--assets',str(a.assets),'--out',str(root/'extracted'),'--cases',c],root/'extract.log',events,'skeleton')
   if not ok:raise RuntimeError('Develop skeleton extraction failed; no mock fallback')
   shutil.copy2(root/'extracted'/f'{c}.json',obspath);stages['skeleton']='fresh_develop'
  obs=read(obspath);bind_inputs(fx,obs)
  condpath=root/'conditions.json';cached=a.conditions/c/'conditions.json' if a.conditions else None
  cache_ok=False
  if cached and cached.exists():
   try:
    cond=read(cached);bind_inputs(fx,obs,cond)
    if cond.get('observation_sha256')!=digest(obspath):raise ValueError('Conditions bound to different skeleton observations')
    cache_ok=True
   except (ValueError,KeyError) as e:stages['condition_cache_rejected']=str(e)
  if cache_ok:
   shutil.copy2(cached,condpath);stages['semantics']='bound_cache'
  elif a.vlm_mode=='auto':
   ok=command([sys.executable,str(HERE.parent/'refine015_relative/extract_overlap_v2.py'),'--rough',str(a.assets/fx['files']['rough']['path']),'--observations',str(obspath),'--out',str(root/'vlm'),'--model','gpt-6-luna','--reasoning-effort','low'],root/'vlm.log',events,'vlm')
   if ok:shutil.copy2(root/'vlm/conditions.json',condpath);stages['semantics']='fresh_gpt6luna_low'
   else:write(condpath,neutral_conditions(obs,obspath));stages['semantics']='failed_geometry_only'
  else:write(condpath,neutral_conditions(obs,obspath));stages['semantics']='cache_missing_geometry_only'
  goal,*_=targets(obs,fx['image_size']);mode_results={}
  if all(n in goal for n in ['LeftArm','RightArm']):
   initial_policy=measured_policy(current/'fit',obs,fx,a.assets)
   modes=[] if initial_policy.get('upper_body_fit_verified') else ['chest']
   stages['minute_upper_route']=dict(modes=modes,reason='already_measured_anchor' if not modes else 'joint_clavicle_chest_solve')
   for mode in modes:
    dest=root/mode;args=[sys.executable,str(SHOULDER/'run.py'),'--assets',str(a.assets),'--baseline',str(baseline),'--observations',str(obspath),'--out',str(dest),'--case',c]+(['--torso'] if mode=='chest' else [])
    if command(args,root/f'{mode}.log',events,mode) and audit(dest,baseline,a,events,mode+'_audit'):mode_results[mode]=read(dest/c/'fit/metrics.json')
   if len(mode_results)==2:selected=choose_mode(mode_results['clavicle'],mode_results['chest'])
   else:selected=next((m for m,d in mode_results.items() if d['geometry_accepted']),None)
   if selected:current=root/selected/c
   stages['shoulder_mode']=selected or 'preserved'
  else:stages['shoulder_mode']='missing_shoulders_preserved'
  upper=root/'upper';shutil.copytree(current,upper/c);current=upper/c;policy=measured_policy(current/'fit',obs,fx,a.assets);meta=read(current/'fit/metrics.json');meta['anchor_policy']=policy;write(current/'fit/metrics.json',meta);stages['upper_anchor_policy']=policy
  # Missing one ankle does not disable another observed chain. No lower evidence skips only this stage.
  lower=[n for n in goal if n.endswith(('Leg','Foot')) and n not in read(condpath).get('unobserved_joints',[])]
  if lower_available(lower):
   legs=root/'legs';args=[sys.executable,str(HERE/'leg_solver.py'),'--assets',str(a.assets),'--baseline',str(upper),'--observations',str(obspath),'--conditions',str(condpath),'--out',str(legs),'--case',c,'--skin-weight','150']
   if command(args,root/'legs.log',events,'legs'):
    shutil.copy2(current/'fixture.json',legs/c/'fixture.json')
    if audit(legs,upper,a,events,'legs_audit'):current=legs/c;stages['legs']='accepted' if read(current/'fit/metrics.json')['geometry_accepted'] else 'rejected_preserved_upper'
    else:stages['legs']='export_failed_preserved_upper'
   else:stages['legs']='solver_failed_preserved_upper'
  else:stages['legs']='no_observed_knee_preserved_upper'
 except Exception as e:
  stages['error']=str(e)
 # Always preserve the latest validated checkpoint, even if a downstream stage failed.
 final=a.out/'final';shutil.copytree(current,final/c);m=read(final/c/'fit/metrics.json');m['anchor_policy']=measured_policy(final/c/'fit',obs,fx,a.assets) if 'error' not in stages else dict(upper_body_fit_verified=False,reason='input_or_stage_error',lower_body_may_run=True);m['automatic_pipeline']=dict(stages=stages,events=events,manual_conditions_used=False,scope='selected_004006010_checkpoint_to_refined_export');write(final/c/'fit/metrics.json',m);write(root/'trace.json',dict(stages=stages,events=events));return dict(case_id=c,status='fit',stages=stages)

def main():
 p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--observations',type=Path);p.add_argument('--conditions',type=Path);p.add_argument('--cases',nargs='+');p.add_argument('--vlm-mode',choices=['auto','cache-only'],default='auto');p.add_argument('--extractor-checkout',type=Path);p.add_argument('--model-manifest',type=Path);p.add_argument('--blender',default=__import__('os').environ.get('REFINE_BLENDER','blender'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);cases=a.cases or sorted(p.parent.name for p in a.baseline.glob('*/fixture.json'));rows=[]
 if not cases:raise ValueError('No input fixtures')
 write(a.out/'run-config.json',dict(assets=str(a.assets.resolve()),baseline=str(a.baseline.resolve()),observations=str(a.observations) if a.observations else None,conditions=str(a.conditions) if a.conditions else None,cases=cases,vlm_mode=a.vlm_mode,model='gpt-6-luna',reasoning_effort='low',code_sha256={str(p.relative_to(HERE.parent)):digest(p) for folder in [HERE,SHOULDER] for p in folder.glob('*.py')}))
 for c in cases:
  try:row=run_case(c,a)
  except Exception as e:row=dict(case_id=c,status='input_failed',stages={'error':str(e)})
  rows.append(row);write(a.out/'final/results.json',[r for r in rows if r['status']=='fit']);print(c,row['stages'],flush=True)
 finish(a,rows)

def finish(a,rows):
 events=[];passed=audit(a.out/'final',a.baseline,a,events,'final_audit') if any(r['status']=='fit' for r in rows) else False;write(a.out/'status.json',dict(execution_complete=True,export_passed=passed,has_errors=any('error' in r['stages'] for r in rows),quality_certified=False,manual_conditions_used=False,cases=rows,events=events))
 for row in rows:
  if row['status']!='fit':continue
  c=row['case_id'];fit=a.out/'final'/c/'fit';m=read(fit/'metrics.json')
  if 'error' not in row['stages']:m['anchor_policy']=measured_policy(fit,read(a.out/'cases'/c/'observation.json'),read(a.baseline/c/'fixture.json'),a.assets)
  write(fit/'metrics.json',m)
 subprocess.run([sys.executable,str(HERE/'review.py'),'--root',str(a.out),'--assets',str(a.assets),'--baseline',str(a.baseline)],check=True)
 if not passed or any('error' in r['stages'] for r in rows):raise SystemExit(2)
if __name__=='__main__':main()
