"""Original selected pose -> 004/010 -> 006 -> 015/Anchor -> terminal lens."""
import argparse,os,subprocess,time,traceback,signal
from concurrent.futures import ThreadPoolExecutor
from common import *
from semantics import extract_once
from projection import choose_projection,render
from head import apply_head
sys.path.insert(0,str(HERE.parent/'refine_auto'))
from pipeline import normalize_checkpoint,neutral_conditions
sys.path.insert(0,str(HERE.parent/'refine006_auto'))
from regional import bend

# Never accept an already refined checkpoint as an input argument.
def run_command(command,log,events,stage,shadow=False):
 start=time.monotonic();env=dict(os.environ,STANDIN_EXPERIMENT_QUALITY='shadow' if shadow else 'preserve',POSE_CANARY_STAGE='shadow',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
 with log.open('w') as f:
  proc=subprocess.Popen(list(map(str,command)),stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=os.getenv("REFINE_MANAGED_PROCESS")!="1")
  try:code=proc.wait(timeout=900)
  except subprocess.TimeoutExpired:
   proc.terminate() if os.getenv("REFINE_MANAGED_PROCESS")=="1" else os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=5)
   except subprocess.TimeoutExpired:proc.kill() if os.getenv("REFINE_MANAGED_PROCESS")=="1" else os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   code=124
 events.append(dict(stage=stage,returncode=code,seconds=time.monotonic()-start,log=str(log)));return code==0

def arm_guard(before,after,fx,obs,assets):
 names,ix,_,_,_=prepare(read(assets/fx['files']['rig']['path']));a=np.load(before/'fit/expected-world.npz')['world'];b=np.load(after/'fit/expected-world.npz')['world'];cam=np.array(read(before/'fit/adjustment.json')['camera']);goal,*_=targets(obs,fx['image_size']);reasons=[]
 for side in ['Left','Right']:
  ids=[ix[side+n] for n in ['Arm','ForeArm','Hand']];pa=a[ids,:3,3];pb=b[ids,:3,3];length=np.linalg.norm(np.diff(pa,axis=0),axis=1).sum();da=((pa[2]-pa[0])@cam)[2]/length;db=((pb[2]-pb[0])@cam)[2]/length
  if abs(da)>.08 and (db*da<0 or abs(db)<abs(da)-.03):reasons.append(side+'_established_arm_depth_regression')
  if all(side+n in goal for n in ['Arm','ForeArm','Hand']):
   rough=bend(np.array([np.r_[goal[side+n],0] for n in ['Arm','ForeArm','Hand']]))
   if rough<20 and bend(pb)>max(20,bend(pa)+3):reasons.append(side+'_straight_arm_regression')
 return reasons

def audit(root,baseline,a,events,label):
 if label=='004010_audit' and not getattr(a,'shadow',False):
  repair_cmd=[a.blender,'-b','--python-exit-code','1','--python',HERE/'collision_repair_blender.py','--',root,baseline,a.assets]
  if not run_command(repair_cmd,root/'local-collision.log',events,label+'_local_repair'):return False
 cmd=[sys.executable,HERE.parent/'refine_shoulder_line/audit.py','--run',root,'--baseline',baseline,'--assets',a.assets,'--blender',a.blender]
 ok=run_command(cmd,root/'audit-command.log',events,label,getattr(a,'shadow',False))
 return ok and (root/'audit.json').exists() and all(r['export_passed'] for r in read(root/'audit.json'))

def run_case(c,a):
 root=a.out/c;root.mkdir();events=[];shadow=getattr(a,'shadow',False);prepared=getattr(a,'prepared',None);state=dict(case=c,manual_conditions_used=False,ground_truth_used=False,stages={});obsdir=root/'observations'
 def cmd(script,args,label):return run_command([sys.executable,HERE.parent/script,*args],root/f'{label}.log',events,label,shadow)
 try:
  if prepared:
   sp=prepared/c/'skeleton-status.json';row=read(sp)
   events.append(dict(stage='skeleton',shared=True,returncode=0 if row['status']=='complete' else 1,seconds=row['seconds']))
   if row['status']!='complete':raise ValueError('Skeleton unavailable: '+row.get('error',''))
   shutil.copytree(prepared/c/'observations',obsdir)
  elif not cmd('refine_new/extract_develop.py',['--checkout',a.extractor_checkout,'--model-manifest',a.model_manifest,'--assets',a.assets,'--out',obsdir,'--cases',c],'skeleton'):raise ValueError('Skeleton unavailable; no mock fallback')
  obs=read(obsdir/f'{c}.json');source=root/'source';fx=source_checkpoint(a.assets,c,obs,source/c);current=source/c;write(source/'results.json',[dict(case_id=c,status='fit')]);state['source_pose']=read(source/c/'provenance.json')
  sem=root/'semantics';cues=root/'camera-cues';body=root/'body-cues';conditions=root/'conditions'
  try:
   cache=a.semantic_cache/f'{c}.json' if a.semantic_cache else None
   if prepared and (not cache or not cache.exists()):raise ValueError('No exact bound Luna response for this cut')
   extract_once(a.assets/fx['files']['rough']['path'],obsdir/f'{c}.json',sem,cache)
   cues.mkdir();body.mkdir();(conditions/c).mkdir(parents=True);write(cues/f'{c}.json',__import__('camera_evidence').route_camera(read(sem/'camera.json')));shutil.copy2(sem/'body.json',body/f'{c}.json');shutil.copy2(sem/'conditions.json',conditions/c/'conditions.json');state['stages']['vlm']='fresh_shared_luna_low' if getattr(a,'fresh_prepared',None) or not a.semantic_cache else 'cached_shared_luna_low'
  except Exception as e:
   if obs.get('ownership_requires_single_person_vlm') or 'ownership' in str(e).lower():raise ValueError('Single-person semantic ownership unavailable') from e
   state['stages']['vlm']='failed_geometry_only';state['semantic_error']=str(e);write(cues/f'{c}.json',dict(image_sha256=obs['image_sha256'],raw=dict(person_count=1,torso_view='uncertain',nearer_shoulder='uncertain',confidence='weak',elevation='uncertain')));write(body/f'{c}.json',dict(image_sha256=obs['image_sha256'],raw=dict(person_count=1,arms=[],torso={},head={},raised_leg={},gesture='uncertain')));write(conditions/c/'conditions.json',neutral_conditions(obs,obsdir/f'{c}.json'))
  base_view=read(cues/f'{c}.json')
  profile=base_view['raw'].get('torso_view')=='profile'
  direction_root=getattr(a,'profile_directions',None);direction_file=direction_root/f'{c}.json' if direction_root else (sem/'profile-direction.json' if (sem/'profile-direction.json').exists() else None)
  from profile_direction import select
  signed,reason=select(base_view,read(direction_file) if profile and direction_file and direction_file.exists() else None)
  state['profile_direction']=reason
  if profile and signed is None:
   state['stages']['profile']='ambiguous_sign_preserve_source_view';profile=False
   # Keep the observed profile cue for the terminal gate; do not fabricate a
   # signed view or suppress otherwise available limb observations.
  initial=root/'camera-pose'
  gated=getattr(a,'gated_root',None)
  shared=None
  if gated and not profile:
   for name in ['camera-pose','partial-camera-pose']:
    g=gated/c/name/c/'unfiltered-proposal'
    if g.exists() and (name=='camera-pose' or not (gated/c/'camera-pose'/c/'fit'/'metrics.json').exists()):
     if name=='partial-camera-pose':
      if 'not converged:' in (gated/c/'004010.log').read_text():continue
      gm=read(g/'metrics.json')
      if any(gm.get(stage,{}).get('converged') is False for stage in ('camera_stage','pose_stage')):continue
     shared=g;initial=root/name;break
  if profile:
   ok=cmd('refine_unified/profile_initial.py',['--assets',a.assets,'--baseline',source,'--cues',cues,'--direction',direction_file,'--out',initial,'--case',c],'004010')
  elif shared:
   shutil.copytree(shared,initial/c/'fit')
   write(initial/'results.json',[dict(case_id=c,status='fit')]);ok=True
   state['shared_initial_candidate_sha256']=sha(shared/'expected-world.npz')
   events.append(dict(stage='004010',returncode=0,shared=True))
  else:ok=cmd('refine_manual_004/semantic_camera.py',['--assets',a.assets,'--baseline',source,'--cues',cues,'--out',initial,'--cases',c],'004010')
  if not profile and (not ok or any(r.get('mode')=='baseline_preserved_uncertain_cues' for r in read(initial/'results.json'))):
   initial=root/'partial-camera-pose';ok=cmd('refine_unified/partial_pose.py',['--assets',a.assets,'--baseline',source,'--cues',cues,'--out',initial,'--case',c],'004010_partial')
  if ok and (initial/c/'fit/expected-world.npz').exists():
   shutil.copy2(source/c/'fixture.json',initial/c/'fixture.json');normalize_checkpoint(initial/c)
   from skin_policy import repair, SurfaceGate
   fitted=read(initial/c/'fit/adjustment.json');prior=read(source/c/'fit/adjustment.json')
   shutil.copytree(initial/c/'fit',initial/c/'unfiltered-proposal')
   goals,*_=targets(obs,fx['image_size'])
   with np.load(a.assets/fx['files']['mesh']['path']) as mesh:
    rig=read(a.assets/fx['files']['rig']['path'])
    if shadow:
     from validity import check_fit
     check_fit(initial/c/'fit',rig,a.assets/fx['files']['source']['path'])
     names,ix,_,_,base=prepare(rig)
     w=np.load(initial/c/'fit/expected-world.npz')['world'];edits=fitted['edits']
     vertices=mesh['vertices'];weights=mesh['weights']
     policy=SurfaceGate(vertices,mesh['faces'],weights,names,np.linalg.norm(base[ix['Spine2'],:3,3]-base[ix['Hips'],:3,3]))
     before=policy.measure(vertices);proposal=policy.measure(skin(vertices,weights,w,base))
     failures=policy.failures(before,proposal)
     diagnostic=dict(before=before,proposal=proposal,selected=proposal,shadow=True,accepted=not failures,
                     would_reject=bool(failures),failures=failures,thresholds=dict(excess_torso=.005,regional_area_fraction=.01,peak_excess_torso=.04))
    else:edits,w,diagnostic=repair(rig,mesh,prior,fitted,goals,prepare)
   safe=diagnostic['accepted'];met=read(initial/c/'fit/metrics.json')
   if not safe and not shadow:raise ValueError('Surface repair could not find a safe local fallback')
   fitted['edits']=edits;write(initial/c/'fit/adjustment.json',fitted)
   if not shadow:np.savez_compressed(initial/c/'fit/expected-world.npz',names=prepare(read(a.assets/fx['files']['rig']['path']))[0],world=w)
   met.update(geometry_accepted=True,skin_surface=diagnostic,rejection_reasons=[])
   fitted=read(initial/c/'fit/adjustment.json');selected=np.load(initial/c/'fit/expected-world.npz')['world'];_,idx,_,_,_=prepare(read(a.assets/fx['files']['rig']['path']));goals,*_=targets(obs,fx['image_size']);xy=((selected[:,:3,3]-fitted['origin'])@fitted['camera'])[:,:2]*[fitted['pixel_scale'],-fitted['pixel_scale']]+fitted['offset'];met['rmse_px']=float(np.sqrt(np.mean([(xy[idx[n]]-v)**2 for n,v in goals.items()])));met['lens']=dict(selected='orthographic',inverse_distance=0,pixel_scale=fitted['pixel_scale'],offset=fitted['offset']);write(initial/c/'fit/camera.json',camera(fitted,selected,idx))
   write(initial/c/'fit/metrics.json',met)
   if audit(initial,source,a,events,'004010_audit'):current=initial/c;state['stages']['004010']='audited_shadow' if shadow else 'audited'
   else:state['stages']['004010']='export_failed_preserved_original'
  else:state['stages']['004010']='failed_original_preserved'
  view_anchor=current
  def guarded(prior,candidate,label):
   if not profile:return candidate
   from view_contract import enforce
   return enforce(view_anchor,prior,candidate,fx,a.assets,root/(label+'-view-contract.json'))
  baseline=root/'depth-input' ;copy_checkpoint(current,baseline/c);write(baseline/'results.json',[dict(case_id=c,status='fit')])
  depth=root/'depth';ok=cmd('refine_new/run_depth_auto.py',['--assets',a.assets,'--baseline',baseline,'--out',depth,'--cases',c,'--blender',a.blender,'--observations-cache',obsdir,'--semantics-cache',body],'006')
  if ok and (depth/'final'/c/'fit/expected-world.npz').exists():
   proposal=depth/'final'/c;shutil.copy2(source/c/'fixture.json',proposal/'fixture.json');chosen=guarded(current,proposal,'006');state['stages']['006']='audited_full_and_regional' if chosen==proposal else 'view_contract_preserved_prior';current=chosen
  else:state['stages']['006']='failed_preserved_prior'
  headroot=root/'head';state['head_stage']=apply_head(current,headroot/c,fx,obs,read(body/f'{c}.json')['raw'],a.assets,shadow=shadow);write(headroot/'results.json',[dict(case_id=c,status='fit')])
  if audit(headroot,current.parent,a,events,'head_audit'):current=guarded(current,headroot/c,'head')
  anchor_input=root/'anchor-input';copy_checkpoint(current,anchor_input/c);write(anchor_input/'results.json',[dict(case_id=c,status='fit')]);anchor=root/'anchor'
  ok=cmd('refine_auto/pipeline.py',['--assets',a.assets,'--baseline',anchor_input,'--out',anchor,'--observations',obsdir,'--conditions',conditions,'--vlm-mode','cache-only','--cases',c,'--blender',a.blender],'015_anchor')
  if ok and read(anchor/'status.json')['export_passed']:
   proposal=anchor/'final'/c;reasons=arm_guard(anchor_input/c,proposal,fx,obs,a.assets);state['arm_preservation_rejections']=reasons;state['anchor_details']=read(anchor/'status.json')
   if not reasons or shadow:
    chosen=guarded(current,proposal,'015_anchor');state['stages']['015_anchor']=('audited_shadow' if shadow else 'audited') if chosen==proposal else 'view_contract_preserved_prior';current=chosen
   else:state['stages']['015_anchor']='depth_regression_preserved_006'
  else:state['stages']['015_anchor']='failed_preserved_prior'
  final=root/'final';copy_checkpoint(current,final/c);write(final/'results.json',[dict(case_id=c,status='fit')]);normalize_checkpoint(final/c)
  previous=current/'fit';dest=final/c/'fit';check=previous/'export-check.json'
  reusable=check.exists() and sha(previous/'expected-world.npz')==sha(dest/'expected-world.npz') and read(previous/'adjustment.json')==read(dest/'adjustment.json') and bool(list(previous.glob('*.fbx')))
  if reusable:
   for item in [check,*previous.glob('*.fbx')]:shutil.copy2(item,dest/item.name)
   write(final/'audit.json',[dict(case_id=c,export_passed=True,reused_exact_validated_pose=True,source=str(previous))]);events.append(dict(stage='final_audit',seconds=0.,returncode=0,exact_export_reused=True,final_terminal_reimport_required=True))
  elif not audit(final,current.parent,a,events,'final_audit'):raise ValueError('Final FBX audit failed')
  state['lens']=choose_projection(final/c/'fit',fx,obs,a.assets,read(conditions/c/'conditions.json'),shadow=shadow);state['stages']['projection']='terminal_frozen_pose';state['export_passed']=True
  from PIL import Image,ImageDraw,ImageOps
  ims=[ImageOps.pad(Image.open(a.assets/fx['files']['rough']['path']).convert('RGB'),(535,687),color='white'),render(source/c/'fit',fx,a.assets,read(final/c/'fit/camera.json')),render(final/c/'fit',fx,a.assets)]
  sheet=Image.new('RGB',(1605,730),'white');draw=ImageDraw.Draw(sheet)
  for i,(im,label) in enumerate(zip(ims,['ROUGH','ORIGINAL / FINAL CAMERA','UNIFIED FINAL'])):sheet.paste(im,(535*i,35));draw.text((535*i+8,8),c+' '+label,fill='black')
  sheet.save(root/'comparison.png');render(source/c/'fit',fx,a.assets).save(root/'original-camera.png');state['status']='complete';state['quality_certified']=False
 except Exception as e:state.update(status='failed',error=str(e),traceback=traceback.format_exc())
 state['events']=events;state['mode']='shadow' if shadow else 'preserve';write(root/'status.json',state);print(c,state['status'],state.get('error',''),flush=True);return state

def main():
 p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cases',nargs='+',required=True);p.add_argument('--extractor-checkout',type=Path,required=True);p.add_argument('--model-manifest',type=Path,required=True);p.add_argument('--semantic-cache',type=Path);p.add_argument('--workers',type=int,choices=[1,2],default=1);p.add_argument('--blender',default=__import__('os').environ.get('REFINE_BLENDER','blender'));a=p.parse_args();a.assets=a.assets.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
 write(a.out/'config.json',dict(version='New Refine Preserve v1',cases=a.cases,initial_pose='manifest_bound_original_selected_pose',assets=str(a.assets),extractor_checkout=str(a.extractor_checkout),model_manifest_sha256=sha(a.model_manifest),semantic_cache=str(a.semantic_cache) if a.semantic_cache else None,model='gpt-6-luna',reasoning_effort='low',code_hashes={str(f.relative_to(HERE.parent)):sha(f) for folder in ['refine_unified','refine_auto','refine006_auto','refine015_relative','refine_manual_004'] for f in (HERE.parent/folder).glob('*.py')}))
 rows=[]
 with ThreadPoolExecutor(max_workers=a.workers) as pool:
  for result in pool.map(lambda c:run_case(c,a),a.cases):rows.append(result);write(a.out/'results.json',rows)
 import html
 cards=''.join(f'<h2>{r["case"]}: {r["status"]}</h2>'+ (f'<img src="{r["case"]}/comparison.png">' if (a.out/r['case']/'comparison.png').exists() else '')+'<pre>'+html.escape(json.dumps(r,ensure_ascii=False,indent=2))+'</pre>' for r in rows)
 (a.out/'report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1600px;margin:30px auto}img{width:100%}pre{white-space:pre-wrap}</style><h1>New Refine Unified v1 · original pose to final</h1><p>Original selected pose → 004/010 → 006 full/regional depth → 015/Auto Anchor → frozen-pose lens. All fallbacks are recorded; export success is not visual certification. No production API changes.</p>'+cards)
 if any(r['status']!='complete' for r in rows):raise SystemExit(2)
if __name__=='__main__':main()
