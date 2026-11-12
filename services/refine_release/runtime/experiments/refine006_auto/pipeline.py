"""Offline new-refine integration: autonomous depth route with immutable fallback."""
import argparse,json,copy,shutil,hashlib,sys,subprocess
from pathlib import Path
import numpy as np
from solve import solve
from core import targets,prepare,skin,deformation
import sys,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine_unified'))
from validity import shadow


def eligible(facts,goal):
    required=[s+n for s in ['Left','Right'] for n in ['Arm','ForeArm','Hand','UpLeg','Leg','Foot']]
    return facts.get('person_count')==1 and all(n in goal for n in required) and facts.get('raised_leg',{}).get('bend')=='backward'

def candidate_score(m):
    d=m['deformation'];return m['rmse_px']+.02*d['extreme_edges']+2*np.log(max(1,d['ratio_max']))

def select(candidates,baseline_deformation,old_error):
    passing=[]
    for index,m in enumerate(candidates):
        d=m['deformation']
        # Shared open-gesture prior: a lateral spread must not drift into forward reach.
        # The negative half-space is allowed, not asserted as VLM-measured depth.
        if m.get('gesture')=='bilateral_lateral_spread' and any(m['wrist_depth'][s]>.01*m['arm_lengths'][s] for s in ['Left','Right']):continue
        if m['rmse_px']<=old_error+1 and d['extreme_edges']<=baseline_deformation['extreme_edges'] and d['ratio_max']<=max(2,baseline_deformation['ratio_max'])*1.05 and m['max_bone_length_error']<1e-6:
            passing.append((candidate_score(m),index))
    return min(passing)[1] if passing else None

def main():
 p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--semantics',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cases',nargs='+',required=True);p.add_argument('--blender',type=Path,required=True);p.add_argument('--skip-audit',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);rows=[];manifest=[]
 for cid in a.cases:
  old=a.baseline/cid/'fit';fixture=json.loads((a.baseline/cid/'fixture.json').read_text());fixture={k:v for k,v in fixture.items() if k in ['case_id','files','image_size']};obs=json.loads((a.observations/f'{cid}.json').read_text());sem=json.loads((a.semantics/f'{cid}.json').read_text());goal,_,_=targets(obs,fixture['image_size']);dest=a.out/cid/'fit';dest.mkdir(parents=True)
  if sem['image_sha256']!=obs['image_sha256'] or obs['image_sha256']!=fixture['files']['rough']['sha256']:raise ValueError('Image binding mismatch')
  oldpr=json.loads((old/'adjustment.json').read_text());oldmetrics=json.loads((old/'metrics.json').read_text());rig=json.loads((a.assets/fixture['files']['rig']['path']).read_text());names,ix,parents,local,base=prepare(rig)
  with np.load(old/'expected-world.npz') as d:oldw=d['world']
  with np.load(a.assets/fixture['files']['mesh']['path']) as d:v,f,weights=d['vertices'],d['faces'],d['weights']
  olddef=deformation(skin(v,weights,oldw,base),v,f);oldxy=((oldw[:,:3,3]-oldpr['origin'])@np.array(oldpr['camera']))[:,:2]*[oldpr['pixel_scale'],-oldpr['pixel_scale']]+oldpr['offset'];ids=[ix[n] for n in goal];oldrmse=float(np.sqrt(np.mean((oldxy[ids]-np.array(list(goal.values())))**2)))
  route=eligible(sem['raw'],goal);candidates=[];paths=[];failures=[]
  if route:
   hypotheses=[0.,-.2] if sem['raw'].get('gesture')=='bilateral_lateral_spread' else [0.]
   for i,bias in enumerate(hypotheses):
    cp=a.out/cid/'candidates'/str(i)
    try:m=solve(a.assets,fixture,obs,sem,cp,bias)
    except RuntimeError as exc:
     cp.mkdir(parents=True,exist_ok=True);failure=dict(candidate=i,lateral_bias=bias,error=str(exc),status='solver_failed');(cp/'failure.json').write_text(json.dumps(failure,indent=2));failures.append(failure);continue
    candidates.append(m);paths.append(cp)
  gated_selected=select(candidates,olddef,oldrmse)
  checks=[]
  for i,m in enumerate(candidates):
   d=m['deformation']
   checks.append(dict(candidate=int(paths[i].name),converged=m['solver_success'],projection_regression=m['rmse_px']>oldrmse+1,
    skin_count_regression=d['extreme_edges']>olddef['extreme_edges'],skin_peak_regression=d['ratio_max']>max(2,olddef['ratio_max'])*1.05,
    lateral_forward_regression=m.get('gesture')=='bilateral_lateral_spread' and any(m['wrist_depth'][s]>.01*m['arm_lengths'][s] for s in ['Left','Right']),
    bone_length_error=m['max_bone_length_error']))
  eligible_ids=[i for i,m in enumerate(candidates) if m['max_bone_length_error']<1e-6 and np.isfinite(candidate_score(m))]
  selected=min(eligible_ids,key=lambda i:(candidate_score(candidates[i]),i),default=None) if shadow() else gated_selected
  if selected is None:
   for name in ['adjustment.json','expected-world.npz']:shutil.copy2(old/name,dest/name)
   pr=oldpr;w=oldw;met=copy.deepcopy(oldmetrics);met.update(case_id=cid,files=fixture['files'],rmse_px=oldrmse,bound_hit_bones=[],manual_targets=False,geometry_accepted=False,rejection_reasons=['unsupported_full_body_evidence' if not route else 'no_safe_depth_candidate'])
  else:
   for file in paths[selected].iterdir():shutil.copy2(file,dest/file.name)
   pr=json.loads((dest/'adjustment.json').read_text());met=copy.deepcopy(candidates[selected]);met.update(geometry_accepted=True,rejection_reasons=[])
   with np.load(dest/'expected-world.npz') as d:w=d['world']
  anchor=((w[ix['Hips'],:3,3]-pr['origin'])@np.array(pr['camera'])).tolist();camera=dict(camera=pr['camera'],origin=pr['origin'],anchor_camera_xyz=anchor,projection='orthographic',inverse_distance=0,pixel_scale=pr['pixel_scale'],offset=pr['offset'],embedded_in_fbx=False)
  (dest/'camera.json').write_text(json.dumps(camera,indent=2));met.update(old_rmse=oldrmse,old_deformation=olddef,lens=dict(selected='orthographic',reason='depth_route',inverse_distance=0,pixel_scale=pr['pixel_scale'],offset=pr['offset']),orientation_stage={},regional_acceptance=[],production_enabled=False,manual_conditions_used=False,reference_used_for_fit=False)
  (dest/'metrics.json').write_text(json.dumps(met,indent=2));rows.append(dict(case_id=cid,status='fit',geometry_accepted=selected is not None,projection='orthographic'))
  manifest.append(dict(case_id=cid,eligible=route,shadow=shadow(),gated_selected_candidate=int(paths[gated_selected].name) if gated_selected is not None else None,quality_checks=checks,selected_candidate=int(paths[selected].name) if selected is not None else None,candidate_scores=[candidate_score(m) for m in candidates],candidates=[str(p) for p in paths],solver_failures=failures,observation_sha256=hashlib.sha256((a.observations/f'{cid}.json').read_bytes()).hexdigest(),semantic_sha256=hashlib.sha256((a.semantics/f'{cid}.json').read_bytes()).hexdigest(),fallback_sha256=hashlib.sha256((old/'expected-world.npz').read_bytes()).hexdigest()))
  print(cid,'candidate',selected,flush=True)
 (a.out/'results.json').write_text(json.dumps(rows,indent=2));(a.out/'selection.json').write_text(json.dumps(manifest,indent=2));(a.out/'status.json').write_text(json.dumps({'status':'provisional_requires_mesh_audit'}))
 if a.skip_audit:return
 cmd=[str(a.blender),'-b','--python-exit-code','1','--python',str(Path(__file__).resolve().parents[1]/'refine_new/audit_blender.py'),'--',str(a.out),str(a.baseline),str(a.assets)]
 with (a.out/'audit.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
 print(a.out)
if __name__=='__main__':main()
