"""Reproducible batch entry point. Existing 004 proposals are immutable anchors."""
import argparse,json,hashlib,copy,time,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from core import *
from lens import select_lens

def write_json(path,obj):path.write_text(json.dumps(obj,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--camera-cues',type=Path);p.add_argument('--depth-cues',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--cases',nargs='+');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 depth_docs=json.loads(a.depth_cues.read_text()) if a.depth_cues else {};records=[];source_hashes={}
 for row in json.loads((a.baseline/'results.json').read_text()):
  cid=row['case_id']
  if a.cases and cid not in a.cases:continue
  if row['status']!='fit':records.append(dict(case_id=cid,status='not_run',reason='baseline_invalid'));continue
  started=time.perf_counter();old=a.baseline/cid;dest=a.out/cid/'fit';dest.mkdir(parents=True)
  fixture=json.loads((old/'fixture.json').read_text());obspath=a.observations/f'{cid}.json';obs=json.loads(obspath.read_text());ih=fixture['files']['rough']['sha256']
  for f in fixture['files'].values():
   if hashlib.sha256((a.assets/f['path']).read_bytes()).hexdigest()!=f['sha256']:raise ValueError('Input changed')
  if not obs.get('producer','').startswith('develop HumanArt'):raise ValueError('Use develop observations; legacy Body fallback is disabled')
  if depth_docs.get(cid,{}).get('source')=='image_model' and depth_docs[cid].get('mapping') and depth_docs[cid].get('observation_sha256')!=hashlib.sha256(obspath.read_bytes()).hexdigest():raise ValueError('Depth overlay belongs to different observations; re-extract conditions')
  if obs['image_sha256']!=ih:raise ValueError('Wrong observation image')
  rig=json.loads((a.assets/fixture['files']['rig']['path']).read_text());names,ix,parents,local,base=prepare(rig);_,origin,torso,basis=frame(rig['bones'])
  pr=json.loads((old/'fit/adjustment.json').read_text());original_pr=copy.deepcopy(pr)
  with np.load(old/'fit/expected-world.npz') as d:original_world=d['world'].copy()
  for name in ['adjustment.json','expected-world.npz']:source_hashes[str((old/'fit'/name).resolve())]=hashlib.sha256((old/'fit'/name).read_bytes()).hexdigest()
  goal,kp,valid=targets(obs,fixture['image_size']);ids=[ix[n] for n in goal];tar=np.array(list(goal.values()))
  if len(ids)<4:records.append(dict(case_id=cid,status='not_run',reason='insufficient_interior_targets'));continue
  cuepath=a.camera_cues/f'{cid}.json' if a.camera_cues else None;cue=json.loads(cuepath.read_text()) if cuepath and cuepath.exists() else None
  rels=validate_relations(depth_docs.get(cid),ih,names)
  pr,camdiag=camera_stage(original_world,pr,basis,origin,ix,goal,cue,ih)
  world,edits,posediag=pose_stage(local,parents,base,ix,original_world,pr,goal,kp,valid,torso,rels,camdiag['changed'])
  pr['edits']=edits
  with np.load(a.assets/fixture['files']['mesh']['path']) as m:v,f,weights=m['vertices'],m['faces'],m['weights']
  original_vertices=skin(v,weights,original_world,base);candidate_vertices=skin(v,weights,world,base)
  olddef=deformation(original_vertices,v,f);newdef=deformation(candidate_vertices,v,f)
  shoulderdiag={'applied':False,'reason':'not_needed'}
  # Only repair if skin strain worsened; preserve 004 when the extra stage is unnecessary.
  if newdef['extreme_edges']>olddef['extreme_edges']:
   sw,se,sd=shoulder_stage(local,parents,ix,world,edits);sv=skin(v,weights,sw,base);sdef=deformation(sv,v,f)
   if sdef['extreme_edges']<newdef['extreme_edges'] and sd.get('endpoint_max_shift',1)<torso*.1:
    world,edits,candidate_vertices,newdef=sw,se,sv,sdef;pr['edits']=edits;shoulderdiag=sd
  def xy(w,p):return ((w[:,:3,3]-p['origin'])@np.array(p['camera']))[:,:2]*[p['pixel_scale'],-p['pixel_scale']]+p['offset']
  oldrmse=float(np.sqrt(np.mean((xy(original_world,original_pr)[ids]-tar)**2)));newrmse=float(np.sqrt(np.mean((xy(world,pr)[ids]-tar)**2)))
  db=float(np.linalg.norm(depth_errors(original_world,np.array(original_pr['camera']),ix,torso,rels)));da=float(np.linalg.norm(depth_errors(world,np.array(pr['camera']),ix,torso,rels)))
  accept,reasons=geometry_accept(olddef,newdef,oldrmse,newrmse,db,da,camdiag['changed'])
  # Reaching an allowed solver bound is recorded; actual skin/collision regression decides acceptance.
  proposal=a.out/cid/'proposal';proposal.mkdir();write_json(proposal/'adjustment.json',pr);np.savez_compressed(proposal/'expected-world.npz',names=names,world=world)
  regional=[]
  if not accept and not camdiag['changed']:
   rw,rp,rv,rd,regional=partial_pose(local,parents,base,ix,original_world,original_pr,pr,goal,torso,rels,v,f,weights)
   if regional:
    world,pr,candidate_vertices,newdef=rw,rp,rv,rd;accept=True
    newrmse=float(np.sqrt(np.mean((xy(world,pr)[ids]-tar)**2)));posediag['bound_hit_bones']=[]
  if not accept:world=original_world.copy();pr=copy.deepcopy(original_pr);candidate_vertices=original_vertices
  orientationdiag={'applied':False}
  if accept:
   ow,op,od=orientation_stage(local,parents,ix,world,pr,goal,cue);ov=skin(v,weights,ow,base);odef=deformation(ov,v,f)
   if od.get('applied') and odef['extreme_edges']<=olddef['extreme_edges'] and odef['ratio_max']<=max(2,olddef['ratio_max'])*1.05:
    world,pr,candidate_vertices,newdef=ow,op,ov,odef;orientationdiag=od
   elif od.get('applied'):orientationdiag=dict(od,applied=False,reason='orientation_deformation_gate')
  cam=np.array(pr['camera']);cj=(world[:,:3,3]-pr['origin'])@cam;cv=(candidate_vertices-pr['origin'])@cam;anchor=cj[ix['Hips']]
  lens=select_lens(cj,cv,anchor,ids,tar,torso,pr['pixel_scale'],pr['offset'])
  # Do not use a lens to disguise rejected geometry/depth evidence.
  if not accept and rels and da>db+1e-5 and lens['selected']=='perspective':lens.update(selected='orthographic',reason='rejected_depth_pose',inverse_distance=0,pixel_scale=pr['pixel_scale'],offset=pr['offset'])
  camera=dict(camera=pr['camera'],origin=pr['origin'],anchor_camera_xyz=anchor.tolist(),projection=lens['selected'],inverse_distance=lens['inverse_distance'],pixel_scale=lens['pixel_scale'],offset=lens['offset'],image_size=[535,687],embedded_in_fbx=False)
  length=max(abs(np.linalg.norm(world[i,:3,3]-world[j,:3,3])-np.linalg.norm(base[i,:3,3]-base[j,:3,3])) for i,j in enumerate(parents) if j>=0)
  if length>1e-6:raise ValueError('Bone lengths changed')
  met=dict(case_id=cid,manual_targets=False,orientation_stage=orientationdiag,files=fixture['files'],bound_hit_bones=posediag.get('bound_hit_bones',[]) if accept else [],rmse_px=newrmse if accept else oldrmse,max_bone_length_error=float(length),camera_stage=camdiag,pose_stage=posediag,shoulder_stage=shoulderdiag,geometry_accepted=accept,regional_acceptance=regional,full_proposal_rejection_reasons=reasons,rejection_reasons=[] if accept else reasons,old_rmse=oldrmse,proposal_rmse=newrmse,old_deformation=olddef,proposal_deformation=newdef,selected_depth_error=depth_errors(world,np.array(pr['camera']),ix,torso,rels).tolist(),depth_source=depth_docs.get(cid,{}).get('source'),depth_relations=rels,lens=lens,production_enabled=False,mesh_safety_certified=False,reference_used_for_fit=False,elapsed_seconds=time.perf_counter()-started)
  write_json(dest/'adjustment.json',pr);np.savez_compressed(dest/'expected-world.npz',names=names,world=world);write_json(dest/'metrics.json',met);write_json(dest/'camera.json',camera)
  if not accept:
   shutil.copy2(old/'fit/adjustment.json',dest/'adjustment.json');shutil.copy2(old/'fit/expected-world.npz',dest/'expected-world.npz')
  rough=ImageOps.pad(Image.open(a.assets/fixture['files']['rough']['path']).convert('RGB'),(535,687),color='white')
  ov=(original_vertices-original_pr['origin'])@np.array(original_pr['camera']);oi=render_projected(ov,f,ov[:,:2]*[original_pr['pixel_scale'],-original_pr['pixel_scale']]+original_pr['offset'],(535,687))
  ortho=render_projected(cv,f,project(cv,anchor,0,pr['pixel_scale'],pr['offset']),(535,687));selected=render_projected(cv,f,project(cv,anchor,lens['inverse_distance'],lens['pixel_scale'],lens['offset']),(535,687))
  selected.save(dest/'adjusted.png');ortho.save(dest/'orthographic.png')
  candidate_inv=lens.get('candidate_inverse_distance',0);pi=render_projected(cv,f,project(cv,anchor,candidate_inv,lens.get('candidate_scale',pr['pixel_scale']),lens.get('candidate_offset',pr['offset'])),(535,687));pi.save(dest/'perspective-proposal.png')
  sheet=Image.new('RGB',(2675,722),'white');draw=ImageDraw.Draw(sheet)
  for i,(im,title) in enumerate([(rough,'ROUGH'),(oi,'004 BASELINE'),(ortho,'NEW ORTHOGRAPHIC'),(pi,'PERSPECTIVE PROPOSAL'),(selected,'SELECTED '+lens['selected'])]):sheet.paste(im,(535*i,35));draw.text((535*i+8,10),cid+' '+title,fill='black')
  sheet.save(dest/'comparison.png');records.append(dict(case_id=cid,status='fit',geometry_accepted=accept,projection=lens['selected'],reasons=[] if accept else reasons));print(records[-1],flush=True)
  write_json(a.out/'results.json',records)
 write_json(a.out/'results.json',records)
 for path,digest in source_hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
 write_json(a.out/'preservation.json',source_hashes)
 write_json(a.out/'status.json',dict(status='provisional_requires_mesh_audit',production_enabled=False))
if __name__=='__main__':main()
