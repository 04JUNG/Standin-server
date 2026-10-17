"""Generic scene-level evidence policy, no cut-specific angles, joints, poses or labels."""
from pathlib import Path
import json,sys,re,math,hashlib,os
import numpy as np
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1];BASE=ROOT.parent/'04j-289-set34-refine-scenes';PRIOR=ROOT.parent/'04j-289-set3-orientation-v3';LAYOUT=ROOT.parent/'04j-289-set3-layout-v2'
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)) and any(s in str(args[0]) for s in ('feedback-eval','evaluation-only','/gt/','visual-review')):raise RuntimeError('Evaluation feedback blocked')
sys.addaudithook(guard);sys.path.insert(0,str(PRIOR));sys.path.insert(0,str(BASE));sys.path.insert(0,str(WORK/'.worktrees/standin-fullbody-routing'));os.environ['REFINE_POSE_QUARANTINE_PATH']=str(WORK/'.worktrees/standin-fullbody-routing/config/refine_pose_quarantine.v1.json')
from evidence_policy import YAW,wrap,measures,posture_cost
from align_refine import alignment,align_bvh
from src.semantic_adapter_285 import excluded_pose_ids
excluded=excluded_pose_ids();library=[r for r in json.loads((BASE/'contact-index.json').read_text())['poses'] if r['pose_id'] not in excluded];lookup={r['pose_id']:r for r in library};metrics={r['pose_id']:measures(r['points']) for r in library};lib=Path(json.loads((WORK/'tmp/standin-local-routing/latest-bound-library.json').read_text())['path'])
def pid(v):return int(str(v).replace('P',''))
def point(v,vp):return np.array(v,float)*(vp[2:]-vp[:2])/1000+vp[:2]
def rect(v,vp):return np.array(v,float)*np.tile(vp[2:]-vp[:2],2)/1000+np.tile(vp[:2],2)
def mid(b):return (np.array(b[:2])+b[2:])/2
def tags(r,field):return {t['value'] for g in (r.get('tags') or {}).values() for t in g if t['field']==field}
def predict(row,yaw,anchor,camera):
 p=np.asarray(row['points'],float);p=p@alignment(p,yaw).T;torso=np.linalg.norm((p[5]+p[6]-p[11]-p[12])/2);p=(p-p[0])*.50/max(torso,1e-6);height=anchor[3]-anchor[1];depth=camera['focal_pixels']*.22/height;depths=depth-p[:,2]
 if depths.min()<.2:return None
 return mid(anchor)+camera['focal_pixels']*np.column_stack((p[:,0],-p[:,1]))/depths[:,None]
config=dict(character=str(WORK/'pose-head-contact-repair-20261015/mesh-review/assets/standin-master-v2.fbx'),people={},manual_inputs=False)
for item in json.loads((ROOT/'inputs.json').read_text()):
 cut=item['cut_id'];folder=ROOT/cut;source=json.loads((BASE/cut/'people.json').read_text());prior={p['person_index']:p for p in json.loads((PRIOR/cut/'selection.json').read_text())['people']};prior_geo={p['person_index']:p for p in json.loads((PRIOR/cut/'geometry.json').read_text())['people']};old=json.loads((LAYOUT/cut/'layout.json').read_text());oldrows={p['person_index']:p for p in old['people']};cfg=old['camera'];size=np.array([cfg['width'],cfg['height']]);vp=np.array(json.loads((folder/'identities.json').read_text())['analysis_viewport_bbox']);fresh={p['person_index']:p for p in json.loads((folder/'fresh-keypoints.json').read_text())['people']};surface={pid(p['person_index']):p for p in json.loads((folder/'surfaces-review.json').read_text())['analysis']['people']};contrast={p['person_index']:p['rankings'] for p in json.loads((folder/'contrastive.json').read_text())['people']};roles=json.loads((folder/'roles-review.json').read_text())['analysis'];roles=roles if isinstance(roles,dict) else {};relations=roles.get('relations',[]);headboxes={i:rect(p['head_bbox'],vp) for i,p in surface.items()};owned_hands={p['person_index']:[] for p in source['people']};ownership=[]
 # A detached large hand is assigned using apparent head/hand scale, not nearest head.
 for candidate in json.loads((folder/'localized-hands.json').read_text())['hands']:
  a=candidate['analysis'];b=np.array(candidate['crop_bbox'],float)
  if not a.get('visible') or not isinstance(a.get('x'),(int,float)) or not isinstance(a.get('y'),(int,float)) or not isinstance(a.get('bbox'),list):continue
  center=point([a['x'],a['y']],b);hb=rect(a['bbox'],b);handheight=max(hb[3]-hb[1],hb[2]-hb[0]);costs={}
  for i,head in headboxes.items():
   hh=max(10,head[3]-head[1]);costs[i]=abs(math.log(handheight/(.85*hh)))+.12*np.linalg.norm(center-mid(head))/hh
  owner=min(costs,key=costs.get);prior_box=np.array(next(p['bbox'] for p in source['people'] if p['person_index']==owner));detached=not bool(((center>=prior_box[:2])&(center<=prior_box[2:])).all());record=dict(owner=owner,center=center.tolist(),bbox=hb.tolist(),costs=costs,detached_from_old_crop=detached,localized=a)
  if detached:owned_hands[owner].append(record)
  ownership.append(record)
 # Relations must be geometrically consistent with a person lying between two targets.
 restraining={}
 for r in relations:
  if r.get('type')=='restraining':restraining.setdefault(pid(r['actor']),set()).add(pid(r['target']))
 restraining={i:sorted(ts) for i,ts in restraining.items() if len(ts)>=2 and min(source['people'][t]['head_center'][0] for t in ts)<source['people'][i]['head_center'][0]<max(source['people'][t]['head_center'][0] for t in ts)}
 rows=[]
 for person in source['people']:
  i=person['person_index'];a=json.loads((folder/f'person_{i}-analysis.json').read_text())['analysis'];s=surface[i];f=fresh[i];kp=np.array(f['keypoints']);scores=np.array(f['scores']);oldp=oldrows[i];pg=prior_geo[i];notes=[];prior_yaw=pg['torso_camera_yaw'];body=prior_yaw;label=s.get('torso_heading','unknown');head=s.get('head_heading','unknown');oldanchor=np.array(oldp['final_head_bbox']);headheight=oldanchor[3]-oldanchor[1];nasal=kp[0]-kp[3:5].mean(0);profile=bool(scores[0]>.5 and scores[3:5].mean()>.45 and abs(nasal[0])/headheight>.12);nose_side='right' if nasal[0]>0 else 'left';shoulder_reliable=bool(scores[5:7].min()>.5 and f['owned_head_match']);posterior=bool(kp[5,0]<kp[6,0]);new_surface=s.get('torso_surface');clipback=contrast[i]['surface'][0]['label']=='back'
  if new_surface=='side':
   if shoulder_reliable and posterior and abs(prior_yaw)>=120 and clipback:notes.append('posterior_torso_confirmed_keep_existing_back_view')
   else:
    side=nose_side if profile else ('right' if label in ('right','front_right','back_right') else 'left');body=YAW[side];notes.append('side_surface_with_owned_nose_ear_profile' if profile else 'independent_side_surface_agreement')
  elif new_surface=='back':body=180.;notes.append('fresh_back_surface_confirmed')
  elif new_surface=='front':
   if shoulder_reliable and not posterior and profile:
    ratio=abs(kp[5,0]-kp[6,0])/max(headheight,1);angle=math.degrees(math.acos(np.clip(ratio/1.6,0,1)));body=(90. if angle>67.5 else 45.)*(1 if nose_side=='right' else -1);notes.append('front_surface_shoulder_order_and_profile_agree')
   elif abs(prior_yaw)>=120 and clipback:notes.append('ambiguous_front_claim_rejected_by_posterior_evidence')
   elif label in YAW:body=YAW[label]
  if profile:head=nose_side
  if head not in YAW:head=next((k for k,v in YAW.items() if v==body),'front')
  if owned_hands[i]:
   side='right' if owned_hands[i][0]['center'][0]>mid(oldanchor)[0] else 'left';body=YAW[side];head=side;notes.append('owned_detached_hand_direction_coherent_with_side_surface')
  # Missing lower regions at a frame edge cannot justify shrinking a proven full rig.
  lowest=str(s.get('lowest_anatomy',''))+' '+str(s.get('scope_evidence',''));positive_legs=bool(re.search(r'발목|정강|무릎|바지.*다리|다리 일부|feet|shins|knees',lowest,re.I)) or a.get('lowest_visible_anatomy') in ('feet','shins','knees','thighs');scope=pg['output_scope'];frame_clipped=person['bbox'][3]>=size[1]*.96
  if positive_legs:scope='full';notes.append('positive_leg_evidence_restore_full')
  elif frame_clipped and oldp['output_scope']=='full':scope='full';notes.append('frame_clipped_lower_body_keep_existing_full')
  if owned_hands[i] and scope in ('head','bust'):scope='half';notes.append('owned_detached_hand_requires_arms_in_output')
  anchor=oldanchor.copy();scale_reason='preserved_previous_head_height'
  if owned_hands[i]:
   hb=headboxes[i];hh=hb[3]-hb[1]
   if hh>headheight*1.35 and .35<(hb[2]-hb[0])/max(hh,1)<1.8:
    center=mid(oldanchor);aspect=(oldanchor[2]-oldanchor[0])/headheight;anchor=np.r_[center-[hh*aspect/2,hh/2],center+[hh*aspect/2,hh/2]];scale_reason='fresh_full_head_box_required_by_detached_hand_scale';notes.append('old_profile_face_proxy_underestimated_whole_head')
  head_yaw=YAW[head];relative=float(np.clip(wrap(head_yaw-body),-85,85));pitch=None
  action_scores=contrast[i]['action'];phone=action_scores[0]['label']=='phone' and action_scores[0]['similarity']-action_scores[1]['similarity']>.025
  if phone:pitch=-25.;notes.append('phone_action_retains_downward_head_attention')
  goals=[];interaction='preserve';posture=prior[i]['posture'];pose_id=prior[i]['pose_id']
  if i in restraining:
   interaction='restraining';posture='standing'
   for side,elbowj,wristj in [('Left',7,9),('Right',8,10)]:
    if scores[wristj]>.4:
     target=kp[wristj].copy();nearest=min(restraining[i],key=lambda t:abs(source['people'][t]['head_center'][0]-target[0]));other=np.array(fresh[nearest]['keypoints']);chest=.75*other[5:7].mean(0)+.25*other[11:13].mean(0);target=.75*target+.25*chest;goals.append(dict(side=side,hand_pixels=target.tolist(),elbow_pixels=kp[elbowj].tolist() if scores[elbowj]>.45 else None,source='fresh_owned_joints_plus_observed_restraining_target',target_person_index=nearest))
   notes.append('scene_relation_two_targets_and_between_geometry_confirmed')
  elif owned_hands[i]:
   interaction='reaching';posture='sitting' if a.get('posture')=='sitting' else 'standing'
   for hand in owned_hands[i]:goals.append(dict(side=None,hand_pixels=hand['center'],elbow_pixels=None,source='localized_detached_hand_and_scale_ownership',target_person_index=None))
  # Only proven gesture/basic-structure failures replace the existing automatic pose.
  candidates=[]
  if interaction!='preserve' or (scope=='full' and posture=='standing' and posture_cost(metrics[pose_id],posture,'both_grounded')>=10):
   pool=[r for r in library if posture in tags(r,'posture') and posture_cost(metrics[r['pose_id']],posture,'both_grounded')<10 and r['measures']['torso_lean_degrees']<35]
   if interaction=='restraining':
    specific=[r for r in pool if re.search(r'양손바닥.*(?:앞|막)|양손.*앞.*막|두 손.*앞.*(?:제지|막)|팔꿈치.*벌.*막',str(r['description']))]
    if not specific:specific=[r for r in pool if re.search(r'팔.*벌|팔.*뻗|손바닥.*앞|막기|가로막',str(r['description']))]
   elif interaction=='reaching':specific=[r for r in pool if re.search(r'팔.*뻗|손.*내밀|손바닥|막기|손을.*뻗',str(r['description']))]
   else:specific=pool
   pool=specific or pool
   def cost(r):
    pred=predict(r,body,anchor,cfg)
    if pred is None:return 1e6
    terms=[]
    for goal in goals:
     indices=[9,10] if goal['side'] is None else ([9] if goal['side']=='Left' else [10]);terms.append(min(np.linalg.norm(pred[j]-goal['hand_pixels'])/max(anchor[3]-anchor[1],1) for j in indices))
    if not terms:terms=[r['measures']['neutral_cost']*.1]
    return sum(terms)/len(terms)+.1*(r['measures']['torso_lean_degrees']/90)**2
   ranked=sorted(pool,key=lambda r:(cost(r),r['pose_id']));pose_id=ranked[0]['pose_id'];candidates=[dict(pose_id=r['pose_id'],cost=cost(r),description=r['description']) for r in ranked[:8]]
  chosen=lookup[pose_id];bvh=folder/f'person_{i}-base.bvh';align_report=align_bvh(lib/'bvh'/(pose_id+'.bvh'),bvh,body)
  row=dict(person_index=i,pose_id=pose_id,description=chosen['description'],bvh=str(bvh),posture=posture,output_scope=scope,direction=dict(torso_camera_yaw=body,torso_label=next((k for k,v in YAW.items() if v==body),'unknown'),head_label=head,head_camera_yaw=head_yaw,head_relative_yaw=relative,head_pitch_degrees=pitch,notes=notes),interaction=interaction,arm_goals=goals,head_anchor_bbox=anchor.tolist(),head_scale_reason=scale_reason,source_head_reference=mid(oldp['target_head_bbox']).tolist(),source_bbox=person['bbox'],baseline_pose_id=prior[i]['pose_id'],baseline_pose_retained=pose_id==prior[i]['pose_id'],structural_metrics=metrics[pose_id],alignment=align_report,notes=notes,candidates=candidates,manual_inputs=False)
  rows.append(row);print('PLAN',cut,i,'body',body,'head',head,'scope',scope,'action',interaction,'pose',pose_id,'arms',len(goals),flush=True)
 (folder/'ownership.json').write_text(json.dumps(dict(image_sha256=item['image_sha256'],hands=ownership,restraining_targets=restraining,raw_scene_description_untrusted=roles.get('scene_action')),ensure_ascii=False,indent=2));(folder/'selection.json').write_text(json.dumps(dict(image_sha256=item['image_sha256'],people=rows),ensure_ascii=False,indent=2));config['people'][cut]=rows
(ROOT/'automatic-selections.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
