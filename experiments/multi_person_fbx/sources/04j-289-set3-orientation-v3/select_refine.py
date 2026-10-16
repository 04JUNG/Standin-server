"""Fresh per-actor evidence -> structural library gate -> conservative limb refine."""
from pathlib import Path
import os,sys,json,time,dataclasses,hashlib,re
import numpy as np
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1];BASE=ROOT.parent/'04j-289-set34-refine-scenes';LAYOUT=ROOT.parent/'04j-289-set3-layout-v2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE))
from evidence_policy import direction_policy,scope_policy,measures,posture_cost
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)) and any(s in str(args[0]) for s in ('user-feedback-eval.json','evaluation-only','/gt/','visual-review.json')):raise RuntimeError('Evaluation labels blocked')
sys.addaudithook(guard)
sys.path.insert(0,str(WORK/'.worktrees/standin-multi-person-fbx'))
import src
src.__path__.insert(0,str(WORK/'.worktrees/standin-fullbody-routing/src'))
os.environ['REFINE_POSE_QUARANTINE_PATH']=str(WORK/'.worktrees/standin-fullbody-routing/config/refine_pose_quarantine.v1.json')
from src.semantic_adapter_285 import build_semantic285_searcher,excluded_pose_ids
from src.semantic_search import SemanticQuery
from src.bvh import parse_bvh,load_coco17
from src.refine import refine_bvh,LIMBS
from src.config import CFG
from src.features import normalize_skeleton,bone_dirs
from src.library import pose_to_feature
from align_refine import alignment,align_bvh
receipt=json.loads((WORK/'tmp/standin-local-routing/latest-bound-library.json').read_text());lib=Path(receipt['path']);excluded=excluded_pose_ids()
library=[r for r in json.loads((BASE/'contact-index.json').read_text())['poses'] if r['pose_id'] not in excluded];lookup={r['pose_id']:r for r in library};metrics={r['pose_id']:measures(r['points']) for r in library}
service=build_semantic285_searcher(tags_path=lib/'search-v2.json',taxonomy_path=lib/'search-taxonomy-v2.json',db_path=lib/'poses.db',library_pose_ids=set(lookup),body_index_path=WORK/'outputs/04j-289-fresh-multicut-experiment/library-body-v2.json',shape_index_path=lib/'shapes.json')
cfg=dataclasses.replace(CFG,refine_limbs='all',refine_v2_enabled=True)
def fit(row,person,yaw):
 scores=np.array(person['scores']);td,tv=bone_dirs(normalize_skeleton(np.array(person['keypoints']),scores));points=np.array(row['points']);points=points@alignment(points,yaw).T;pd,pv=bone_dirs(pose_to_feature(points,'front',scores));valid=tv&pv
 return float(np.mean((1-np.sum(td[valid]*pd[valid],axis=1))/2)) if valid.sum()>=2 else .5
def tag(r,field):return {t['value'] for g in (r.get('tags') or {}).values() for t in g if t['field']==field}
config=dict(character=str(WORK/'pose-head-contact-repair-20261015/mesh-review/assets/standin-master-v2.fbx'),people={},manual_inputs=False,library=receipt['validation'])
selected=set(sys.argv[1:])
for item in json.loads((ROOT/'inputs.json').read_text()):
 cut=item['cut_id'];folder=ROOT/cut;source=json.loads((BASE/cut/'people.json').read_text());rows=[]
 if selected and cut not in selected:continue
 baseline={p['person_index']:p for p in json.loads((LAYOUT/cut/'geometry.json').read_text())['people']}
 for person in source['people']:
  i=person['person_index'];dr=json.loads((folder/f'person_{i}-direction.json').read_text());st=json.loads((folder/f'person_{i}-structure.json').read_text());ar=json.loads((folder/f'person_{i}-anatomy.json').read_text());assert dr['image_sha256']==st['image_sha256']==ar['image_sha256']==item['image_sha256'];a=dict(st['analysis'] or {});d=dict(dr['analysis'] or {});inventory=ar['analysis'];lowest=inventory.get('lowest_anatomy','');legs=inventory.get('leg_evidence','none').lower().strip();clearlegs=legs not in ('none','unknown','') and any(x in lowest for x in ('thighs','knees','shins','feet'))
  if clearlegs:
   a['visible_regions']=sorted(set(a.get('visible_regions') or [])|{'thighs'});a['body_scope']='full';a['scope_evidence']=inventory.get('lower_half_description');a['posture']=inventory.get('posture',a.get('posture'))
  elif a.get('body_scope')=='full' and lowest in ('head','chest','waist','hips') and legs=='none':
   a['visible_regions']=[r for r in a.get('visible_regions',[]) if r not in ('thighs','knees','shins','feet')];a['body_scope']='half';a['scope_evidence']=inventory.get('lower_half_description')
  if inventory.get('head_heading') not in ('unknown',None) and any(c in str(inventory.get('head_evidence','')).lower() for c in ('nose','eyes','back of head')):d['head_heading']=inventory['head_heading'];d['head_evidence']=inventory['head_evidence']
  direction=direction_policy(d);scope,notes=scope_policy(a);posture=a.get('posture','unknown');rawposture=posture
  if posture not in ('standing','sitting','kneeling','crouching','lying','bent_over_surface'):posture='standing';notes.append('unknown_support_neutral_standing_reference')
  regions=set(a.get('visible_regions') or []);mask=np.array(person['valid_mask'],bool)
  if not regions&{'upper_arms','forearms','hands'}:mask[7:11]=False
  if not regions&{'thighs','knees','shins','feet'}:mask[11:]=False
  if not regions&{'knees','shins','feet'}:mask[13:]=False
  if not regions&{'feet'}:mask[15:]=False
  owned={**person,'scores':(np.array(person['scores'])*mask).tolist(),'valid_mask':mask.tolist()};yaw=direction['torso_camera_yaw'];query=a.get('search_query_ko') or '똑바로 서서 양팔을 자연스럽게 내린 자세';gatepost='sitting' if posture=='bent_over_surface' else posture
  pool=[r for r in library if gatepost in tag(r,'posture')]
  if not pool:pool=library;notes.append('no_library_posture_tag')
  costs={r['pose_id']:posture_cost(metrics[r['pose_id']],posture,a.get('leg_action')) for r in pool};validpool=[r for r in pool if costs[r['pose_id']]<10]
  if not validpool:validpool=sorted(pool,key=lambda r:costs[r['pose_id']])[:64];notes.append('no_candidate_passes_structural_gate')
  if a.get('torso_lean')=='upright':
   upright=[r for r in validpool if r['measures']['torso_lean_degrees']<=25]
   if upright:validpool=upright;notes.append('upright_torso_geometry_gate')
  ids={r['pose_id'] for r in validpool};hits=service.search(SemanticQuery(text=query,postures=[gatepost],body_details=query),top_k=256).hits;hits=[h for h in hits if h.pose_id in ids]
  if hits:band=[h for h in hits if h.score>=hits[0].score-.06];candidates=[lookup[h.pose_id] for h in band];semantic={h.pose_id:float(h.score) for h in band}
  else:candidates=validpool;semantic={};notes.append('semantic_empty_after_structural_gate')
  action=str(a.get('arm_action','')).lower();support=a.get('head_support');actionpool=[]
  if support=='both_palms':actionpool=[r for r in validpool if re.search(r'두 볼|양손으로 얼굴을 감싸',str(r['description']))]
  elif 'crossed' in action or '교차' in query:actionpool=[r for r in validpool if re.search(r'팔짱|팔을 가슴 앞에 교차|두 팔을 가슴 앞에 교차',str(r['description']))]
  if actionpool:candidates=actionpool;notes.append('visible_arm_action_semantic_gate')
  unobserved_action=not mask[7:11].any() or any(x in action for x in ('not visible','not clearly','obscured'))
  if unobserved_action:
   candidates=validpool;notes.append('unobserved_arms_use_neutral_posture_reference');mask[7:11]=False;owned['scores']=(np.array(person['scores'])*mask).tolist()
  fits={r['pose_id']:fit(r,owned,yaw) for r in candidates};support=a.get('head_support')
  def cost(r):
   m=metrics[r['pose_id']];c=fits[r['pose_id']]+.10*costs.get(r['pose_id'],0)
   if support in ('one_palm','both_palms'):c+=.30*max(0,m['head_hand_torso']-.35)
   if a.get('torso_lean')=='upright':c+=.35*(r['measures']['torso_lean_degrees']/90)**2
   if unobserved_action:c=.15*costs.get(r['pose_id'],0)+.40*r['measures']['neutral_cost']+.35*(r['measures']['torso_lean_degrees']/90)**2
   return c
  candidates=sorted(candidates,key=lambda r:(cost(r),r['pose_id']));chosen=candidates[0];baseline_id=baseline[i]['pose_id'];baseline_retained=False
  if baseline_id in lookup:
   reference=lookup[baseline_id];reference_metrics=metrics[baseline_id];visible_lower=scope=='full' and bool(regions&{'thighs','knees','shins','feet'});reference_basic_ok=not visible_lower or posture_cost(reference_metrics,posture,a.get('leg_action'))<10;reference_upright_ok=a.get('torso_lean')!='upright' or reference['measures']['torso_lean_degrees']<=25
   if reference_basic_ok and reference_upright_ok:
    chosen=reference;baseline_retained=True;notes.append('existing_automatic_pose_retained_after_fresh_basic_structure_gate')
   else:notes.append('existing_automatic_pose_failed_fresh_basic_structure_gate')
  base=folder/f'person_{i}-base.bvh';alignment_report=align_bvh(lib/'bvh'/(chosen['pose_id']+'.bvh'),base,yaw);allowed=[]
  for limb,(_,bones) in LIMBS.items():
   js=sorted({j for pair in bones for j in pair})
   if mask[js].all() and (limb.endswith('arm') or regions&{'knees','shins','feet'}):allowed.append(limb)
  if allowed and person.get('detector_matched'):
   result=refine_bvh(str(base),person['keypoints'],owned['scores'],view='front',out_path=str(folder/f'person_{i}-refined.bvh'),allowed_limbs=allowed,lower_body_observed=bool(regions&{'knees','shins','feet'}),deadline=time.monotonic()+8,refine_mode='conservative',cfg=cfg);refinement=result.to_dict();refinement.pop('bvh_text',None);bvh=Path(result.bvh_path or base)
  else:refinement={'refined':False,'reason':'insufficient_owned_visible_limbs'};bvh=base
  refined_points,_=load_coco17(str(bvh));refined_metrics=measures(refined_points)
  if scope=='full' and posture_cost(refined_metrics,posture,a.get('leg_action'))>=10 and posture_cost(metrics[chosen['pose_id']],posture,a.get('leg_action'))<10:
   refinement={**refinement,'refined':False,'reason':'reverted_structural_posture_gate','rejected_refine_path':str(bvh)};bvh=base
  ja,da=parse_bvh(str(base));jb,db=parse_bvh(str(bvh));assert len(ja)==len(jb) and all(np.array_equal(x[2],y[2]) for x,y in zip(ja,jb));assert np.allclose(da[0,:len(ja[0][3])],db[0,:len(jb[0][3])],atol=1e-5)
  row=dict(person_index=i,bbox=person['bbox'],pose_id=chosen['pose_id'],description=chosen['description'],bvh=str(bvh),base_bvh=str(base),posture=posture,raw_posture=rawposture,output_scope=scope,structure=a,direction=direction,refinement=refinement,alignment=alignment_report,structural_metrics=metrics[chosen['pose_id']],structural_cost=posture_cost(metrics[chosen['pose_id']],posture,a.get('leg_action')),structural_candidates=len(validpool),query=query,notes=notes,baseline_pose_id=baseline_id,baseline_pose_retained=baseline_retained,baseline_receipt_scope='pose ID only; previous heading, scope, query and semantic analysis are not inference inputs',manual_inputs=False,candidates=[dict(pose_id=r['pose_id'],cost=cost(r),geometry_loss=fits[r['pose_id']],semantic_score=semantic.get(r['pose_id']),description=r['description']) for r in candidates[:8]])
  rows.append(row);print('SELECT',cut,i,chosen['pose_id'],posture,scope,direction['torso_label'],direction['head_label'],refinement['reason'],flush=True)
 config['people'][cut]=rows;(folder/'selection.json').write_text(json.dumps(dict(image_sha256=item['image_sha256'],people=rows),ensure_ascii=False,indent=2));(ROOT/'automatic-selections.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
