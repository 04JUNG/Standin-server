"""Local relation helpers; no case IDs or hidden reference coordinates."""
import numpy as np

def interval(zone):
 return {'proximal':(0.,.4),'middle':(.25,.75),'distal':(.6,1.)}.get(zone,(0.,1.))

def local_points(front,back,front_zone='uncertain',back_zone='uncertain'):
 """Bounded projected closest points on the named local segments."""
 u=front[1]-front[0];v=back[1]-back[0];lo=np.array([interval(front_zone)[0],interval(back_zone)[0]]);hi=np.array([interval(front_zone)[1],interval(back_zone)[1]])
 st=np.clip(np.linalg.lstsq(np.column_stack([u[:2],-v[:2]]),(back[0]-front[0])[:2],rcond=None)[0],lo,hi)
 for _ in range(8):
  st[0]=np.clip(np.dot((back[0]+st[1]*v-front[0])[:2],u[:2])/max(np.dot(u[:2],u[:2]),1e-12),lo[0],hi[0]);st[1]=np.clip(np.dot((front[0]+st[0]*u-back[0])[:2],v[:2])/max(np.dot(v[:2],v[:2]),1e-12),lo[1],hi[1])
 return front[0]+st[0]*u,back[0]+st[1]*v,st

def partial_visibility_ok(before,after):
 if not after['projected_pixels']:return False
 # Never turn partial into fully hidden/fully visible, or worsen existing occlusion.
 a=after['visible_fraction'];b=before['visible_fraction']
 return .03<a<.97 and (not before['projected_pixels'] or a<=b+.02)

def screen_supported_relations(conditions,goal):
 """Reject a categorical junction that contradicts four observed endpoints."""
 import copy
 c=copy.deepcopy(conditions);kept=[];rejected=[];segments={'thigh':('UpLeg','Leg'),'shin':('Leg','Foot')}
 if ('near' in c or 'far' in c) and (c.get('near') not in ('Left','Right') or c.get('far') not in ('Left','Right') or c.get('near')==c.get('far')):
  c['relative_order_enabled']=False
 for r in c.get('local_occlusions') or []:
  if not isinstance(r,dict) or r.get('near') not in ('Left','Right') or r.get('far') not in ('Left','Right') or r.get('near')==r.get('far') or r.get('front_segment') not in ('thigh','shin','foot') or r.get('back_segment') not in ('thigh','shin','foot'):
   rejected.append(dict(relation=r,reason='uncertain_or_invalid_relation'));continue
  endpoints=[r.get('near','')+n for n in segments.get(r.get('front_segment'),())]+[r.get('far','')+n for n in segments.get(r.get('back_segment'),())]
  if len(endpoints)==4 and all(n in goal and n not in c.get('unobserved_joints',[]) for n in endpoints):
   a=np.array([np.r_[goal[n],0] for n in endpoints[:2]]);b=np.array([np.r_[goal[n],0] for n in endpoints[2:]]);p,q,_=local_points(a,b,r.get('along_front'),r.get('along_back'))
   if np.linalg.norm(p-q)>25:
    rejected.append(dict(relation=r,reason='observed_segments_do_not_overlap',gap_px=float(np.linalg.norm(p-q))));continue
  kept.append(r)
 c['local_occlusions']=kept
 if rejected and not kept:c['relative_order_enabled']=False
 return c,rejected
