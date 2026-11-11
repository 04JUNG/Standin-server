"""No case labels: spatial defect extent and unresolved candidate disagreement."""
import numpy as np
from policy import LIMITS

def largest_component_area(faces,mask,area):
 ids=np.flatnonzero(mask);parent={int(i):int(i) for i in ids};edges={}
 def root(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 for i in ids:
  for a,b in [(faces[i,0],faces[i,1]),(faces[i,1],faces[i,2]),(faces[i,2],faces[i,0])]:
   key=tuple(sorted((int(a),int(b))))
   if key in edges:parent[root(int(i))]=root(edges[key])
   else:edges[key]=int(i)
 sums={}
 for i in ids:
  k=root(int(i));sums[k]=sums.get(k,0.)+float(area[i])
 return max(sums.values(),default=0.)

def contact_issue(pair,count,collision,scale):
 m=collision.get('extent',{}).get(pair)
 # Missing/one-sided distance evidence cannot certify shallow contact.
 if not m or not m['complete'] or m['inward_samples']==0:return dict(kind='mesh_intersection',pair=pair,count=count,reason='insufficient_depth_extent_evidence')
 if m['max_depth']/scale<=LIMITS.contact_depth_torso and m['max_area_fraction']<=LIMITS.contact_area_fraction and (m['max_depth']/scale)*m['max_area_fraction']<=LIMITS.contact_severity:return None
 return dict(kind='mesh_intersection',pair=pair,count=count,depth_torso=m['max_depth']/scale,area_fraction=m['max_area_fraction'])

def unresolved_hand_disagreement(c):
 if len(c.pool)<2:return []
 a,b=c.pool[:2];ca=np.array(a['cam']['camera']);cb=np.array(b['cam']['camera'])
 out=[]
 for side in ['Left','Right']:
  hand=side+'Hand';head='Head'
  if any({r['near'],r['far']}=={hand,head} for r in c.relations):continue
  i,j=c.ix[hand],c.ix[head]
  va=(a['world'][i,:3,3]-a['world'][j,:3,3])@ca/c.scale;vb=(b['world'][i,:3,3]-b['world'][j,:3,3])@cb/c.scale
  if min(np.linalg.norm(va[:2]),np.linalg.norm(vb[:2]))<.7 and abs(va[2]-vb[2])>.12 and va[2]*vb[2]<0:
   out.append(dict(kind='unresolved_hand_head_candidate_disagreement',region=side+'_arm',depths=[float(va[2]),float(vb[2])]))
 return out
