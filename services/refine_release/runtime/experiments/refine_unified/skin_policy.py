"""Scale invariant surface damage gate; local rotation rollback, never vertex editing."""
import numpy as np

def region(name):
 for side in ('Left','Right'):
  if name.startswith(side):
   return side+('_leg' if any(x in name for x in ('UpLeg','Leg','Foot','Toe')) else '_arm')
 return 'head' if name.startswith(('Head','Neck')) else 'torso'

class SurfaceGate:
 # Body-normalized thresholds are shared across cuts, not fitted to 004.
 def __init__(self,v,faces,weights,names,torso):
  self.v=v;self.f=faces;self.scale=max(float(torso),1e-8)
  self.edges=faces[:,[[0,1],[1,2],[2,0]]]
  self.length=np.linalg.norm(v[self.edges[:,:,0]]-v[self.edges[:,:,1]],axis=2)
  self.area=np.linalg.norm(np.cross(v[faces[:,1]]-v[faces[:,0]],v[faces[:,2]]-v[faces[:,0]]),axis=1)/2
  regions=sorted(set(map(region,names)));mass=np.stack([weights[:,[i for i,n in enumerate(names) if region(n)==r]].sum(1) for r in regions],axis=1)
  self.owner=np.array(regions)[mass[faces].mean(1).argmax(1)]
 def measure(self,vertices):
  if not np.isfinite(vertices).all():raise ValueError('Nonfinite skinned mesh')
  length=np.linalg.norm(vertices[self.edges[:,:,0]]-vertices[self.edges[:,:,1]],axis=2)
  # Absolute excess beyond allowed 0.5x..2x band, normalized by torso length.
  excess=np.maximum(np.maximum(length-2*self.length,.5*self.length-length),0).max(1)/self.scale
  out={}
  for r in sorted(set(self.owner)):
   mask=self.owner==r;area=self.area[mask].sum();bad=mask&(excess>.005)
   out[r]=dict(affected_area_fraction=float(self.area[bad].sum()/max(area,1e-12)),peak_excess_torso=float(excess[mask].max(initial=0)))
  return out
 @staticmethod
 def failures(before,after):
  return {r:d for r,d in after.items() if d['affected_area_fraction']>max(.01,before[r]['affected_area_fraction'])+1e-9 or d['peak_excess_torso']>max(.04,before[r]['peak_excess_torso'])+1e-9}

def repair(rig,mesh,prior,proposal,goal,prepare):
 from core import fk,skin
 names,ix,parents,local,base=prepare(rig);v,f,weights=mesh['vertices'],mesh['faces'],mesh['weights']
 gate=SurfaceGate(v,f,weights,names,np.linalg.norm(base[ix['Spine2'],:3,3]-base[ix['Hips'],:3,3]))
 def world(edits):return fk(local,parents,[ix[n] for n in edits],np.radians(list(edits.values()))) if edits else base.copy()
 def measure(edits):
  w=world(edits);d=gate.measure(skin(v,weights,w,base));xy=((w[:,:3,3]-proposal['origin'])@proposal['camera'])[:,:2]*[proposal['pixel_scale'],-proposal['pixel_scale']]+proposal['offset'];err=float(np.sqrt(np.mean([(xy[ix[n]]-p)**2 for n,p in goal.items()])))
  return w,d,err
 old=prior.get('edits',{});edits=dict(proposal['edits']);_,before,_=measure(old);w,initial,error=measure(edits);steps=[];fractions={r:1. for r in set(map(region,edits))}
 def severity(d):return sum(max(0,x['affected_area_fraction']-max(.01,before[r]['affected_area_fraction']))+max(0,x['peak_excess_torso']-max(.04,before[r]['peak_excess_torso'])) for r,x in d.items())
 d=initial
 for _ in range(30):
  failed=gate.failures(before,d)
  if not failed:break
  candidates=[]
  # Evaluate causal effect: parent torso changes can damage a limb patch.
  for group,frac in fractions.items():
   if frac==0:continue
   for alpha in (.75,.5,.25,0):
    newfrac=frac*alpha;e=dict(edits)
    for n in proposal['edits']:
     if region(n)==group:e[n]=(np.array(old.get(n,[0,0,0]))+newfrac*(np.array(proposal['edits'][n])-old.get(n,[0,0,0]))).tolist()
    nw,nd,ne=measure(e);score=severity(nd)
    if score<severity(d)-1e-10:candidates.append((score,ne,group,newfrac,e,nw,nd))
  if not candidates:break
  score,error,group,frac,edits,w,d=min(candidates,key=lambda x:(x[0],x[1]));fractions[group]=frac;steps.append(dict(region=group,retained_fraction=frac,rmse=error,remaining_failures=list(gate.failures(before,d))))
 # Recover the largest safe fraction after coarse local repair. This avoids
 # unnecessarily keeping a 50% step when the safe boundary lies near 100%.
 for group in sorted(fractions):
  if fractions[group]>=1:continue
  lo,hi=fractions[group],1.
  for _ in range(8):
   mid=(lo+hi)/2;e=dict(edits)
   for n in proposal['edits']:
    if region(n)==group:e[n]=(np.array(old.get(n,[0,0,0]))+mid*(np.array(proposal['edits'][n])-old.get(n,[0,0,0]))).tolist()
   nw,nd,ne=measure(e)
   if not gate.failures(before,nd):lo=mid;edits,w,d,error=e,nw,nd,ne
   else:hi=mid
  fractions[group]=lo
 return edits,w,dict(policy='surface-size-area-v2-preserve',thresholds=dict(excess_torso=.005,regional_area_fraction=.01,peak_excess_torso=.04),before=before,proposal=initial,selected=d,steps=steps,retained_fractions=fractions,accepted=not gate.failures(before,d),rmse=error)
