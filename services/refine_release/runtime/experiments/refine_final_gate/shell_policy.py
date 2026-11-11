import cv2,numpy as np
from policy import LIMITS
from common import skin,project,write
def rasterize_silhouette(xy,faces,size=(687,535)):
 mask=np.zeros(size,np.uint8)
 # Fill triangles individually: a multi-contour fill uses even/odd holes at overlaps.
 for tri in xy[faces].astype(np.int32):cv2.fillConvexPoly(mask,tri,1)
 return mask.astype(bool)

def silhouette(c,row):
 v=skin(c.v,c.weights,row['world'],c.base);cam=row['cam'];cv=(v-cam['origin'])@cam['camera'];xy=project(cv,np.array(cam['anchor_camera_xyz']),cam['inverse_distance'],cam['pixel_scale'],cam['offset']);return rasterize_silhouette(xy,c.f)
def accept(c,seed,row):
 d=row['diagnostic'];a=silhouette(c,seed);b=silhouette(c,row);iou=float((a&b).sum()/max((a|b).sum(),1));fixed=['Hips','Spine','Spine1','Spine2','Neck','Head','LeftShoulder','RightShoulder'];fixed_ok=all(np.array_equal(seed['world'][c.ix[n]],row['world'][c.ix[n]]) for n in fixed);samecam=row['cam']==seed['cam'];issues=[x for x in d['issues'] if x['kind'] not in ['intent_landmark_shift','local_motion_budget']]
 # A hidden-hand repair must not hide or displace an observed hand target.
 before=c.xy(seed['world'],seed['cam']);after=c.xy(row['world'],row['cam'])
 for name in ['LeftHand','RightHand']:
  if name not in c.goal:continue
  # Detector keypoints are inferred coordinates, not proof of hand visibility.
  # Preserve the original rear-hand branch unless clear image evidence opposes it.
  source=(c.base[:,:3,3]-seed['cam']['origin'])@seed['cam']['camera'];side=name[:-4]
  if source[c.ix[name],2]-source[c.ix['Spine'],2]<-.04*c.scale:continue
  i=c.ix[name];goal=np.asarray(c.goal[name]);shift=float(np.linalg.norm(after[i]-before[i]))
  if shift>LIMITS.landmark_budget_px and np.linalg.norm(after[i]-goal)>np.linalg.norm(before[i]-goal)+LIMITS.landmark_error_increase_px:issues.append(dict(kind='visible_hand_target_regression',bone=name,shift_px=shift))
 # Separate silhouette-preserving repair acceptance; legacy gate stays in diagnostics.
 if iou<LIMITS.shell_iou:issues.append(dict(kind='silhouette_regression',iou=iou,minimum=LIMITS.shell_iou))
 if not fixed_ok or not samecam:issues.append(dict(kind='fixed_frame_changed'))
 if any(n not in [s+t for s in ['Left','Right'] for t in ['Arm','ForeArm','Hand']] for n in set(row['pr']['edits'])|set(seed['pr']['edits']) if row['pr']['edits'].get(n)!=seed['pr']['edits'].get(n)):issues.append(dict(kind='out_of_scope_edit'))
 result=dict(passed=not issues,issues=issues,silhouette_iou=iou,fixed_bones_unchanged=fixed_ok,camera_unchanged=samecam,legacy_gate_passed=d['passed'],legacy_gate_issues=d['issues'],manual_targets=False);write(row['path']/'shell-gate.json',result);return result

