"""Camera-only confidence routing. Never rewrites body, head or limb observations."""
import copy

def route_camera(document):
 result=copy.deepcopy(document);raw=result['raw'];before=copy.deepcopy(raw)
 if raw.get('confidence')=='weak':
  raw['nearer_shoulder']='uncertain'
  raw['elevation']='uncertain'
  if raw.get('torso_view')=='profile':raw['torso_view']='uncertain'
  reason='weak_view_preserve_orientation_fit_framing'
 else:reason='clear_view_existing_solver'
 result['camera_evidence_policy']={'reason':reason,'observed_camera':before,'head_and_limbs_modified':False,'case_specific_rules':False}
 return result
