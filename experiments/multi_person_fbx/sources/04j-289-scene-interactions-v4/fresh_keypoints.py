from pathlib import Path
import sys,json,os,hashlib
import numpy as np,cv2
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1];BASE=ROOT.parent/'04j-289-set34-refine-scenes'
manifest=WORK/'.worktrees/standin-dynamic-routing/tmp/humanart-local-bundle/manifest.json'
for k,v in json.loads(manifest.read_text())['calibration'].items():
 if k in ('skeleton_kpt_threshold','min_skeleton_score','distance_metric','fallback_pos_full','fallback_pos_reduced','fallback_angle_full','fallback_angle_reduced','fallback_hybrid_full','fallback_hybrid_reduced'):os.environ[k.upper()]=str(v)
os.environ.update(POSE_MODEL_MANIFEST=str(manifest),POSE_MODEL_VARIANT='humanart-m',POSE_CANARY_STAGE='canary-100',POSE_STRICT='1');sys.path.insert(0,str(WORK/'.worktrees/standin-fullbody-routing'))
from src.pose_humanart import HumanArtPoseModel
model=HumanArtPoseModel(str(manifest))
for item in json.loads((ROOT/'inputs.json').read_text()):
 folder=ROOT/item['cut_id'];im=cv2.imdecode(np.fromfile(item['image_path'],np.uint8),cv2.IMREAD_COLOR);h,w=im.shape[:2];vp=np.array(json.loads((folder/'identities.json').read_text())['analysis_viewport_bbox']);source=json.loads((BASE/item['cut_id']/'people.json').read_text());rows=[]
 for p in source['people']:
  i=p['person_index'];a=json.loads((folder/f'person_{i}-analysis.json').read_text())['analysis'];boxes=[np.array(p['bbox'],float)]
  if a and isinstance(a.get('body_bbox'),list) and len(a['body_bbox'])==4:
   b=np.array(a['body_bbox'])*np.tile(vp[2:]-vp[:2],2)/1000+np.tile(vp[:2],2);boxes.append(b)
  candidates=[]
  for b in boxes:
   pose=model.estimate_with_detections(im,np.array([b],np.float32),w,h)[0];kp=pose.keypoints;sc=pose.scores;head=np.mean(kp[:5][sc[:5]>.35],0) if (sc[:5]>.35).any() else kp[0];dist=float(np.linalg.norm(head-np.array(p['head_center'])));candidates.append(dict(bbox=b.tolist(),keypoints=kp.tolist(),scores=sc.tolist(),head_distance_pixels=dist))
  chosen=min(candidates,key=lambda x:x['head_distance_pixels']);chosen['person_index']=i;chosen['owned_head_match']=bool(chosen['head_distance_pixels']<.20*np.linalg.norm(np.array(p['bbox'][2:])-p['bbox'][:2]));rows.append(chosen);print(item['cut_id'],i,'headerr',round(chosen['head_distance_pixels'],2),'match',chosen['owned_head_match'],flush=True)
 (folder/'fresh-keypoints.json').write_text(json.dumps(dict(image_sha256=item['image_sha256'],model=model.runtime_identity(),people=rows),indent=2))
