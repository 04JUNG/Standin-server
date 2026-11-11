"""Use develop's HumanArt multiview consensus extractor, never legacy Body fallback."""
import argparse,sys,json,hashlib,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--checkout',type=Path,required=True);p.add_argument('--model-manifest',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cases',nargs='+',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(a.checkout.resolve()))
from experiments.refine_v5.develop_skeleton_seeds import extract
from src.pose import _load_bgr
from src.pose_humanart import HumanArtPoseModel
model=HumanArtPoseModel(str(a.model_manifest),calibration_owner='manifest-rescue');revision=subprocess.check_output(['git','-C',str(a.checkout),'rev-parse','HEAD']).decode().strip()
for cid in a.cases:
 paths=[p for p in (a.assets/'input/rough').glob(f'{cid}.*') if p.suffix.lower() in ['.png','.jpg','.jpeg']]
 if len(paths)!=1:raise ValueError(f'{cid}: ambiguous or missing rough')
 path=paths[0];image=_load_bgr(str(path));status,hypotheses,box=extract(model,image,path);rescue=False
 # Sparse drawings can defeat the person box detector while the same HumanArt
 # pose model agrees across crops. Never choose a box from multiple people.
 if status=='requires_one_detected_person':
  detected=np.asarray(model.detector(image))
  if detected.shape==(0,4):
   original_detector=model.detector;height,width=image.shape[:2]
   try:
    model.detector=lambda _:np.array([[0,0,width,height]],np.float32)
    status,hypotheses,box=extract(model,image,path);rescue=True
   finally:model.detector=original_detector
   agreed=hypotheses.get('consensus');mask=np.asarray(agreed['joint_mask'],bool) if agreed else np.zeros(17,bool)
   if mask[5:].sum()<8 or mask[[5,6,11,12]].sum()<3:status='insufficient_full_image_consensus';hypotheses.pop('consensus',None)
 diagnostic=dict(case=cid,status=status,full_image_rescue=rescue,ownership_requires_single_person_vlm=True,available_hypotheses=list(hypotheses),box=box,image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),joint_counts={n:int(np.asarray(h['joint_mask'],bool).sum()) for n,h in hypotheses.items()},model_manifest_sha256=hashlib.sha256(a.model_manifest.read_bytes()).hexdigest(),extractor_revision=revision)
 (a.out/f'{cid}-diagnostic.json').write_text(json.dumps(diagnostic,indent=2))
 if status!='ready' or 'consensus' not in hypotheses:
  print(f'{cid}: insufficient agreed detector evidence; no fabricated coordinates',flush=True);raise SystemExit(42)
 h=hypotheses['consensus'];mask=np.asarray(h['joint_mask'],bool);scores=np.where(mask,h['joint_scores'],0)
 record=dict(case_id=cid,full_image_rescue=rescue,ownership_requires_single_person_vlm=True,producer='develop HumanArt M multiview consensus',extractor_revision=revision,model_manifest_sha256=hashlib.sha256(a.model_manifest.read_bytes()).hexdigest(),image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),image_size=[image.shape[1],image.shape[0]],keypoint_format='COCO17',keypoints=np.asarray(h['keypoints']).tolist(),scores=scores.tolist(),joint_mask=mask.tolist(),available_hypotheses=list(hypotheses),box=box)
 (a.out/f'{cid}.json').write_text(json.dumps(record,indent=2));print(cid,'consensus',int(mask.sum()),flush=True)
