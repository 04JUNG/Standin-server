#!/usr/bin/env python3
"""Measure actual reimported FBX joints with one fixed search camera."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    import numpy as np
    from src.refine_projection import FixedProjection
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--joints',type=Path,required=True);p.add_argument('--pose-plan',type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=json.loads(a.joints.read_text())
    with np.load(a.pose_plan,allow_pickle=False) as z:camera=json.loads(str(z['camera_json']))
    q=np.array(json.loads(a.analysis.read_text())['analyze']['people'][0]['keypoints'])
    center=q[[11,12]].mean(0);scale=np.linalg.norm(q[[5,6]].mean(0)-center);target=(q-center)/scale
    rotation=np.array(camera['rotation'])
    def aligned(row):return (np.array(row['joints_world'])[:,[0,2,1]]*[1,1,-1])@rotation.T
    base=aligned(rows[0]);projection=FixedProjection.from_aligned_base(base,camera['distance_torso'])
    bxy=projection.project(base).astype(float);origin=bxy[[11,12]].mean(0);unit=np.linalg.norm(bxy[[5,6]].mean(0)-origin)
    result=[]
    for row in rows:
        xyz=aligned(row);xy=(projection.project(xyz)-origin)/unit
        distances=np.linalg.norm(xy-target,axis=1)
        result.append(dict(label=row['label'],artifact_sha256=row['artifact_sha256'],
            joint_nme=float(distances[5:].mean()),wrist_nme=float(distances[[9,10]].mean()),
            ankle_nme=float(distances[[15,16]].mean()),limb_nme={k:float(distances[v].mean()) for k,v in
                dict(left_arm=[7,9],right_arm=[8,10],left_leg=[13,15],right_leg=[14,16]).items()},
            camera_joints_m=xyz.tolist(),projected_pixels=(xy*scale+center).tolist(),joints_world=row['joints_world']))
    out=dict(schema='fixed-camera-actual-fbx-comparison-v1',camera=camera,
        normalization='same previous-FBX projected torso and hip center for both files',
        depth_ground_truth=False,observations='saved actual Human-Art/Gemini 124629 analysis',
        rows=result,relative_error_reduction=1-result[1]['joint_nme']/result[0]['joint_nme'])
    a.output.write_text(json.dumps(out,indent=2));print([(r['label'],r['joint_nme'],r['wrist_nme'],r['ankle_nme']) for r in result]);print(out['relative_error_reduction'])
if __name__=='__main__':main()
