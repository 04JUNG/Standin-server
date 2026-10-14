"""RTMW input smoke test on existing photo and synthetic image; no GT claims."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_humanart_z import read_rgb
from scripts.experiment_z_rendered import save_json,sha256
from standin_eval.z_diagnostics import infer_pose
from standin_eval.mixed_z import corrected_rtmw,model_agreement


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--human-model',type=Path,required=True)
    p.add_argument('--z-model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    from rtmlib import RTMPose
    human=RTMPose(str(a.human_model),model_input_size=(192,256),backend='onnxruntime',device='cpu')
    depth=corrected_rtmw(a.z_model)
    images={'existing_photo':ROOT/'../러프/러프 데이터/러프/rough_292.png',
            'synthetic_query_001':ROOT/'out/z-diagnosis-20261012-heldout/images/query_001.png'}
    result={}
    for label,path in images.items():
        rgb=read_rgb(path);h=infer_pose(rgb,human,crop=True,turns=(0,1,2,3),is_3d=False)
        record={'image':str(path.resolve()),'sha256':sha256(path),'human':h,
                'grayscale':bool(np.array_equal(rgb[:,:,0],rgb[:,:,1]) and np.array_equal(rgb[:,:,1],rgb[:,:,2])),
                'variants':{}}
        for mode in ('rgb_normalized','bgr_normalized','rgb_raw_negative_control'):
            depth.mean=None if mode=='rgb_raw_negative_control' else (123.675,116.28,103.53)
            depth.std=(58.395,57.12,57.375)
            image=rgb if mode!='bgr_normalized' else np.ascontiguousarray(rgb[:,:,::-1])
            z=infer_pose(image,depth,crop=True,turns=(h['turn'],),is_3d=True)
            agreement=model_agreement(h['keypoints2d'],z['keypoints2d'])
            record['variants'][mode]={'prediction':z,'agreement':agreement,
                'body_scores_at_least_03':int((np.array(z['scores'])[5:]>=.3).sum())}
            print(label,mode,'body valid',record['variants'][mode]['body_scores_at_least_03'],agreement,flush=True)
        result[label]=record
    save_json(a.out/'results.json',result)
    save_json(a.out/'protocol.json',{'purpose':'pipeline smoke test, not quantitative accuracy or replacement model selection',
        'normalization':'official RGB mean [123.675,116.28,103.53], std [58.395,57.12,57.375]',
        'crop_rotation':'same pixel crop and selected Human-Art rotation for each variant',
        'controls':'BGR and unnormalized raw are diagnostics, not adopted corrections',
        'sources':['https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose3d/configs/rtmw3d-x_8xb32_cocktail14-384x288.py',
                   'https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/pose_estimation/rtmpose3d.py'],
        'model_hashes':{'human':sha256(a.human_model),'depth':sha256(a.z_model)}})
