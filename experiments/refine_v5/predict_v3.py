"""Strict label-free offline inference boundary, source only, pre-selection safe."""
from pathlib import Path
import cv2
import numpy as np
from experiments.refine_v5.learning_data import axes,frame,geometry,image_features,normalized_points,visibility,COCO,digest
from experiments.refine_v5.learned import infer
ALLOWED={'case_id','image','image_sha256','source_fbx','source_sha256','source_bones','keypoints','scores','review','seed_kind','fixed_camera','ownership_verified'}

def predict(checkpoint,row):
    if set(row)-ALLOWED:raise ValueError('unknown input fields: labels and target files are forbidden')
    if row['seed_kind'] not in ('reconstructed_seed','selected_source'):raise ValueError('unknown seed provenance')
    if digest(row['image'])!=row['image_sha256'] or digest(row['source_fbx'])!=row['source_sha256']:raise ValueError('stale input digest')
    camera=np.asarray(row['fixed_camera'],float)
    if camera.shape!=(3,3) or not np.isfinite(camera).all() or not np.allclose(camera@camera.T,np.eye(3),atol=1e-5) or np.linalg.det(camera)<.99:raise ValueError('invalid fixed camera')
    base=axes(row['source_bones']);q,m=normalized_points(row['keypoints'],row['scores']);review=row['review']
    im=cv2.imread(str(row['image']),cv2.IMREAD_GRAYSCALE)
    if im is None:raise ValueError('image unavailable')
    feature=image_features(im,np.array(row['keypoints'])[COCO]);g=geometry(base,q,m,review['representation'],camera)
    pred=infer(checkpoint,g,feature,base);partmask=visibility(review)
    if not row['ownership_verified']:partmask[:]=0
    return dict(case_id=row['case_id'],axes=pred['axes'].tolist(),uncertainty_rad=pred['uncertainty'].tolist(),observed_parts=partmask.tolist(),source_sha256=row['source_sha256'],image_sha256=row['image_sha256'],fixed_camera=camera.tolist(),ownership_verified=row['ownership_verified'],target_applied=False,version=pred['version'])
