"""Offline ablation: immutable Human-Art XY plus separately decoded RTMW Z.

No query answer or gallery ID is accepted here. Anatomical COCO17 labels are
kept as-is; model disagreement is reported rather than secretly swapping XY.
"""
import numpy as np
from src.features import normalize_skeleton


def decode_depth(simcc_z_positions, z_bins, split_ratio=2., z_range=2.1744869):
    """Positions already divided by split_ratio; depth dimension != image H.

    Matches OpenMMLab SimCC3DLabel.decode. This ONNX has 576 Z bins, hence
    depth input size 288, while its 2D image height is 384.
    """
    if z_bins<=0 or split_ratio<=0:
        raise ValueError('invalid depth dimension')
    return (np.asarray(simcc_z_positions)/(z_bins/split_ratio/2)-1)*z_range


def corrected_rtmw(model_path):
    from rtmlib import RTMPose3d

    class DepthCorrectedRTMW(RTMPose3d):
        def postprocess(self,outputs,center,scale,simcc_split_ratio=2.):
            kp,score,simcc,k2=super().postprocess(outputs,center,scale,simcc_split_ratio)
            kp[...,2]=decode_depth(simcc[...,2],outputs[2].shape[-1],
                                   simcc_split_ratio,self.z_range)
            return kp,score,simcc,k2

    return DepthCorrectedRTMW(str(model_path),model_input_size=(288,384),
                               backend='onnxruntime',device='cpu')


def compose_query(human_xy,human_scores,rtmw_z,threshold=.3,torso_meters=.5):
    if torso_meters<=0:
        raise ValueError('torso_meters must be positive')
    points=np.asarray(human_xy,dtype=np.float32)
    scores=np.asarray(human_scores,dtype=np.float32)
    if points.shape!=(17,2) or scores.shape!=(17,) or not np.isfinite(points).all() or not np.isfinite(scores).all():
        raise ValueError('invalid Human-Art output')
    mask=scores>=threshold
    xy=normalize_skeleton(points,scores,kpt_thr=threshold,valid_mask=mask).reshape(17,2)
    z=np.asarray(rtmw_z,dtype=np.float32)
    if z.shape!=(17,) or not np.isfinite(z).all():
        raise ValueError('invalid RTMW depth')
    z=(z-(z[11]+z[12])/2)/torso_meters
    z[~mask]=0
    return xy,z,mask


def model_agreement(human_xy,rtmw_xy):
    """Diagnostic only: low error is NOT ground-truth depth confidence."""
    h=np.asarray(human_xy); r=np.asarray(rtmw_xy)
    torso=max(float(np.linalg.norm((h[5]+h[6]-h[11]-h[12])/2)),1e-6)
    swap=np.arange(17)
    swap[5:]=np.arange(5,17).reshape(6,2)[:,::-1].reshape(-1)
    direct=float(np.linalg.norm(h[5:]-r[5:],axis=1).mean()/torso)
    reversed_error=float(np.linalg.norm(h[5:]-r[swap][5:],axis=1).mean()/torso)
    return {'mean_xy_disagreement_torso':direct,'swapped_xy_disagreement_torso':reversed_error,
            'swap_would_reduce_error':reversed_error<direct,
            'labels_actually_swapped':False}
