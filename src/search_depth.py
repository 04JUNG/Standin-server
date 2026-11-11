"""Optional independent RTMW3D evidence for fully observed single people."""
from hashlib import sha256
from pathlib import Path
from threading import Lock

import numpy as np

from .pose import _load_bgr
from .search_computation import cached_search_work


class SearchDepthObserver:
    def __init__(self, path):
        from rtmlib import RTMPose3d

        class DepthCorrectedRTMW(RTMPose3d):
            def postprocess(self, outputs, center, scale, simcc_split_ratio=2.):
                keypoints, score, simcc, xy = super().postprocess(outputs, center, scale, simcc_split_ratio)
                # The Z heatmap length differs from the image input height.
                half_depth = outputs[2].shape[-1] / simcc_split_ratio / 2
                keypoints[..., 2] = (simcc[..., 2] / half_depth - 1.) * self.z_range
                return keypoints, score, simcc, xy

        model = Path(path).resolve(strict=True)
        if not model.is_file():
            raise ValueError('Search depth model must be a local file')
        self.model_path = model
        self.model = DepthCorrectedRTMW(str(model), model_input_size=(288, 384),
                                       backend='onnxruntime', device='cpu')
        self.lock = Lock()

    def observe(self, image, box, evidence):
        points = np.asarray(evidence['keypoints'], dtype=np.float32)
        mask = np.asarray(evidence['joint_mask'], dtype=bool)
        if points.shape != (17, 2) or mask.shape != (17,) or not mask[[5, 6, 11, 12]].all():
            return None
        torso = np.linalg.norm((points[5] + points[6] - points[11] - points[12]) * .5)
        if not np.isfinite(torso) or torso < 8:
            return None
        prediction = self.predict(image, box)
        if prediction is None:
            return None
        projected, scores, z, _ = prediction
        if not np.isfinite(z).all() or not np.isfinite(projected).all():
            return None
        reliable = mask & (np.linalg.norm(points - projected, axis=-1) <= .45 * torso) & (scores >= .3)
        if not reliable[[11, 12]].all() or reliable[5:].sum() < 6:
            return None
        return (z - .5 * (z[11] + z[12])) / .5, reliable

    def predict(self, image, box):
        if box is None:
            return None
        image = _load_bgr(image)
        height, width = image.shape[:2]
        box = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
        if not np.isfinite(box).all() or min(box[2] - box[0], box[3] - box[1]) < 32:
            return None
        # The API and Recall decode the image separately. Share predictions only
        # when the decoded pixels and the model's clipped box are identical.
        key = (self.model_path, image.shape, image.dtype.str, sha256(image).digest(),
               box.dtype.str, box.tobytes())
        return cached_search_work('depth_prediction', key, lambda: self._predict(image, box), limit=16)

    def _predict(self, image, box):
        rgb = np.ascontiguousarray(image[..., ::-1])
        with self.lock:
            joints, scores, _, xy = self.model(rgb, bboxes=[box])
        if len(xy) != 1:
            return None
        return xy[0, :17], scores[0, :17], joints[0, :17, 2], box
