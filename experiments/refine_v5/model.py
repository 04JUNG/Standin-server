"""Small supervised ridge residual baseline. Inference has no GT file access.
Pixel features are a sanity baseline, not a pretrained semantic image encoder.
"""
import numpy as np
from PIL import Image
ACTIVE = ['mixamorig:LeftArm', 'mixamorig:LeftForeArm', 'mixamorig:RightArm', 'mixamorig:RightForeArm']
VERSION = 'v5-p1-pixel-bone-ridge-v1'

def feature(image, state):
    with Image.open(image) as im:
        pixels = np.asarray(im.convert('L').resize((24, 24)), dtype=float).reshape(-1) / 255
    pixels -= pixels.mean()
    pixels /= max(np.linalg.norm(pixels), 1e-08)
    base = np.array([state['arms'][n]['direction'] for n in ACTIVE], float)
    if not np.isfinite(base).all() or not np.allclose(np.linalg.norm(base, axis=1), 1, atol=0.0001):
        raise ValueError('invalid base directions')
    return (np.r_[1.0, pixels, base.ravel()], base)

class Corrector:

    def __init__(self, weights):
        self.weights = np.asarray(weights, float)

    @classmethod
    def fit(cls, x, y, regularization=1e-06):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if len(x) == 0 or not np.isfinite(x).all() or (not np.isfinite(y).all()):
            raise ValueError('invalid training arrays')
        if regularization <= 0:
            raise ValueError('positive regularization required')
        return cls(x.T @ np.linalg.solve(x @ x.T + regularization * np.eye(len(x)), y))

    def predict(self, image, state):
        x, base = feature(image, state)
        out = base + (x @ self.weights).reshape(4, 3)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        if not np.isfinite(out).all() or (norms < 1e-08).any():
            raise ValueError('invalid prediction')
        return out / norms

    def save(self, path):
        np.savez(path, weights=self.weights, version=VERSION)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            if str(z['version']) != VERSION:
                raise ValueError('checkpoint version mismatch')
            w = z['weights']
        if w.shape != (589, 12) or not np.isfinite(w).all():
            raise ValueError('invalid weights')
        return cls(w)

def angles(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(axis=-1), -1, 1)))
