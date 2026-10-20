"""Geometry checks and nearest-existing-pose diagnostics (not artistic approval)."""
from __future__ import annotations

import sqlite3
from pathlib import Path
import numpy as np

from src.bvh import load_coco17
from .motion import normalized_body
from .storage import contained_path


def existing_bvh_path(data_dir: Path, raw: str) -> Path:
    relative = raw.replace("\\", "/")
    if "data/" in relative:
        relative = relative.split("data/", 1)[1]
    return contained_path(data_dir, relative)


def existing_poses(data_dir: Path) -> list[dict]:
    with sqlite3.connect((data_dir / "poses.db").resolve().as_uri() + "?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        return [dict(row) for row in con.execute("SELECT * FROM poses ORDER BY pose_id")]


class ReferenceIndex:
    def __init__(self, ids: list[str], bodies: np.ndarray):
        self.ids = ids
        self.bodies = bodies

    @classmethod
    def load(cls, data_dir: Path) -> "ReferenceIndex":
        ids, bodies = [], []
        for pose in existing_poses(data_dir):
            path = existing_bvh_path(data_dir, pose["bvh_path"])
            kp, scores = load_coco17(str(path))
            if not np.all(scores[5:] > 0):
                raise ValueError(f"existing pose has missing body joints: {pose['pose_id']}")
            ids.append(pose["pose_id"])
            bodies.append(normalized_body(kp))
        if not ids:
            raise ValueError("existing pose database is empty")
        return cls(ids, np.stack(bodies))

    def nearest(self, keypoints: np.ndarray) -> dict:
        body = normalized_body(keypoints)
        distance = np.sqrt(np.mean(np.sum((self.bodies - body) ** 2, axis=-1), axis=-1))
        index = int(np.argmin(distance))
        return {"pose_id": self.ids[index], "distance": round(float(distance[index]), 6)}
