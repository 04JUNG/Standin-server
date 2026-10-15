"""Rig helpers: global-frame posing on the licensed Quaternius source rig.

All rest local rotations of this rig are identity (T-pose, Y-up, +Z forward,
actor-left = +X), so a joint's world rotation directly maps rest vectors.
Units are centimetres.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

import sys
from src.bvh import parse_bvh, channel_starts, write_single_frame_bvh  # noqa: E402

import os
RIG = Path(os.environ.get("POSE_GENERATION_RIG", "data/bvh/webtoon_combat_03.bvh"))
RIG_SHA256 = "38eb05b7432893fefd4cf31cb6f5fb07e21009946541d5a9d48d19c949991ee1"
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")


class Skeleton:
    def __init__(self, source: Path = RIG):
        self.source = Path(source)
        self.joints, frames = parse_bvh(str(self.source))
        self.captured = frames[0].copy()
        self.starts = channel_starts(self.joints)
        self.names = {j[0]: i for i, j in enumerate(self.joints)}
        self.children = {i: [] for i in range(len(self.joints))}
        for i, j in enumerate(self.joints):
            if j[1] >= 0:
                self.children[j[1]].append(i)
        self.reset()

    # ---- state -------------------------------------------------------
    def reset(self):
        self.frame = np.zeros_like(self.captured)
        self.local = {i: np.eye(3) for i, j in enumerate(self.joints) if not j[4]}
        self.root_pos = np.array(self.joints[0][2], dtype=float)

    def idx(self, name):
        return self.names[name]

    def offset(self, name):
        return np.array(self.joints[self.idx(name)][2], dtype=float)

    def child_offset(self, name):
        """Rest bone vector from joint to its first non-finger child."""
        i = self.idx(name)
        kids = [k for k in self.children[i]
                if not any(f in self.joints[k][0] for f in FINGERS)]
        return np.array(self.joints[kids[0]][2], dtype=float)

    def world(self):
        """World rotations and positions for every joint (incl. End Sites)."""
        rot, pos = {}, {}
        for i, j in enumerate(self.joints):
            parent = j[1]
            if parent < 0:
                rot[i] = self.local[i]
                pos[i] = self.root_pos.copy()
                continue
            pos[i] = pos[parent] + rot[parent] @ j[2]
            rot[i] = rot[parent] @ self.local[i] if not j[4] else rot[parent]
        return rot, pos

    def world_rot(self, name):
        i = self.idx(name)
        chain = []
        while i >= 0:
            chain.append(i)
            i = self.joints[i][1]
        m = np.eye(3)
        for k in reversed(chain):
            m = m @ self.local[k]
        return m

    def set_global(self, name, matrix):
        i = self.idx(name)
        parent = self.joints[i][1]
        pw = self.world_rot(self.joints[parent][0]) if parent >= 0 else np.eye(3)
        self.local[i] = pw.T @ matrix

    def set_local(self, name, matrix):
        self.local[self.idx(name)] = matrix

    def position(self, name):
        _, pos = self.world()
        return pos[self.idx(name)]

    # ---- serialisation ------------------------------------------------
    def to_frame(self):
        frame = np.zeros_like(self.captured)
        for i, j in enumerate(self.joints):
            if j[4]:
                continue
            channels = j[3]
            start = self.starts[i]
            axes = "".join(c[0] for c in channels if c.endswith("rotation"))
            angles = iter(R.from_matrix(self.local[i]).as_euler(axes, degrees=True))
            for n, c in enumerate(channels):
                if c.endswith("rotation"):
                    frame[start + n] = next(angles)
                elif c.endswith("position"):
                    k = "XYZ".index(c[0])
                    frame[start + n] = self.root_pos[k] - j[2][k]
        return frame

    def load_frame(self, frame):
        self.reset()
        for i, j in enumerate(self.joints):
            if j[4]:
                continue
            channels, start = j[3], self.starts[i]
            axes = "".join(c[0] for c in channels if c.endswith("rotation"))
            vals = [frame[start + n] for n, c in enumerate(channels) if c.endswith("rotation")]
            self.local[i] = R.from_euler(axes, vals, degrees=True).as_matrix()
            for n, c in enumerate(channels):
                if c.endswith("position"):
                    k = "XYZ".index(c[0])
                    self.root_pos[k] = frame[start + n] + j[2][k]

    def write(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_single_frame_bvh(str(self.source), self.to_frame(), str(path))


def rot_between(a, b):
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    s, c = np.linalg.norm(v), float(np.clip(a @ b, -1, 1))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, np.eye(3)[int(np.argmin(abs(a)))])
        return R.from_rotvec(axis / np.linalg.norm(axis) * np.pi).as_matrix()
    return R.from_rotvec(v / s * np.arctan2(s, c)).as_matrix()


def frame_from(x_axis, z_hint):
    """Orthonormal frame with exact x_axis and z as close as possible to z_hint."""
    x = np.asarray(x_axis, float)
    x = x / np.linalg.norm(x)
    z = np.asarray(z_hint, float) - x * (x @ z_hint)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)
