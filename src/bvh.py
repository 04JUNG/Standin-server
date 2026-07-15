"""
BVH 파싱 + 순전파(FK) + 관절명→COCO17 매핑. 라이브러리와 검수 스크립트의 단일 소스.

- parse_bvh: HIERARCHY(계층·OFFSET·CHANNELS) + MOTION(프레임) 파싱
- fk:        한 프레임의 채널값 → 관절 world 좌표(3D)
- load_coco17: BVH 1프레임 → COCO-17 (17,3) + scores  (검색 쿼리와 같은 규격)

지원 스켈레톤: CMU cgspeed BVH(LeftArm/LeftUpLeg…) 및 Mixamo(mixamorig:LeftArm…).
  → 이름은 접두사를 떼고 접미사로 매칭하므로 두 소스가 같은 매핑을 탄다.
얼굴 키포인트(코·눈·귀)는 BVH에 없으므로 Head로 근사(낮은 score) — 몸통 정규화 후 영향 작음.
"""
from __future__ import annotations

import math
import numpy as np

from .schema import COCO17


# ---------- 파싱 ----------
def parse_bvh(path: str):
    toks = open(path).read().split()
    i = 0
    joints = []          # [name, parent_idx, offset(3,), channels(list), is_end]
    stack = []

    def add(name, is_end=False):
        parent = stack[-1] if stack else -1
        joints.append([name, parent, np.zeros(3), [], is_end])
        return len(joints) - 1

    assert toks[i] == "HIERARCHY"; i += 1
    while toks[i] != "MOTION":
        t = toks[i]
        if t in ("ROOT", "JOINT"):
            add(toks[i + 1]); i += 2
        elif t == "End":                      # End Site
            add(joints[stack[-1]][0] + "_End", True); i += 2
        elif t == "{":
            stack.append(len(joints) - 1); i += 1
        elif t == "}":
            stack.pop(); i += 1
        elif t == "OFFSET":
            joints[stack[-1]][2] = np.array(list(map(float, toks[i + 1:i + 4]))); i += 4
        elif t == "CHANNELS":
            n = int(toks[i + 1]); joints[stack[-1]][3] = toks[i + 2:i + 2 + n]; i += 2 + n
        else:
            i += 1
    i += 1
    assert toks[i] == "Frames:"; nframes = int(toks[i + 1]); i += 2
    assert toks[i] == "Frame"; i += 3         # 'Frame' 'Time:' value
    nch = sum(len(j[3]) for j in joints)
    data = np.array(list(map(float, toks[i:i + nframes * nch]))).reshape(nframes, nch)
    return joints, data


def _rot(axis, deg):
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    if axis == "X": return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y": return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def fk(joints, frame):
    """frame(채널값 1D) → {joint_idx: world_pos(3,)}."""
    pos, world = {}, {}
    ci = 0
    for idx, (name, parent, off, chans, is_end) in enumerate(joints):
        t = off.astype(float).copy(); R = np.eye(3); order = ""; rots = {}
        for ch in chans:
            val = frame[ci]; ci += 1
            if ch.endswith("position"):
                ax = ch[0]
                t = t + np.array([val if ax == "X" else 0, val if ax == "Y" else 0,
                                  val if ax == "Z" else 0])
            elif ch.endswith("rotation"):
                order += ch[0]; rots[ch[0]] = val
        for ax in order:
            R = R @ _rot(ax, rots[ax])
        L = np.eye(4); L[:3, :3] = R; L[:3, 3] = t
        W = L if parent == -1 else world[parent] @ L
        world[idx] = W; pos[idx] = W[:3, 3]
    return pos


def bones(joints):
    return [(j[1], i) for i, j in enumerate(joints) if j[1] != -1]


# ---------- COCO17 매핑 ----------
# COCO 인덱스 → BVH 관절 접미사 후보(우선순위순). score: 매핑 신뢰도.
_MAP = {
    0:  (["Head"], 0.8),                        # nose ← Head(근사)
    1:  (["Head"], 0.4), 2: (["Head"], 0.4),    # eyes ← Head(근사)
    3:  (["Head"], 0.4), 4: (["Head"], 0.4),    # ears ← Head(근사)
    5:  (["LeftArm", "LeftShoulder"], 1.0),     # left_shoulder ← 상완 관절
    6:  (["RightArm", "RightShoulder"], 1.0),
    7:  (["LeftForeArm", "LeftElbow"], 1.0),
    8:  (["RightForeArm", "RightElbow"], 1.0),
    9:  (["LeftHand", "LeftWrist"], 1.0),
    10: (["RightHand", "RightWrist"], 1.0),
    11: (["LeftUpLeg", "LeftHip"], 1.0),
    12: (["RightUpLeg", "RightHip"], 1.0),
    13: (["LeftLeg", "LeftKnee"], 1.0),
    14: (["RightLeg", "RightKnee"], 1.0),
    15: (["LeftFoot", "LeftAnkle"], 1.0),
    16: (["RightFoot", "RightAnkle"], 1.0),
}


def _suffix_map(joints, pos):
    """{'LeftArm': world_pos, ...}  (mixamorig: 등 접두사 제거)."""
    out = {}
    for idx, j in enumerate(joints):
        suf = j[0].split(":")[-1]
        out.setdefault(suf, pos[idx])
    return out


def load_coco17(path: str, frame: int = 0):
    """
    BVH 1프레임 → (kp(17,3) float32, scores(17,) float32).
    hips 중심으로 평행이동해서 반환(투영이 몸 중심을 축으로 돌도록).
    """
    joints, data = parse_bvh(path)
    fr = data[min(frame, len(data) - 1)]
    pos = fk(joints, fr)
    smap = _suffix_map(joints, pos)

    kp = np.zeros((17, 3), dtype=np.float32)
    sc = np.zeros(17, dtype=np.float32)
    for ci, (cands, score) in _MAP.items():
        for suf in cands:
            if suf in smap:
                kp[ci] = smap[suf]; sc[ci] = score; break

    # hips 중심 이동(좌우 hip 중점). 둘 다 있으면 사용, 없으면 root.
    if sc[11] > 0 and sc[12] > 0:
        mid = (kp[11] + kp[12]) / 2.0
    else:
        mid = smap.get("Hips", smap.get("Hip", np.zeros(3)))
    kp[sc > 0] -= mid
    return kp, sc
