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
import os
import re

import numpy as np

from .schema import COCO17


# ---------- 파싱 ----------
def parse_bvh(path: str):
    # 일부 Blender/변환 스크립트는 HIERARCHY의 마지막 닫는 괄호와
    # MOTION 헤더 사이 줄바꿈을 생략해 ``}MOTION``으로 기록한다.
    # 공백 토큰화 전에 이 경계만 정규화해 원본 BVH를 다시 쓰지 않고 읽는다.
    text = open(path).read().replace("}MOTION", "} MOTION")
    toks = text.split()
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


# ---------- 채널 인덱스 (refine이 회전각을 파라미터로 잡는 데 사용) ----------

def channel_starts(joints):
    """관절 인덱스 → MOTION 프레임 배열에서 그 관절의 채널이 시작하는 위치."""
    starts, ci = [], 0
    for j in joints:
        starts.append(ci)
        ci += len(j[3])
    return starts


def find_joint(joints, suffix: str) -> int:
    """접미사(mixamorig: 등 접두사 무시)로 관절 인덱스 찾기. 없으면 -1."""
    for idx, j in enumerate(joints):
        if j[0].split(":")[-1] == suffix:
            return idx
    return -1


def rotation_channel_indices(joints, joint_idx: int):
    """해당 관절의 '회전' 채널이 프레임 배열에서 갖는 전역 인덱스 목록."""
    start = channel_starts(joints)[joint_idx]
    return [start + k for k, ch in enumerate(joints[joint_idx][3])
            if ch.endswith("rotation")]


# ---------- 쓰기 ----------

def hierarchy_text(path: str) -> str:
    """원본 BVH의 HIERARCHY 블록을 텍스트 그대로 반환(MOTION 직전까지)."""
    text = open(path).read()
    match = re.search(r"(?im)\bMOTION\s*\r?\n\s*Frames:", text)
    if match:
        return text[: match.start()].rstrip()
    raise ValueError(f"MOTION 섹션이 없습니다: {path}")


def single_frame_bvh_text(src_path: str, frame_values,
                          frame_time: float = 0.033333) -> str:
    """
    원본의 HIERARCHY를 **원문 그대로 복사**하고 MOTION만 1프레임으로 교체한 본문.

    계층·OFFSET(뼈 길이)을 재직렬화하지 않는 것이 핵심이다. refine은 회전각만
    바꾸므로 뼈 길이·계층이 달라질 이유가 없고, 재직렬화하면 반올림 오차나
    파서 미지원 구문(주석 등)으로 원본이 조용히 손상될 수 있다.

    개행은 항상 LF다. hierarchy_text()가 universal newlines로 읽어 재조립하고
    나머지는 여기서 직접 만들기 때문에, 원본이 CRLF여도 결과는 LF다.
    """
    head = hierarchy_text(src_path)
    vals = " ".join(f"{float(v):.6f}" for v in np.asarray(frame_values).ravel())
    return f"{head}\nMOTION\nFrames: 1\nFrame Time: {frame_time:.6f}\n{vals}\n"


def write_single_frame_bvh(src_path: str, frame_values, out_path: str,
                           frame_time: float = 0.033333) -> str:
    """single_frame_bvh_text()의 결과를 파일로 저장한다."""
    text = single_frame_bvh_text(src_path, frame_values, frame_time)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # newline="\n" 필수. 미지정이면 "\n"이 os.linesep으로 번역돼 Windows에서만
    # CRLF 파일이 나오고, 같은 조정본이 경로에 따라 다른 바이트가 된다.
    with open(out_path, "w", newline="\n") as f:
        f.write(text)
    return out_path


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


def coco17_from_fk(joints, pos):
    """
    이미 계산된 FK 결과 → (kp(17,3) float32, scores(17,) float32).
    hips 중심으로 평행이동해서 반환(투영이 몸 중심을 축으로 돌도록).

    load_coco17과 refine의 반복 전방계산이 **이 함수를 공유**한다.
    한쪽만 바꾸면 라이브러리 색인과 refine 목표가 어긋난다.
    """
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


def load_coco17(path: str, frame: int = 0):
    """BVH 1프레임 → (kp(17,3) float32, scores(17,) float32)."""
    joints, data = parse_bvh(path)
    fr = data[min(frame, len(data) - 1)]
    return coco17_from_fk(joints, fk(joints, fr))
