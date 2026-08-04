"""
포즈 라이브러리: 로드 → 대표 키프레임 → 다중 가상카메라 2D 투영 → 정규화 피처 색인.

핵심 아이디어(설계문서 v2 §3, §12, §15):
  '그림에서 3D 복원(리프팅)'을 우회하고, 이미 3D인 포즈를 여러 카메라로 2D 투영해
  2D 스켈레톤끼리 비교한다. 쿼리(추출 스켈레톤)와 같은 피처 공간을 쓴다.

데이터 상태(사용자 확인): 실데이터는 나중, 지금은 소량+합성으로 검색 로직만 검증.
→ 실제 BVH가 꽂히는 자리는 load_bvh_pose()에 TODO로 명시.
"""
from __future__ import annotations

import os
import pickle
from typing import List

import numpy as np

from .schema import LibraryEntry, View, Shot, Action, Relationship, COCO17
from .features import normalize_skeleton, _BODY
from .bvh import load_coco17


# ---- 3D → 2D 투영 --------------------------------------------------------

def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


# 가상 카메라 = Y축(수직축) 회전각 → View 라벨.
VIRTUAL_CAMERAS = {
    View.FRONT: 0.0,
    View.THREE_QUARTER: np.deg2rad(45),
    View.SIDE: np.deg2rad(90),
    View.BACK: np.deg2rad(180),
}


def project_3d_to_2d(joints3d: np.ndarray, y_angle: float) -> np.ndarray:
    """
    joints3d: (17,3) → 2D (17,2). Y축 회전 후 정사영(x,y)만 취함.
    실제로는 원근/카메라 높이도 넣을 수 있으나 MVP는 정사영+수직축 회전으로 충분.
    """
    r = _rot_y(y_angle)
    p = joints3d @ r.T
    return p[:, :2].astype(np.float32)


def view_angle(view) -> float:
    """View(또는 그 문자열 값) → 가상 카메라 Y축 회전각."""
    return VIRTUAL_CAMERAS[view if isinstance(view, View) else View(view)]


def pose_to_feature(joints3d: np.ndarray, view, scores: np.ndarray | None = None
                    ) -> np.ndarray:
    """
    3D 포즈 → 특정 view의 정규화 피처(34,).

    ⚠ 피처 공간 대칭성의 단일 소스(CLAUDE.md 불변식 4).
    라이브러리 색인(build_entries_from_pose)과 refine의 전방계산이 **반드시**
    이 함수를 공유한다. 한쪽만 바꾸면 검색이나 refine이 조용히 망가진다.
    """
    if scores is None:
        scores = np.ones(17, dtype=np.float32)
    kp2d = project_3d_to_2d(joints3d, view_angle(view)).copy()
    # 투영 좌표는 y가 위로 +이므로 이미지 좌표(y 아래로 +)와 맞추려 부호 반전
    kp2d[:, 1] *= -1
    return normalize_skeleton(kp2d, scores)


# ---- 3D 포즈 소스 --------------------------------------------------------

def load_bvh_pose(bvh_path: str, frame: int = 0):
    """
    실제 BVH → 대표 키프레임의 COCO-17 (17,3) 관절 + scores(17,).
    src.bvh가 파싱·FK·관절명 매핑을 담당(cgspeed·Mixamo 이름 모두 처리, hips 중심 반환).
    단일 프레임 BVH(우리 워크플로 산출물)면 frame=0. 원본 멀티프레임이면 대표 프레임 지정.
    """
    return load_coco17(bvh_path, frame)


# 합성 3D 포즈 몇 종(검색 로직 검증용). 힙 중심, Y는 위로 +.
def _synthetic_pose(kind: str) -> np.ndarray:
    # base: 곧은 서기 (x,y,z)
    base = np.array([
        [0.0, 0.85, 0], [-0.05, 0.9, 0], [0.05, 0.9, 0], [-0.1, 0.88, 0], [0.1, 0.88, 0],
        [-0.18, 0.55, 0], [0.18, 0.55, 0], [-0.22, 0.2, 0], [0.22, 0.2, 0],
        [-0.24, -0.1, 0], [0.24, -0.1, 0], [-0.12, 0.0, 0], [0.12, 0.0, 0],
        [-0.13, -0.5, 0], [0.13, -0.5, 0], [-0.14, -0.95, 0], [0.14, -0.95, 0],
    ], dtype=np.float32)
    p = base.copy()
    if kind == "sitting":       # 무릎 접기: 무릎/발목 앞(+z)·위로
        p[[13, 14]] += np.array([0, 0.35, 0.3])
        p[[15, 16]] += np.array([0, 0.15, 0.55])
    elif kind == "reaching":    # 오른팔 위로 뻗기
        p[8] = np.array([0.28, 0.55, 0]); p[10] = np.array([0.30, 0.9, 0])
    elif kind == "walking":     # 다리 벌림
        p[15] += np.array([-0.2, 0.05, 0.2]); p[16] += np.array([0.2, 0.05, -0.2])
    elif kind == "running":
        p[15] += np.array([-0.25, 0.2, 0.35]); p[16] += np.array([0.25, 0.1, -0.35])
        p[9] += np.array([-0.1, 0.3, 0]); p[10] += np.array([0.1, 0.3, 0])
    return p


# ---- 인덱스 구축 ---------------------------------------------------------

def build_entries_from_pose(pose_id: str, joints3d: np.ndarray, tags: dict,
                            bvh_path: str | None = None,
                            scores: np.ndarray | None = None) -> List[LibraryEntry]:
    """
    한 3D 포즈 → 가상 카메라 수만큼 LibraryEntry 생성(각 view별 2D 투영 피처).
    scores: 관절 신뢰도(BVH 얼굴 근사·결측 마스킹용). None이면 전부 1.
    """
    if scores is None:
        scores = np.ones(17, dtype=np.float32)
    joints3d = np.asarray(joints3d, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if joints3d.shape != (17, 3):
        raise ValueError(
            f"pose '{pose_id}' COCO mapping must have shape (17,3), got {joints3d.shape}"
        )
    if scores.shape != (17,) or not np.isfinite(joints3d).all() or not np.isfinite(scores).all():
        raise ValueError(f"pose '{pose_id}' has invalid COCO mapping values")
    missing_body = [index for index in _BODY if scores[index] < 0.3]
    if missing_body:
        raise ValueError(
            f"pose '{pose_id}' has incomplete BVH body mapping: {missing_body}"
        )
    entries = []
    for view in VIRTUAL_CAMERAS:
        feat = pose_to_feature(joints3d, view, scores)   # ← refine과 공유
        t = dict(tags); t["view"] = view.value
        entries.append(LibraryEntry(pose_id=pose_id, view=view, feature=feat,
                                    tags=t, bvh_path=bvh_path))
    return entries


def build_synthetic_index() -> List[LibraryEntry]:
    """
    합성 포즈 세트로 인덱스 구축. 실데이터가 붙으면 이 함수를
    'BVH 폴더 순회 → load_bvh_pose → build_entries_from_pose'로 교체한다.
    """
    specs = [
        ("stand_solo",   "standing", Action.STANDING, Relationship.SOLO),
        ("sit_solo",     "sitting",  Action.SITTING,  Relationship.SOLO),
        ("reach_solo",   "reaching", Action.REACHING, Relationship.SOLO),
        ("walk_solo",    "walking",  Action.WALKING,  Relationship.SOLO),
        ("run_solo",     "running",  Action.RUNNING,  Relationship.SOLO),
    ]
    entries: List[LibraryEntry] = []
    for pose_id, kind, action, rel in specs:
        joints = _synthetic_pose(kind)
        tags = {"shot": Shot.FULL_HALF.value, "action": action.value,
                "relationship": rel.value}
        entries.extend(build_entries_from_pose(pose_id, joints, tags,
                                               bvh_path=f"data/bvh/{pose_id}.bvh"))
    return entries


# ---- 저장/로드 -----------------------------------------------------------

import glob as _glob


def infer_tags_from_name(fname: str) -> dict:
    """
    파일명에서 임시 태그 추론(실 태깅 전 플레이스홀더).
    실제로는 VLM 반자동 태깅+검수로 대체(설계문서 §12). 지금은 파일명 힌트만.
    예: 'Waving_02.bvh' → action 매칭 시도, 없으면 other.
    """
    low = fname.lower()
    action = Action.OTHER
    for a in Action:
        if a.value in low:
            action = a; break
    return {"shot": Shot.FULL_HALF.value, "action": action.value,
            "relationship": Relationship.SOLO.value}


def build_index_from_bvh_dir(bvh_dir: str, tag_fn=None,
                             frame: int = 0) -> List[LibraryEntry]:
    """
    실 BVH 폴더 → 인덱스. 각 .bvh를 load_bvh_pose로 읽어 뷰별 색인 생성.
    tag_fn(fname)->dict 로 태그 주입(기본=파일명 추론). pose_id=파일명(확장자 제외).
    """
    tag_fn = tag_fn or (lambda fn: infer_tags_from_name(fn))
    entries: List[LibraryEntry] = []
    for path in sorted(_glob.glob(os.path.join(bvh_dir, "*.bvh"))):
        fname = os.path.basename(path)
        pose_id = os.path.splitext(fname)[0]
        kp, scores = load_bvh_pose(path, frame)
        tags = tag_fn(fname)
        entries.extend(build_entries_from_pose(pose_id, kp, tags,
                                               bvh_path=path, scores=scores))
    return entries


def save_index(entries: List[LibraryEntry], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(entries, f)


def load_index(path: str) -> List[LibraryEntry]:
    with open(path, "rb") as f:
        return pickle.load(f)
