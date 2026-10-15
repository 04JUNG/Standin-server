"""간이 입체 3D 인형 렌더러(실제 BVH FK 좌표 → 회색 캡슐·구 볼륨).

옛 2D warm mannequin(`src/thumbnail_renderer.py`)처럼 관절 점을 선으로 잇지 않는다.
BVH 전방계산으로 얻은 실제 3D 관절 위치에 몸통·팔다리·머리 볼륨을 입히고, 정사영 레이캐스팅과
조명으로 음영을 넣는다. 서로 다른 포즈를 **같은 몸통 기준 시점**으로 그려 비교하기 위한 표시용이다.

한계(화면에 함께 표시할 것): 실제 캐릭터 FBX 피부 메시가 아니다. 손가락·손목 비틀림·피부 접촉·
관통은 보여 주지 못한다. 포즈를 바꾸거나 맞추지 않으며, 이미지 생성 모델을 쓰지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.bvh import fk

VERSION = "volume-mannequin-v2"
# 러프에 보이는 신체 범위만 그린다: full(전신) · knee(무릎 위) · hip(허리 위) · chest(가슴 위)
EXTENTS = ("full", "knee", "hip", "chest", "bust")
_DROP = {"full": set(), "knee": {"shin", "foot"}, "hip": {"thigh", "shin", "foot"},
         "chest": {"thigh", "shin", "foot", "pelvis", "low_spine"},
         "bust": {"thigh", "shin", "foot", "pelvis", "low_spine", "arm"}}   # 얼굴·어깨 클로즈업: 팔 생략

_GREY = np.array([196.0, 197.0, 200.0])
_LIGHT = np.array([-0.45, 0.65, 0.62])
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)


@dataclass(frozen=True)
class Capsule:
    a: np.ndarray
    b: np.ndarray
    r: float
    part: str = "upper"


def _suffix_positions(joints, positions):
    out = {}
    for index, joint in enumerate(joints):
        name = joint[0].split(":")[-1]
        out.setdefault(name, np.asarray(positions[index], dtype=float))
    return out


def _children(joints):
    kids = {i: [] for i in range(len(joints))}
    for i, joint in enumerate(joints):
        if joint[1] != -1:
            kids[joint[1]].append(i)
    return kids


def _first(named, *candidates):
    for name in candidates:
        if name in named:
            return named[name]
    return None


def body_points(joints, frame):
    """리그 차이를 흡수한 몸 기준점(월드 좌표). 없는 점은 None."""
    positions = fk(joints, frame)
    named = _suffix_positions(joints, positions)
    kids = _children(joints)
    index_of = {joint[0].split(":")[-1]: i for i, joint in enumerate(joints)}

    def child_tip(name):
        index = index_of.get(name)
        if index is None or not kids[index]:
            return None
        return np.asarray(positions[kids[index][0]], dtype=float)

    points = {
        "l_shoulder": _first(named, "LeftArm", "LeftShoulder"),
        "r_shoulder": _first(named, "RightArm", "RightShoulder"),
        "l_elbow": _first(named, "LeftForeArm", "LeftElbow"),
        "r_elbow": _first(named, "RightForeArm", "RightElbow"),
        "l_wrist": _first(named, "LeftHand", "LeftWrist"),
        "r_wrist": _first(named, "RightHand", "RightWrist"),
        "l_hip": _first(named, "LeftUpLeg", "LeftHip"),
        "r_hip": _first(named, "RightUpLeg", "RightHip"),
        "l_knee": _first(named, "LeftLeg", "LeftKnee"),
        "r_knee": _first(named, "RightLeg", "RightKnee"),
        "l_ankle": _first(named, "LeftFoot", "LeftAnkle"),
        "r_ankle": _first(named, "RightFoot", "RightAnkle"),
        "l_toe": _first(named, "LeftToeBase", "LeftToe", "LeftToe_End"),
        "r_toe": _first(named, "RightToeBase", "RightToe", "RightToe_End"),
        "neck": _first(named, "Neck", "Neck1"),
        "head": _first(named, "Head"),
        "l_hand_tip": child_tip("LeftHand"),
        "r_hand_tip": child_tip("RightHand"),
        "head_tip": child_tip("Head"),
    }
    required = ("l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
                "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle")
    missing = [name for name in required if points[name] is None]
    if missing:
        raise ValueError("volume mannequin: missing joints " + ",".join(missing))
    spine = []
    for i, joint in enumerate(joints):
        name = joint[0].split(":")[-1]
        if name.startswith("Spine") or name in ("Chest", "UpperChest"):
            spine.append(np.asarray(positions[i], dtype=float))
    points["spine"] = spine
    return points


def body_axes(points):
    """(인물의 오른쪽, 위, 얼굴 방향) 정규직교 축.

    사람은 얼굴이 +z, 위가 +y일 때 오른손이 -x에 있다 → 얼굴 = 위 × 오른쪽.
    좌우 이름이 붙은 실제 관절로 계산하므로 미러 포즈도 같은 규칙을 따른다.
    """
    shoulder = (points["l_shoulder"] + points["r_shoulder"]) / 2
    hip = (points["l_hip"] + points["r_hip"]) / 2
    up = shoulder - hip
    up /= np.linalg.norm(up)
    right = points["r_shoulder"] - points["l_shoulder"]
    right = right - up * np.dot(right, up)
    right /= np.linalg.norm(right)
    facing = np.cross(up, right)
    return right, up, facing


def capsules(points):
    shoulder = (points["l_shoulder"] + points["r_shoulder"]) / 2
    hip = (points["l_hip"] + points["r_hip"]) / 2
    length = float(np.linalg.norm(shoulder - hip))
    neck = points["neck"] if points["neck"] is not None else shoulder + 0.12 * (shoulder - hip)
    head = points["head"] if points["head"] is not None else neck + 0.2 * (shoulder - hip)
    up = (shoulder - hip) / length
    if points["head_tip"] is not None and np.linalg.norm(points["head_tip"] - head) > 0.05 * length:
        head_center = (head + points["head_tip"]) / 2
    else:
        head_center = head + 0.22 * length * up
    out = []
    chain = [hip] + [p for p in points["spine"] if np.linalg.norm(p - hip) > 1e-6] + [neck]
    for index in range(len(chain) - 1):
        t = index / max(1, len(chain) - 2)
        out.append(Capsule(chain[index], chain[index + 1], length * (0.19 + 0.04 * t),
                           "low_spine" if index < (len(chain) - 1) / 2 else "upper"))
    out += [
        Capsule(points["l_hip"], points["r_hip"], 0.17 * length, "pelvis"),
        Capsule(points["l_shoulder"], points["r_shoulder"], 0.12 * length),
        Capsule(neck, points["l_shoulder"], 0.11 * length),
        Capsule(neck, points["r_shoulder"], 0.11 * length),
        Capsule(neck, head, 0.075 * length),
        Capsule(head_center, head_center, 0.21 * length),
    ]
    if points.get("nose") is not None:
        out.append(Capsule(points["nose"], points["nose"], 0.06 * length))   # 얼굴 방향 표시
    for side in ("l", "r"):
        out += [
            Capsule(points[side + "_shoulder"], points[side + "_elbow"], 0.075 * length, "arm"),
            Capsule(points[side + "_elbow"], points[side + "_wrist"], 0.064 * length, "arm"),
            Capsule(points[side + "_hip"], points[side + "_knee"], 0.115 * length, "thigh"),
            Capsule(points[side + "_knee"], points[side + "_ankle"], 0.085 * length, "shin"),
        ]
        tip = points[side + "_hand_tip"]
        hand_end = tip if tip is not None and np.linalg.norm(tip - points[side + "_wrist"]) > 1e-6 \
            else points[side + "_wrist"] + 0.4 * (points[side + "_wrist"] - points[side + "_elbow"])
        hand_dir = hand_end - points[side + "_wrist"]
        hand_end = points[side + "_wrist"] + hand_dir / max(np.linalg.norm(hand_dir), 1e-8) * 0.17 * length
        out.append(Capsule(points[side + "_wrist"], hand_end, 0.058 * length, "arm"))
        toe = points[side + "_toe"]
        if toe is not None:
            out.append(Capsule(points[side + "_ankle"], toe, 0.06 * length, "foot"))
    return out, length


def _intersect(capsule, ro, rd):
    """iq 캡슐 교차. ro:(N,3), rd:(3,). 반환 t(N,) — 없으면 inf."""
    pa, pb, r = capsule.a, capsule.b, capsule.r
    ba = pb - pa
    baba = float(np.dot(ba, ba))
    t_out = np.full(len(ro), np.inf)
    if baba < 1e-12:
        oc = ro - pa
        b = oc @ rd if rd.ndim==1 else np.einsum('ij,ij->i',oc,rd)
        c = np.einsum("ij,ij->i", oc, oc) - r * r
        h = b * b - c
        ok = h > 0
        t_out[ok] = -b[ok] - np.sqrt(h[ok])
        return t_out
    oa = ro - pa
    bard = float(np.dot(ba, rd)) if rd.ndim==1 else rd@ba
    baoa = oa @ ba
    rdoa = oa @ rd if rd.ndim==1 else np.einsum('ij,ij->i',oa,rd)
    oaoa = np.einsum("ij,ij->i", oa, oa)
    a = baba - bard * bard
    b = baba * rdoa - baoa * bard
    c = baba * oaoa - baoa * baoa - r * r * baba
    h = b * b - a * c
    hit = h >= 0
    nonparallel = np.abs(a)>1e-12
    if np.any(nonparallel):
        t = np.where(hit & nonparallel, (-b - np.sqrt(np.maximum(h, 0))) / np.where(nonparallel,a,1), np.inf)
        y = baoa + np.where(hit, t, 0.0) * bard
        body = hit & nonparallel & (y > 0) & (y < baba)
        t_out[body] = t[body]
    else:
        y = np.zeros(len(ro))
        body = np.zeros(len(ro), dtype=bool)
    # 양 끝 반구
    rest = hit & ~body
    if rest.any():
        cap_oc = np.where((y <= 0)[:, None], oa, ro - pb)
        bb = cap_oc @ rd if rd.ndim==1 else np.einsum('ij,ij->i',cap_oc,rd)
        cc = np.einsum("ij,ij->i", cap_oc, cap_oc) - r * r
        hh = bb * bb - cc
        ok = rest & (hh > 0)
        t_out[ok] = -bb[ok] - np.sqrt(hh[ok])
    return t_out


def _normal(capsule, pos):
    ba = capsule.b - capsule.a
    baba = float(np.dot(ba, ba))
    if baba < 1e-12:
        closest = np.broadcast_to(capsule.a, pos.shape)
    else:
        s = np.clip(((pos - capsule.a) @ ba) / baba, 0, 1)
        closest = capsule.a + s[:, None] * ba
    n = pos - closest
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def camera_points(points, yaw_deg=0.0, pitch_deg=8.0):
    """몸통 기준 정면(얼굴이 화면을 봄)에서 yaw만큼 돈 카메라 좌표로 변환한 함수."""
    right, up, facing = body_axes(points)
    origin = (points["l_hip"] + points["r_hip"]) / 2
    # 화면 x = 인물의 왼쪽(관찰자 기준 오른쪽), y = 위, z = 관찰자 쪽
    basis = np.stack([-right, up, facing])
    yaw, pitch = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    transform = rx @ ry @ basis

    def to_camera(p):
        return None if p is None else transform @ (np.asarray(p, float) - origin)

    out = {}
    for key, value in points.items():
        out[key] = [to_camera(p) for p in value] if key == "spine" else to_camera(value)
    return out


def add_face_direction(joints, frame, points):
    """머리 회전(가슴 대비)을 몸 정면 방향에 적용해 코 위치를 더한다. 고개 방향을 눈으로 보이게 한다."""
    from .part_composition import world_transforms
    index_of = {joint[0].split(":")[-1]: i for i, joint in enumerate(joints)}
    head = index_of.get("Head")
    if head is None or points.get("head") is None:
        return points
    neck = index_of.get("Neck", index_of.get("Neck1"))
    chest = joints[neck][1] if neck is not None else joints[head][1]
    if chest is None or chest < 0:
        return points
    world = world_transforms(joints, frame)
    _right, up, facing = body_axes(points)
    head_facing = world[head][:3, :3] @ world[chest][:3, :3].T @ facing
    shoulder = (points["l_shoulder"] + points["r_shoulder"]) / 2
    hip = (points["l_hip"] + points["r_hip"]) / 2
    length = float(np.linalg.norm(shoulder - hip))
    tip = points.get("head_tip")
    if tip is not None and np.linalg.norm(tip - points["head"]) > 0.05 * length:
        center = (points["head"] + tip) / 2
    else:
        center = points["head"] + 0.22 * length * up
    points = dict(points)
    points["nose"] = center + head_facing / max(np.linalg.norm(head_facing), 1e-9) * 0.2 * length
    return points


def visible_extent(joint_mask):
    """러프에서 관측된 COCO17 관절로 그릴 범위를 정한다(가장 아래 관측 관절 기준)."""
    m = [bool(v) for v in (joint_mask or [])]
    if len(m) != 17:
        return "full"
    if m[15] or m[16]:
        return "full"
    if m[13] or m[14]:
        return "knee"
    if m[11] or m[12]:
        return "hip"
    return "chest"


def extent_for(scope, lower_observed, lower_known, joint_mask):
    """그릴 범위 = Gemini가 본 인물 범위(scope·하체 보임)를 우선, 관절은 보조.

    관절 검출기는 러프의 다리를 자주 놓치고(선이 거칠거나 칸에 잘림) 가끔 없는 발목을 잡는다.
      full  → 전신. 단 Gemini가 '하체 안 보임'이라 했고 양 무릎·발목이 모두 잡히지 않았으면 허리 위.
      half  → 무릎 위(허벅지까지)
      bust/head → 얼굴·어깨(팔 생략)
      모름  → 관절 기준(visible_extent)
    """
    m = [bool(v) for v in (joint_mask or [])]
    legs_seen = len(m) == 17 and all(m[i] for i in (13, 14, 15, 16))
    if scope == "full":
        if lower_known and lower_observed is False and not legs_seen:
            return "hip"
        return "full"
    if scope == "half":
        return "full" if legs_seen else "knee"
    if scope in ("bust", "head"):
        return "bust"
    return visible_extent(joint_mask)


def render(joints, frame, *, yaw_deg=0.0, pitch_deg=8.0, size=(300, 400), supersample=2, extent="full",search_camera=None):
    """BVH 1프레임 → PIL RGBA 이미지(투명 배경, 회색 볼륨 인형). extent로 보이는 범위만 그린다."""
    if extent not in EXTENTS:
        raise ValueError("unknown extent: " + str(extent))
    points = add_face_direction(joints, frame, body_points(joints, frame))
    if search_camera is None:
        points=camera_points(points,yaw_deg,pitch_deg)
    else:
        from src.bvh import coco17_from_fk
        p,_=coco17_from_fk(joints,fk(joints,frame))
        origin=(p[11]+p[12])/2;scale=np.linalg.norm((p[5]+p[6])/2-origin)
        y,x=np.deg2rad([search_camera['yaw'],search_camera.get('pitch',0)])
        ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
        rx=np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
        def transform(value):
            if value is None:return None
            result=(rx@ry)@(np.asarray(value)-origin)/scale
            result[2]*=-1 # retrieval +Z farther; ray renderer +Z toward viewer
            return result
        points={k:[transform(v) for v in value] if k=='spine' else transform(value) for k,value in points.items()}
    caps, length = capsules(points)
    caps = [c for c in caps if c.part not in _DROP[extent]]
    width, height = size[0] * supersample, size[1] * supersample
    lo = np.min([np.minimum(c.a, c.b) - c.r for c in caps], axis=0)
    hi = np.max([np.maximum(c.a, c.b) + c.r for c in caps], axis=0)
    distance=search_camera.get('distance') if search_camera else None
    if distance is not None:
        # Bound projected capsule boxes; normalized perspective rays share the
        # exact eye/distance convention of camera_search.project, no 2D warping.
        corners=np.array([point+np.array([a,b,c])*cap.r for cap in caps for point in (cap.a,cap.b)
                          for a in (-1,1) for b in (-1,1) for c in (-1,1)])
        depth=distance-corners[:,2]
        if np.any(depth<=.05):raise ValueError('preview_camera_too_close')
        screen=corners[:,:2]/depth[:,None]
        lo[:2]=screen.min(axis=0);hi[:2]=screen.max(axis=0)
    span = max((hi[0] - lo[0]) / (width * 0.86), (hi[1] - lo[1]) / (height * 0.9))
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    xs = cx + (np.arange(width) - width / 2 + 0.5) * span
    ys = cy - (np.arange(height) - height / 2 + 0.5) * span
    gx, gy = np.meshgrid(xs, ys)
    ro = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, hi[2] + length)], axis=1)
    rd = np.array([0.0, 0.0, -1.0])
    if distance is not None:
        ro=np.broadcast_to([0.,0.,float(distance)],ro.shape).copy()
        rd=np.stack([gx.ravel(),gy.ravel(),-np.ones(gx.size)],axis=1)
        rd/=np.linalg.norm(rd,axis=1,keepdims=True)
    best = np.full(len(ro), np.inf)
    which = np.full(len(ro), -1)
    for index, capsule in enumerate(caps):
        t = _intersect(capsule, ro, rd)
        closer = t < best
        best[closer] = t[closer]
        which[closer] = index
    hit = np.isfinite(best)
    normals = np.zeros((len(ro), 3))
    for index, capsule in enumerate(caps):
        sel = hit & (which == index)
        if sel.any():
            normals[sel] = _normal(capsule, ro[sel] + best[sel, None] * (rd if rd.ndim==1 else rd[sel]))
    diffuse = np.clip(normals @ _LIGHT, 0, 1)
    fill = np.clip(normals @ np.array([0.5, -0.1, 0.85]), 0, 1)
    rim = (1 - np.clip(normals[:, 2], 0, 1)) ** 3
    shade = 0.30 + 0.62 * diffuse + 0.14 * fill - 0.10 * rim
    rgb = np.clip(_GREY[None, :] * shade[:, None], 0, 255)
    alpha = hit.astype(float) * 255
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[..., :3] = rgb.reshape(height, width, 3).astype(np.uint8)
    image[..., 3] = alpha.reshape(height, width).astype(np.uint8)
    out = Image.fromarray(image, "RGBA")
    if supersample > 1:
        out = out.resize(size, Image.LANCZOS)
    return out
