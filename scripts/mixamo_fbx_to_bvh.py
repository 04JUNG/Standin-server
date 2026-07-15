"""
Mixamo FBX 폴더 → 클립당 대표 포즈 K개를 각각 1프레임 BVH로 (Blender 헤드리스 배치).

실행:
    blender --background --python scripts/mixamo_fbx_to_bvh.py -- <in_fbx_dir> <out_bvh_dir> [옵션]
옵션:
    --num K        클립당 뽑을 포즈 수(기본 3). 뽑은 뒤 사람이 검수로 keeper만 남긴다.
    --frame N      전략 무시, 모든 클립을 N번 프레임 1개로 고정(수동).

선택 방식(다중 포즈):
    1) 표현 관절(팔·다리·머리·척추)을 hips 상대좌표로 봐서 프레임별 포즈 벡터를 만든다.
    2) '유지되는(속도 국소최소)' 프레임 = 자연스러운 포즈 후보만 추린다(스윙 중간 흐릿한 프레임 배제).
    3) 후보 중 '중립에서 가장 벗어난' 프레임(제스처 정점)을 시드로,
       farthest-point 방식으로 서로 가장 다양한 K개를 고른다(거의 같은 프레임 중복 방지).

이전 버그: 'Hips 이동 최소'만 봐서 팔만 움직이는 인사 포즈에서 가만히 선 프레임을 뽑았음 → 팔 포함으로 수정.

주의: Blender 3.x+ 안에서만 동작(bpy). Mixamo 조인트명 'mixamorig:' 접두사는 접미사 매칭으로 흡수.
      출력은 Clip_01.bvh, Clip_02.bvh ... 각 1프레임 1인 BVH → load_bvh_pose가 동일 처리.
"""
import bpy, sys, os

EXPRESSIVE = {
    "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand",
    "Head", "Neck", "Spine", "Spine1", "Spine2",
    "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg", "LeftFoot", "RightFoot",
}


def _args():
    a = sys.argv[sys.argv.index("--") + 1:]
    in_dir, out_dir = a[0], a[1]
    num = int(a[a.index("--num") + 1]) if "--num" in a else 3
    frame = int(a[a.index("--frame") + 1]) if "--frame" in a else None
    return in_dir, out_dir, num, frame


def _clear():
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()


def _import_fbx(path):
    bpy.ops.import_scene.fbx(filepath=path, automatic_bone_orientation=True)


def _armature():
    for o in bpy.data.objects:
        if o.type == 'ARMATURE':
            return o
    return None


def _range(arm):
    s, e = arm.animation_data.action.frame_range
    return int(s), int(e)


def _bone(arm, suffix):
    for b in arm.pose.bones:
        if b.name.split(':')[-1] == suffix:
            return b
    return None


def _pose_vec(arm, hips):
    o = (arm.matrix_world @ hips.head)
    v = []
    for suf in sorted(EXPRESSIVE):
        b = _bone(arm, suf)
        if b is not None:
            p = (arm.matrix_world @ b.head) - o
            v.extend((p.x, p.y, p.z))
    return v


def _d(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _select_frames(arm, s, e, k):
    scene = bpy.context.scene
    hips = _bone(arm, "Hips") or _bone(arm, "Hip")
    if hips is None or e <= s:
        return [(s + e) // 2]

    frames = list(range(s, e + 1))
    vecs = []
    for f in frames:
        scene.frame_set(f)
        vecs.append(_pose_vec(arm, hips))
    dim = min(len(v) for v in vecs)
    vecs = [v[:dim] for v in vecs]

    # 속도(인접 프레임 변화)와 중립(중앙값) 대비 편차
    vel = [0.0] + [_d(vecs[i], vecs[i - 1]) for i in range(1, len(vecs))]
    ref = [sorted(col)[len(col) // 2] for col in zip(*vecs)]
    dev = [_d(v, ref) for v in vecs]

    # 후보 = 속도 국소최소('유지되는' 프레임). 없으면 전체.
    cand = [i for i in range(1, len(vecs) - 1)
            if vel[i] <= vel[i - 1] and vel[i] <= vel[i + 1]]
    if len(cand) < k:
        cand = list(range(len(vecs)))

    # 시드 = 편차 최대(정점). 이후 farthest-point로 다양하게 K개.
    chosen = [max(cand, key=lambda i: dev[i])]
    while len(chosen) < k and len(chosen) < len(cand):
        nxt = max((i for i in cand if i not in chosen),
                  key=lambda i: min(_d(vecs[i], vecs[j]) for j in chosen))
        # 거의 같은 프레임(다양성 없음)이면 중단
        if min(_d(vecs[nxt], vecs[j]) for j in chosen) < 1e-4:
            break
        chosen.append(nxt)

    return sorted(frames[i] for i in chosen)


def _export_bvh_single(path, f):
    bpy.context.scene.frame_set(f)
    bpy.ops.export_anim.bvh(filepath=path, frame_start=f, frame_end=f,
                            root_transform_only=False)


def main():
    in_dir, out_dir, num, fixed = _args()
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for fn in sorted(os.listdir(in_dir)):
        if not fn.lower().endswith(".fbx"):
            continue
        _clear()
        _import_fbx(os.path.join(in_dir, fn))
        arm = _armature()
        if arm is None or arm.animation_data is None:
            print("skip (no anim):", fn); continue
        s, e = _range(arm)
        picks = [max(s, min(e, fixed))] if fixed is not None \
            else _select_frames(arm, s, e, num)
        base = os.path.splitext(fn)[0]
        for idx, f in enumerate(picks, 1):
            out = os.path.join(out_dir, f"{base}_{idx:02d}.bvh")
            _export_bvh_single(out, f)
            total += 1
        print(f"[ok] {fn}: {len(picks)} poses @ frames {picks} (of {s}-{e})")
    print(f"done: {total} BVH written to {out_dir}")


if __name__ == "__main__":
    main()
