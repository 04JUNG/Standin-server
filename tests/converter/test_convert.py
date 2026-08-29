"""
변환 파이프라인 실동작 검증.

검사 항목
  A. 3개 출력 모드가 모두 FBX 를 산출하는가
  B. 산출 FBX 를 **다시 읽었을 때** 포즈가 실제로 들어가 있는가
     - rigged_rest : rest 포즈 자체가 바뀌어야 한다 (정적 임포터 대비)
     - rigged_anim : rest 는 T-pose 그대로, 애니메이션에 포즈가 있어야 한다
     - static_mesh : 본이 없어야 하고 메시 정점이 움직여 있어야 한다
  C. 포즈 충실도 RMSE 가 임계 이하인가
  D. 미러 스위치가 좌우를 실제로 뒤집는가
"""

from __future__ import annotations

import os
import sys

TEST_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(TEST_ROOT))
ARTIFACT_ROOT = os.path.abspath(
    os.environ.get("CONVERTER_TEST_ARTIFACT_ROOT", TEST_ROOT)
)
sys.path.insert(0, REPO_ROOT)

import bpy
from mathutils import Vector

from converter.convert import convert
from converter.bone_map import MIXAMO

ASSETS = os.path.join(ARTIFACT_ROOT, "assets")
OUT = os.path.join(ARTIFACT_ROOT, "out")
BVH = os.path.join(ASSETS, "pose_raise.bvh")
CHAR = os.path.join(ASSETS, "character_mixamo.fbx")

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")


def bone_dir(arm_name_filter, bone: str) -> Vector | None:
    """다시 임포트한 씬에서 지정 본의 rest 방향 벡터."""
    for o in bpy.data.objects:
        if o.type == "ARMATURE" and bone in o.data.bones:
            b = o.data.bones[bone]
            return (b.tail_local - b.head_local).normalized()
    return None


def reimport(path: str) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=path, ignore_leaf_bones=True,
                             automatic_bone_orientation=False)


def mesh_bbox() -> tuple[Vector, Vector]:
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        for v in o.data.vertices:
            w = o.matrix_world @ v.co
            for i in range(3):
                lo[i], hi[i] = min(lo[i], w[i]), max(hi[i], w[i])
    return lo, hi


LARM = MIXAMO["upperarm.L"]
RARM = MIXAMO["upperarm.R"]

# ---------------------------------------------------------------------------
# A/C. 세 모드 변환
# ---------------------------------------------------------------------------
reports = {}
for mode in ("rigged_rest", "rigged_anim", "static_mesh"):
    out = os.path.join(OUT, f"pose_{mode}.fbx")
    rep = convert(bvh_path=BVH, character_fbx=CHAR, out_path=out,
                  frame=1, output_mode=mode, embed_textures=False)
    reports[mode] = rep
    check(f"{mode}: 변환 성공", rep.ok, str(rep.warnings))
    check(f"{mode}: 파일 생성", os.path.exists(out) and os.path.getsize(out) > 1000,
          f"{os.path.getsize(out) if os.path.exists(out) else 0} bytes")
    check(f"{mode}: 22본 전부 매핑", rep.mapped_bones == 22, f"mapped={rep.mapped_bones}")
    check(f"{mode}: 프로파일 자동판별",
          rep.src_profile == "mixamo_noprefix" and rep.dst_profile == "mixamo",
          f"{rep.src_profile} -> {rep.dst_profile}")
    check(f"{mode}: 포즈 충실도 RMSE < 0.10",
          rep.pose_fidelity_rmse is not None and rep.pose_fidelity_rmse < 0.10,
          f"rmse={rep.pose_fidelity_rmse}")

# ---------------------------------------------------------------------------
# B. 재임포트 검사
# ---------------------------------------------------------------------------
# 원본 캐릭터: 왼팔은 +X 방향(T-pose)
reimport(CHAR)
d0 = bone_dir(None, LARM)
check("원본 캐릭터 왼팔이 T-pose(+X)", d0 is not None and d0.x > 0.9, f"dir={tuple(round(c,2) for c in d0)}")

# rigged_rest: rest 포즈가 바뀌어 팔이 위(+Z)를 향해야 한다
reimport(os.path.join(OUT, "pose_rigged_rest.fbx"))
d1 = bone_dir(None, LARM)
check("rigged_rest: rest 포즈에 포즈가 구워짐(왼팔 +Z)",
      d1 is not None and d1.z > 0.85, f"dir={tuple(round(c,2) for c in d1)}")
check("rigged_rest: 아마추어 유지",
      any(o.type == "ARMATURE" for o in bpy.data.objects))
check("rigged_rest: 메시 유지", any(o.type == "MESH" for o in bpy.data.objects))
lo_r, hi_r = mesh_bbox()
dR_rest = bone_dir(None, RARM)
check("rigged_rest: 오른팔은 아래로 (좌우 비대칭 유지)",
      dR_rest is not None and dR_rest.z < -0.5,
      f"dir={tuple(round(c, 2) for c in dR_rest)}")

# rigged_anim: rest 는 T-pose 그대로여야 한다
reimport(os.path.join(OUT, "pose_rigged_anim.fbx"))
d2 = bone_dir(None, LARM)
check("rigged_anim: rest 는 T-pose 유지(왼팔 +X)",
      d2 is not None and d2.x > 0.9, f"dir={tuple(round(c,2) for c in d2)}")
has_action = any(o.animation_data and o.animation_data.action
                 for o in bpy.data.objects if o.type == "ARMATURE")
check("rigged_anim: 애니메이션 트랙 존재", has_action)

# static_mesh: 본 없음 + 메시가 rigged_rest 와 같은 형상
reimport(os.path.join(OUT, "pose_static_mesh.fbx"))
check("static_mesh: 아마추어 없음",
      not any(o.type == "ARMATURE" for o in bpy.data.objects))
lo_s, hi_s = mesh_bbox()
check("static_mesh: 메시 형상이 rigged_rest 와 일치(포즈 반영됨)",
      (lo_s - lo_r).length < 0.05 and (hi_s - hi_r).length < 0.05,
      f"bbox_delta={(hi_s - hi_r).length:.4f}")

# T-pose 대비 실제로 형상이 바뀌었는지 (포즈가 안 들어간 채 통과하는 걸 방지)
reimport(CHAR)
lo_t, hi_t = mesh_bbox()
check("static_mesh: T-pose 대비 형상 변화 있음",
      (hi_s - hi_t).length > 0.1, f"bbox_delta={(hi_s - hi_t).length:.4f}")

# ---------------------------------------------------------------------------
# D. 미러
# ---------------------------------------------------------------------------
out_m = os.path.join(OUT, "pose_mirror.fbx")
rep_m = convert(bvh_path=BVH, character_fbx=CHAR, out_path=out_m,
                frame=1, mirror=True, output_mode="rigged_rest",
                embed_textures=False)
check("mirror: 변환 성공", rep_m.ok, str(rep_m.warnings))
check("mirror: 포즈 충실도 RMSE < 0.10",
      rep_m.pose_fidelity_rmse is not None and rep_m.pose_fidelity_rmse < 0.10,
      f"rmse={rep_m.pose_fidelity_rmse:.4f}")
reimport(out_m)
dl, dr = bone_dir(None, LARM), bone_dir(None, RARM)
check("mirror: 들린 팔이 왼쪽->오른쪽으로 바뀜",
      dr is not None and dr.z > 0.85 and (dl is None or dl.z < 0.5),
      f"L={tuple(round(c,2) for c in dl)} R={tuple(round(c,2) for c in dr)}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
failed = [r for r in RESULTS if not r[1]]
print(f"총 {len(RESULTS)}건 · 통과 {len(RESULTS) - len(failed)} · 실패 {len(failed)}")
for n, _, d in failed:
    print("  FAIL:", n, d)
sys.exit(1 if failed else 0)
