"""
변환 1건을 수행하는 엔트리포인트.

사용:
    python -m converter.convert --job job.json
    (또는) blender --background --python -m ... 동일 로직

워커는 이 프로세스를 **1건당 1회** 띄운다. Blender 상태가 프로세스에 누적되는
문제(씬 잔여물·메모리 파편화)를 프로세스 경계로 끊는 게 운영상 가장 싸다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import retarget as rt
from .retarget import ConvertReport


def convert(
    *,
    bvh_path: str,
    character_fbx: str,
    out_path: str,
    frame: int = 0,
    mirror: bool = False,
    output_mode: str = "rigged_rest",
    apply_root_translation: bool = False,
    embed_textures: bool = True,
    src_profile: str | None = None,
    dst_profile: str | None = None,
    palm_roll_mu: float | dict[str, float] | None = None,
) -> ConvertReport:
    t0 = time.time()
    rt.reset_scene()

    dst_arm, meshes = rt.import_character(character_fbx)
    src_arm = rt.import_bvh(bvh_path, frame=frame)

    rep = ConvertReport(output_mode=output_mode, frame=frame)
    rep = rt.retarget(
        src_arm, dst_arm,
        src_profile=src_profile, dst_profile=dst_profile,
        mirror=mirror,
        apply_root_translation=apply_root_translation,
        palm_roll_mu=palm_roll_mu,
        report=rep,
    )
    if not rep.ok:
        return rep

    rep.pose_fidelity_rmse = rt.pose_fidelity_rmse(
        src_arm, dst_arm, rep.src_profile, rep.dst_profile, mirror=mirror
    )
    rep.skeleton_baseline_rmse = rt.skeleton_baseline_rmse(
        src_arm, dst_arm, rep.src_profile, rep.dst_profile
    )
    rep.pose_fidelity_delta = rep.pose_fidelity_rmse - rep.skeleton_baseline_rmse

    # 소스 아마추어는 산출물에 포함하지 않는다
    import bpy
    bpy.data.objects.remove(src_arm, do_unlink=True)

    rt.apply_output_mode(dst_arm, meshes, output_mode)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    rt.export_fbx(
        out_path,
        embed_textures=embed_textures,
        bake_anim=(output_mode == "rigged_anim"),
    )

    rep.warnings.append(f"elapsed_sec={time.time() - t0:.2f}")
    return rep


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--job", help="JSON 파일 경로 (인자 일괄 전달)")
    p.add_argument("--bvh")
    p.add_argument("--character")
    p.add_argument("--out")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--output-mode", default="rigged_rest", choices=rt.OUTPUT_MODES)
    p.add_argument(
        "--palm-roll-mu", type=float, default=None,
        help="QA only: uniform mu ladder member; user-approved QA default is 0.5",
    )
    p.add_argument("--report", help="리포트 JSON 출력 경로")
    # blender --background --python script -- <args> 대응
    argv = argv if argv is not None else (
        sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    )
    a = p.parse_args(argv)

    kwargs = json.load(open(a.job)) if a.job else dict(
        bvh_path=a.bvh, character_fbx=a.character, out_path=a.out,
        frame=a.frame, mirror=a.mirror, output_mode=a.output_mode,
        palm_roll_mu=a.palm_roll_mu,
    )
    rep = convert(**kwargs)
    payload = json.dumps(rep.as_dict(), ensure_ascii=False, indent=2)
    if a.report:
        open(a.report, "w").write(payload)
    print(payload)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
