"""
refine 3D 건전성 진단 — **매칭된 view 말고 다른 각도에서도 멀쩡한가**를 본다.

왜 필요한가(중요):
  refine의 손실은 '매칭된 view로 투영한 2D 뼈 방향'만 본다. 2D 방향은 3D 자세의
  일부만 구속하므로, **투영에서 안 보이는 방향으로는 손실이 거의 안 변한다.**
  그 방향으로 최적화가 크게 움직여도 손실은 좋아진 것처럼 보인다.
  → 정면에선 러프와 잘 맞는데 옆에서 보면 팔다리가 앞뒤로 튀어나온 포즈가 나온다.
  → CSP는 3D다. 작가가 인형을 돌리는 순간 드러난다.

  기존 eval_refine / eval_refine_batch는 **매칭된 view 한 각도만** 그렸기 때문에
  이 실패를 구조적으로 볼 수 없었다. 이 스크립트가 그 구멍을 메운다.

특히 다리:
  선 자세의 대퇴는 거의 수직이라 투영 방향이 회전에 둔감하고, 웹툰 컷은 허벅지
  아래가 잘린 경우가 많아 타깃 자체가 부실하다. 실측(12쌍 무작위): 다리는 팔보다
  손실 감도가 **3.4배 낮다** = 같은 3D 이동량에 손실이 1/3.4밖에 안 변한다.
  → 다리가 먼저 이상해지는 것은 우연이 아니라 구조적이다.

실행:
    py -3.12 scripts/diag_refine_3d.py --base "data/bvh/A.bvh" --target-bvh "data/bvh/B.bvh"
    py -3.12 scripts/diag_refine_3d.py --from-keypoints out/컷_top5refine/query.json \
        --base "data/bvh/A.bvh" --view three_quarter --out diag.png

읽는 법:
    · 위 줄(base)과 아래 줄(refined)을 **4개 view 전부** 비교한다.
    · 매칭 view [*]에서만 좋아지고 나머지 view에서 이상해졌다면 = 깊이 자유도 문제.
    · 표의 '효율'이 낮은 관절 = 많이 움직였는데 손실은 별로 안 좋아진 관절 = 범인.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from src.bvh import load_coco17, parse_bvh                          # noqa: E402
from src.config import CFG                                          # noqa: E402
from src.features import _BONES                                     # noqa: E402
from src.library import (VIRTUAL_CAMERAS, project_3d_to_2d,         # noqa: E402
                         view_angle)
from src.refine import (LIMB_MASK, _Forward, _angle_loss,           # noqa: E402
                        _bend_degrees, _torso_length, refine_bvh,
                        target_bone_dirs)
from eval_search import draw                                        # noqa: E402

VIEWS = [v.value for v in VIRTUAL_CAMERAS]

# COCO17 관절 이름(진단 표에 쓰는 것만)
WATCH = {7: "왼팔꿈치", 9: "왼손목", 8: "오른팔꿈치", 10: "오른손목",
         13: "왼무릎", 15: "왼발목", 14: "오른무릎", 16: "오른발목"}

# 뼈 → 그 뼈를 움직이는 사지 그룹
LIMB_OF = {(5, 7): "왼팔", (7, 9): "왼팔", (6, 8): "오른팔", (8, 10): "오른팔",
           (11, 13): "왼다리", (13, 15): "왼다리",
           (12, 14): "오른다리", (14, 16): "오른다리"}


def target_from_bvh(path, view):
    kp3, sc = load_coco17(path)
    kp2 = project_3d_to_2d(kp3, view_angle(view)).copy()
    kp2[:, 1] *= -1
    return kp2.astype(np.float32), sc


def target_from_json(path):
    with open(path, encoding="utf-8") as f:
        q = json.load(f)
    q = q.get("query", q)
    return (np.asarray(q["keypoints"], dtype=np.float32).reshape(17, 2),
            np.asarray(q["scores"], dtype=np.float32).reshape(17))


def render(base_bvh, ref_bvh, matched_view, out, collisions=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    def proj(bvh, v):
        kp3, sc = load_coco17(bvh)
        kp2 = project_3d_to_2d(kp3, view_angle(v)).copy()
        kp2[:, 1] *= -1
        return kp2, sc

    final_kp3, _ = load_coco17(ref_bvh)
    shoulder = (final_kp3[5] + final_kp3[6]) * 0.5
    hip = (final_kp3[11] + final_kp3[12]) * 0.5
    torso = float(np.linalg.norm(shoulder - hip))
    shoulder_width = float(np.linalg.norm(final_kp3[5] - final_kp3[6]))
    hip_width = float(np.linalg.norm(final_kp3[11] - final_kp3[12]))
    r_shoulder = float(np.clip(
        CFG.refine_collision_torso_shoulder_scale * shoulder_width,
        0.16 * torso, 0.24 * torso,
    ))
    r_hip = float(np.clip(
        CFG.refine_collision_torso_hip_scale * hip_width,
        0.14 * torso, 0.20 * torso,
    ))

    fig, axes = plt.subplots(2, len(VIEWS), figsize=(3 * len(VIEWS), 6.4))
    for c, v in enumerate(VIEWS):
        mark = " [*matched]" if v == matched_view else ""
        draw(axes[0][c], *proj(base_bvh, v), f"base — {v}{mark}")
        draw(axes[1][c], *proj(ref_bvh, v), f"refined — {v}{mark}", color="#1a6fb5")
        for r in (0, 1):
            axes[r][c].invert_yaxis()

        # P3 몸통 내부 코어(보라 원)와 full P1에서 폐기한 최대 관통점(빨간 X).
        # 최종 자세에는 충돌 팔이 이미 원복됐으므로 X는 '현재 점'이 아니라
        # 게이트가 거절한 P1 지점을 보여주는 진단 오버레이다.
        if torso > 1e-6:
            for t in np.linspace(0.0, 1.0, 5):
                center3 = shoulder + t * (hip - shoulder)
                center2 = project_3d_to_2d(
                    np.asarray([center3]), view_angle(v)
                )[0]
                center2[1] *= -1
                radius = (1.0 - t) * r_shoulder + t * r_hip
                axes[1][c].add_patch(Circle(
                    center2, radius, fill=False, lw=0.8,
                    edgecolor="#7b3fb4", alpha=0.38, zorder=1,
                ))
        for collision in (collisions or {}).values():
            point = collision.get("collision_point")
            if collision.get("status") != "new_penetration" or point is None:
                continue
            point2 = project_3d_to_2d(
                np.asarray([point], dtype=np.float64), view_angle(v)
            )[0]
            point2[1] *= -1
            axes[1][c].scatter(
                [point2[0]], [point2[1]], marker="x", s=65, linewidths=2,
                color="#d00040", zorder=5,
            )
    fig.suptitle("위=base  아래=final   보라=몸통 코어 · 빨간 X=폐기된 P1 최대 관통점",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=115)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="베이스 BVH")
    t = ap.add_mutually_exclusive_group(required=True)
    t.add_argument("--target-bvh", help="타깃을 BVH로(모델 불필요)")
    t.add_argument("--from-keypoints", help="러프 스켈레톤 json(refine_top5의 query.json)")
    ap.add_argument("--view", default="front", choices=VIEWS, help="매칭 view")
    ap.add_argument("--out", default="diag_refine.png")
    a = ap.parse_args()

    if a.target_bvh:
        tgt_kp, tgt_sc = target_from_bvh(a.target_bvh, a.view)
    else:
        tgt_kp, tgt_sc = target_from_json(a.from_keypoints)

    res = refine_bvh(a.base, tgt_kp, tgt_sc, a.view,
                     out_path=os.path.splitext(a.out)[0] + ".bvh")
    print(f"[refine] refined={res.refined}  reason={res.reason}  "
          f"loss {res.loss_base:.4f} → {res.loss_final:.4f}  (gain {res.gain:.1%})\n")
    if res.axis_lambda_mult:
        print("P1a 축별 관측 감도 / lambda 강화 배수:")
        for label, mult in res.axis_lambda_mult.items():
            sens = res.axis_observability.get(label, float("nan"))
            print(f"   {label:24} sensitivity={sens:9.6f}  lambda={mult:6.1f}x")
        print()
    if res.svd_lambda_mult:
        print("P1b SVD 조합별 특이값 / lambda 강화 배수:")
        for i, (singular, mult) in enumerate(
                zip(res.svd_singular_values, res.svd_lambda_mult), 1):
            print(f"   mode_{i:02d} singular={singular:10.7f}  lambda={mult:5.2f}x")
        print()
    collisions = {
        limb: decision.get("collision")
        for limb, decision in res.limb_decisions.items()
        if decision.get("collision") is not None
    }
    if collisions:
        print("P3 팔-몸통 충돌 진단 (깊이=몸통 길이 기준):")
        for limb, collision in collisions.items():
            base = collision.get("base_depth")
            solved = collision.get("solved_depth")
            final = collision.get("final_depth")
            part = collision.get("part") or "-"
            print(f"   {limb:10} status={collision.get('status', 'unavailable'):16} "
                  f"part={part:7}  base={base!s:>7}  "
                  f"solved={solved!s:>7}  final={final!s:>7}")
        print()
    if not res.refined:
        print("조정이 폐기됨 → 진단할 3D 변화가 없습니다.")
        return

    # ---- 타깃 뼈 유효성: 애초에 '보이는' 사지가 무엇인가 ----
    td, tok = target_bone_dirs(tgt_kp, tgt_sc)
    limb_valid = {}
    for i, b in enumerate(_BONES):
        name = LIMB_OF.get(tuple(b))
        if name:
            limb_valid.setdefault(name, []).append(bool(tok[i]))
    print("러프에서 '보이는' 뼈 (타깃이 없는 사지는 refine이 손대면 안 된다):")
    for k, v in limb_valid.items():
        n = sum(v)
        flag = "" if n == len(v) else ("   ← 일부만 보임(위험)" if n else "   ← 전혀 안 보임")
        print(f"   {k:6} {n}/{len(v)} 뼈 유효{flag}")

    # ---- 3D 이동량 vs 손실 개선 ----
    j, d = parse_bvh(a.base)
    fwd = _Forward(j, d[0], a.view)
    p0 = fwd.base_frame[fwd.param_idx]
    kb, _ = fwd.joints3d(p0)
    kn, _ = load_coco17(res.bvh_path), None
    kn = kn[0]
    scale = _torso_length(kb)
    gain_abs = res.loss_base - res.loss_final

    print(f"\n3D 이동량 (몸통 길이 기준. 0.3 넘으면 '미세조정'이 아니다):")
    print(f"{'관절':10} {'이동량':>8}")
    moves = {}
    for idx, name in WATCH.items():
        mv = float(np.linalg.norm(kn[idx] - kb[idx]) / scale)
        moves[name] = mv
        bar = "#" * int(min(mv, 1.0) * 30)
        print(f"{name:10} {mv:8.3f}  {bar}")

    total_move = float(np.mean(list(moves.values())))
    print(f"\n손실 개선 {gain_abs:.4f}   평균 3D 이동 {total_move:.3f}"
          f"   효율 {gain_abs / max(total_move, 1e-9):.3f}")
    print("   효율이 낮다 = 많이 움직였는데 별로 안 좋아졌다 = 손실이 못 보는 방향으로 갔다")

    # ---- 해부학 ----
    bb, bn = _bend_degrees(kb), _bend_degrees(kn)
    print("\n팔꿈치·무릎 굽힘각(180°=편 상태):")
    for k in bn:
        d_ = bn[k] - bb.get(k, bn[k])
        print(f"   {k:12} {bb.get(k, float('nan')):6.1f}° → {bn[k]:6.1f}°  ({d_:+.1f}°)")

    try:
        render(a.base, res.bvh_path, a.view, a.out, collisions)
        print(f"\n[saved] {a.out}  ← 4개 view를 전부 눈으로 확인할 것")
    except ImportError:
        print("\n[skip] matplotlib 없음")


if __name__ == "__main__":
    main()
