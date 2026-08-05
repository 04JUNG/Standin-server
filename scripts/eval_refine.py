"""
refine 정성평가 — 베이스 · 조정 · 러프를 한 장에 나란히 렌더한다.

각도 손실이 줄었다는 '숫자'와 사람 눈에 '러프에 가까워졌다'는 다를 수 있다.
숫자만 보고 refine을 신뢰하지 말고 반드시 이 그림을 확인할 것(REFINE_DESIGN.md §6).

실행:
    # 실제: 러프 PNG를 타깃으로 (rtmlib 필요)
    py -3.12 scripts/eval_refine.py --base "data/bvh/Sitting Idle_01.bvh" \
        --image rough.png --view front --out refine.png

    # 플러밍/회귀: 다른 라이브러리 BVH를 타깃으로 (모델 불필요)
    py -3.12 scripts/eval_refine.py --base "data/bvh/A.bvh" --target-bvh "data/bvh/B.bvh"

    # 검색 Top-1을 베이스로 자동 선택 (--db 필요)
    py -3.12 scripts/eval_refine.py --image rough.png --db data/poses.db --auto-base

판정 기준:
  · 검색이 맞은 컷    → 조정본이 러프에 더 가까워야 한다.
  · 검색이 틀린 컷    → 게이트가 걸려 'refined=False, 베이스 그대로'가 나와야 한다.
  · 어떤 경우에도     → 해부학적으로 깨진 포즈가 나오면 안 된다.
"""
import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from src.bvh import load_coco17                                    # noqa: E402
from src.features import angle_distance                            # noqa: E402
from src.library import pose_to_feature, project_3d_to_2d, view_angle  # noqa: E402
from src.refine import refine_bvh, LIMB_MASK                       # noqa: E402
from eval_search import draw                                       # noqa: E402


def target_from_bvh(path, view):
    """라이브러리 BVH를 타깃으로(모델 없이 회귀 확인). 이미지 좌표(y-down)로 맞춘다."""
    kp3, sc = load_coco17(path)
    kp2 = project_3d_to_2d(kp3, view_angle(view)).copy()
    kp2[:, 1] *= -1
    return kp2.astype(np.float32), sc


def target_from_image(path):
    from eval_search import query_from_rtmpose
    return query_from_rtmpose(path)


def base_from_search(db_path, feat, view_hint=None):
    """검색 Top-1을 베이스로. 실사용과 같은 경로(검색 → 선택 → refine)를 재현한다."""
    from src.repo import load_entries
    from src.search import _dist
    entries = load_entries(db_path)
    best = min(entries, key=lambda e: _dist(feat, e.feature))
    return best.bvh_path, best.view.value, _dist(feat, best.feature)


def render(base_bvh, refined_bvh, tgt_kp, tgt_sc, view, res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def proj(bvh):
        kp3, sc = load_coco17(bvh)
        kp2 = project_3d_to_2d(kp3, view_angle(view)).copy()
        kp2[:, 1] *= -1
        return kp2, sc

    base_kp, base_sc = proj(base_bvh)
    ref_kp, ref_sc = proj(refined_bvh)

    # 패널 라벨은 ASCII로 둔다 — 한글 폰트가 없는 환경에서 두부(□)가 되면
    # 정성평가 도구로서 쓸모가 없어진다.
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    draw(axes[0], base_kp, base_sc, f"1. base (search)\nloss={res.loss_base:.4f}")
    draw(axes[1], ref_kp, ref_sc,
         f"2. refined [{'OK' if res.refined else 'SKIPPED: ' + res.reason}]\n"
         f"loss={res.loss_final:.4f}  gain={res.gain:.1%}", color="#1a6fb5")
    draw(axes[2], tgt_kp, tgt_sc, "3. target (rough)", color="#c03020")
    # 이미지 좌표계(y 아래로 +)에 맞춰 뒤집어 보여준다
    for ax in axes:
        ax.invert_yaxis()
    fig.suptitle(f"view={view}   backend={res.backend}   nfev={res.iterations}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"[saved] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="베이스 BVH(작가가 고른 후보)")
    ap.add_argument("--image", help="러프 PNG(타깃)")
    ap.add_argument("--target-bvh", help="타깃을 BVH로 대체(모델 불필요, 회귀용)")
    ap.add_argument("--db", default="data/poses.db")
    ap.add_argument("--auto-base", action="store_true",
                    help="검색 Top-1을 베이스로 자동 선택")
    ap.add_argument("--view", default="front",
                    choices=["front", "three_quarter", "side", "back"])
    ap.add_argument("--out", default="refine_eval.png")
    ap.add_argument("--no-gate-distance", action="store_true",
                    help="베이스 불일치 게이트를 끄고 강제로 조정(진단용)")
    a = ap.parse_args()

    if not a.image and not a.target_bvh:
        ap.error("--image 또는 --target-bvh 중 하나는 필요합니다")

    # --- 타깃(러프) ---
    if a.image:
        tgt_kp, tgt_sc = target_from_image(a.image)
    else:
        tgt_kp, tgt_sc = target_from_bvh(a.target_bvh, a.view)

    # --- 베이스 ---
    dist = None
    if a.auto_base:
        from src.features import normalize_skeleton
        feat = normalize_skeleton(tgt_kp, tgt_sc)
        base, view, dist = base_from_search(a.db, feat)
        print(f"[search] Top-1 = {base}  view={view}  distance={dist:.4f}")
    else:
        if not a.base:
            ap.error("--base 또는 --auto-base가 필요합니다")
        base, view = a.base, a.view

    out_bvh = os.path.splitext(a.out)[0] + ".bvh"
    t0 = time.time()
    res = refine_bvh(base, tgt_kp, tgt_sc, view, out_path=out_bvh,
                     search_distance=None if a.no_gate_distance else dist)
    elapsed = time.time() - t0

    print(f"[refine] refined={res.refined}  reason={res.reason}  "
          f"backend={res.backend}  nfev={res.iterations}  {elapsed:.2f}s")
    print(f"[loss]   base={res.loss_base:.4f} → final={res.loss_final:.4f}  "
          f"(gain {res.gain:.1%})   ※ 팔·다리 {int(LIMB_MASK.sum())}개 뼈 평균 (1−cos)")

    # 전체 12뼈 기준으로도 확인 — 검색 metric과 같은 척도라 비교 가능
    tgt_feat = pose_to_feature(*(load_coco17(a.target_bvh)[0], view,
                                 load_coco17(a.target_bvh)[1])) \
        if a.target_bvh else None
    if tgt_feat is not None:
        f_base = pose_to_feature(*(load_coco17(base)[0], view, load_coco17(base)[1]))
        f_ref = pose_to_feature(*(load_coco17(res.bvh_path)[0], view,
                                  load_coco17(res.bvh_path)[1]))
        print(f"[검색척도] angle_distance  base={angle_distance(f_base, tgt_feat):.4f}"
              f" → refined={angle_distance(f_ref, tgt_feat):.4f}")

    try:
        render(base, res.bvh_path, tgt_kp, tgt_sc, view, res, a.out)
    except ImportError:
        print("[skip] matplotlib 없음 → 수치만 출력")


if __name__ == "__main__":
    main()
