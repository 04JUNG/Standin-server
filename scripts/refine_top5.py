"""
러프 1장 → Top-5 썸네일 + **Top-5 전부의 조정된 BVH**를 한 번에 뽑는다.

왜 있는가:
  실사용 워크플로는 "고른 뒤 refine"이라, 고르기 전에는 뭐가 나올지 알 수 없다.
  검증할 때는 그게 병목이다 — 하나 골라야 다음으로 못 넘어가는데 5개를 다 눌러볼 순 없다.
  이 도구는 5개를 다 돌려놓고 **썸네일로 한 번에 보여준다.** 고르는 건 그 다음이다.

  ⚠ 서버(`POST /refine`)의 계약은 그대로다 — 고른 1개만 조정한다(연산 최소).
    이건 검증·시연 준비용 오프라인 도구다. 제품에 올릴지는 별개 결정.

실행:
    # 실 러프 (rtmlib 필요)
    py -3.12 scripts/refine_top5.py --image rough.png --db data/poses.db --out out/cut01

    # 모델 없이 (라이브러리 BVH를 러프 대용으로)
    py -3.12 scripts/refine_top5.py --from-bvh "data/bvh/X.bvh" --db data/poses.db --out out/x

    # 한 번 뽑아둔 스켈레톤 재사용 — RTMPose 재실행 없음(임계값 튜닝 반복에 쓴다)
    py -3.12 scripts/refine_top5.py --from-keypoints out/x/query.json --db data/poses.db --out out/x2

산출물 (--out 폴더):
    sheet.png            러프 + Top-5 [베이스 | 조정] 한 장     ← 여기서 고른다
    01_<pose_id>.png     후보별 썸네일(조정 결과)
    01_<pose_id>.bvh     후보별 BVH — **5개 전부**. 게이트에 걸린 건 베이스 원본 복사
    manifest.json        랭크·pose_id·view·거리·게이트 사유·파일명
    query.json           추출된 러프 스켈레톤(17×2 + scores) — 재실행 캐시

manifest의 `refined:false`는 오류가 아니다 — 안전 게이트가 조정을 버린 것이고
그 경우 .bvh는 베이스 원본이다("좋아지거나, 그대로", REFINE_DESIGN.md §4-3).
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from src.bvh import load_coco17                                     # noqa: E402
from src.config import CFG                                          # noqa: E402
from src.features import angle_distance, normalize_skeleton         # noqa: E402
from src.library import pose_to_feature, project_3d_to_2d, view_angle   # noqa: E402
from src.refine import refine_bvh, _bend_degrees                    # noqa: E402
from src.repo import load_entries                                   # noqa: E402
from src.search import _dist                                        # noqa: E402
from eval_search import draw                                        # noqa: E402


def _safe(name: str) -> str:
    """파일명으로 쓸 수 있게. 한글은 살리고 경로 문자만 턴다."""
    return "".join(c if c not in '\\/:*?"<>|' else "_" for c in name).strip()


def _proj(bvh, view):
    kp3, sc = load_coco17(bvh)
    kp2 = project_3d_to_2d(kp3, view_angle(view)).copy()
    kp2[:, 1] *= -1
    return kp2, sc


def load_query(args):
    if args.image:
        from eval_search import query_from_rtmpose
        kp, sc = query_from_rtmpose(args.image)
        return os.path.splitext(os.path.basename(args.image))[0], kp, sc
    if args.from_keypoints:
        # 한 번 뽑아둔 스켈레톤 재사용. RTMPose(176MB 모델)를 다시 돌리지 않는다 —
        # 임계값·metric을 바꿔가며 반복할 때 추출이 매번 병목이 되는 걸 막는다.
        with open(args.from_keypoints, encoding="utf-8") as f:
            q = json.load(f)
        q = q.get("query", q)          # manifest.json도 그대로 받는다
        return (q.get("cut", "query"),
                np.asarray(q["keypoints"], dtype=np.float32).reshape(17, 2),
                np.asarray(q["scores"], dtype=np.float32).reshape(17))
    kp3, sc = load_coco17(args.from_bvh)
    kp2 = project_3d_to_2d(kp3, view_angle(args.view)).copy()
    kp2[:, 1] *= -1
    return (os.path.splitext(os.path.basename(args.from_bvh))[0],
            kp2.astype(np.float32), sc)


def search_topk(entries, feat, k):
    """pose_id별 최선 1개만 남긴 Top-K — /analyze가 작가에게 보여주는 것과 동일."""
    scored = sorted(((_dist(feat, e.feature), e) for e in entries), key=lambda t: t[0])
    seen, out = set(), []
    for d, e in scored:
        if e.pose_id in seen or not e.bvh_path:
            continue
        seen.add(e.pose_id)
        out.append((d, e))
        if len(out) >= k:
            break
    return out


def save_thumb(path, kp2, sc, title, color):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2.6, 3.2))
    draw(ax, kp2, sc, title, color=color)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=110, transparent=False)
    plt.close(fig)


def save_sheet(path, cut, tgt_kp, tgt_sc, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = len(rows)
    fig = plt.figure(figsize=(9, 2.4 * k + 0.6))
    gs = fig.add_gridspec(k, 3, width_ratios=[1.15, 1, 1])
    ax = fig.add_subplot(gs[:, 0])
    draw(ax, tgt_kp, tgt_sc, "rough (input)", color="#c03020")
    ax.invert_yaxis()
    for i, r in enumerate(rows):
        a0 = fig.add_subplot(gs[i, 1])
        a1 = fig.add_subplot(gs[i, 2])
        draw(a0, *r["_base_draw"],
             f"#{r['rank']} base  d={r['distance']:.3f}\n{r['pose_id'][:26]}")
        tag = "refined" if r["refined"] else f"base ({r['reason']})"
        draw(a1, *r["_ref_draw"],
             f"#{r['rank']} {tag}\nall-bone {r['fit_base']:.3f} -> {r['fit_final']:.3f}",
             color="#1a6fb5" if r["refined"] else "#999")
        a0.invert_yaxis()
        a1.invert_yaxis()
    fig.suptitle(f"{cut}   —   고를 것 하나만 정하면 됩니다(BVH는 5개 다 있음)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="러프 PNG 1장")
    src.add_argument("--from-bvh", help="모델 없이: 라이브러리 BVH를 러프 대용으로")
    src.add_argument("--from-keypoints",
                     help="이전 실행의 query.json(또는 manifest.json) 재사용 — RTMPose 안 돌림")
    ap.add_argument("--db", default="data/poses.db")
    ap.add_argument("--out", default="top5_out")
    ap.add_argument("--topk", type=int, default=CFG.top_k_final)
    ap.add_argument("--view", default="front",
                    choices=["front", "three_quarter", "side", "back"],
                    help="--from-bvh일 때 러프를 만들 투영 각도")
    ap.add_argument("--no-gate-distance", action="store_true",
                    help="베이스 불일치 게이트를 끄고 강제로 5개 다 조정(진단용)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    entries = load_entries(a.db)
    cut, kp, sc = load_query(a)
    feat = normalize_skeleton(kp, sc)

    # 추출 결과를 먼저 떨궈 둔다. RTMPose는 176MB 모델이라 재실행이 비싸고,
    # 임계값·metric을 바꿔가며 반복할 때 매번 다시 뽑을 이유가 없다.
    query = {"cut": cut, "source": a.image or a.from_bvh or a.from_keypoints,
             "keypoints": np.asarray(kp, dtype=float).tolist(),
             "scores": np.asarray(sc, dtype=float).tolist()}
    with open(os.path.join(a.out, "query.json"), "w", encoding="utf-8") as f:
        json.dump(query, f, ensure_ascii=False, indent=2)
    hits = search_topk(entries, feat, a.topk)
    if not hits:
        print("후보 없음 — 라이브러리를 확인하세요")
        return

    t0 = time.time()
    rows = []
    for rank, (dist, e) in enumerate(hits, 1):
        stem = f"{rank:02d}_{_safe(e.pose_id)}"
        work = os.path.join(a.out, stem + ".bvh")
        res = refine_bvh(e.bvh_path, kp, sc, e.view.value, out_path=work,
                         search_distance=None if a.no_gate_distance else dist)

        # 게이트에 걸렸어도 파일은 5개 다 준다 — 작가가 고를 때 빈칸이 없어야 한다.
        if not res.refined:
            shutil.copyfile(e.bvh_path, work)

        f_base = pose_to_feature(*(load_coco17(e.bvh_path)[0], e.view.value,
                                   load_coco17(e.bvh_path)[1]))
        f_ref = pose_to_feature(*(load_coco17(work)[0], e.view.value,
                                  load_coco17(work)[1]))
        bend = _bend_degrees(load_coco17(work)[0])
        bad = {k: round(v, 1) for k, v in bend.items() if v < CFG.refine_min_bend_deg}

        row = dict(
            rank=rank, pose_id=e.pose_id, view=e.view.value,
            distance=round(float(dist), 4),
            refined=bool(res.refined), reason=res.reason,
            limbs=list(res.limbs),
            limb_decisions=res.limb_decisions or None,
            limb_observability=res.observability or None,
            axis_observability=res.axis_observability or None,
            axis_lambda_mult=res.axis_lambda_mult or None,
            svd_singular_values=list(res.svd_singular_values) or None,
            svd_lambda_mult=list(res.svd_lambda_mult) or None,
            # 검색과 같은 척도(12뼈)로 본 러프와의 거리 — refine이 최소화한 값이 아니라
            # 독립 지표다. final > base면 뭔가 잘못된 것.
            fit_base=round(angle_distance(f_base, feat), 4),
            fit_final=round(angle_distance(f_ref, feat), 4),
            bad_bend=bad or None,
            bvh=os.path.basename(work), thumb=stem + ".png",
        )
        row["_base_draw"] = _proj(e.bvh_path, e.view.value)
        row["_ref_draw"] = _proj(work, e.view.value)
        rows.append(row)

        try:
            save_thumb(os.path.join(a.out, row["thumb"]), *row["_ref_draw"],
                       f"#{rank} {e.pose_id[:22]}\n"
                       f"{'refined' if res.refined else 'base(' + res.reason + ')'}",
                       "#1a6fb5" if res.refined else "#999")
        except ImportError:
            pass

    try:
        save_sheet(os.path.join(a.out, "sheet.png"), cut, kp, sc, rows)
    except ImportError:
        print("[skip] matplotlib 없음 → 썸네일/시트 생략, BVH·manifest는 생성됨")

    for r in rows:
        r.pop("_base_draw", None)
        r.pop("_ref_draw", None)
    manifest = dict(cut=cut, source=query["source"], db=a.db,
                    distance_metric=CFG.distance_metric,
                    topk=len(rows), elapsed_sec=round(time.time() - t0, 2),
                    note="refined=false면 .bvh는 베이스 원본(안전 게이트가 조정을 폐기)",
                    query=query, candidates=rows)
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[{cut}]  Top-{len(rows)}  {manifest['elapsed_sec']}s → {a.out}/")
    print(f"{'#':>2} {'pose_id':32} {'view':14} {'dist':>6} {'refined':>8} "
          f"{'러프와 거리(12뼈)':>18}")
    for r in rows:
        arrow = f"{r['fit_base']:.3f} → {r['fit_final']:.3f}"
        warn = "  ⚠BEND" if r["bad_bend"] else ""
        if r["refined"] and r["fit_final"] > r["fit_base"] + 1e-6:
            warn += "  ⚠WORSE"
        print(f"{r['rank']:>2} {r['pose_id'][:32]:32} {r['view']:14} "
              f"{r['distance']:>6.3f} {str(r['refined']):>8} {arrow:>18}{warn}")
    print(f"\n→ {a.out}/sheet.png 를 열어 하나 고르세요. "
          f"BVH는 {len(rows)}개 모두 같은 폴더에 있습니다.")


if __name__ == "__main__":
    main()
