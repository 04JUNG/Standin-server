"""
refine 배치 정성평가 — 컷마다 **Top-K 후보 전부**를 조정해서 한 장에 담는다.

왜 배치인가:
  실사용은 "작가가 하나 고른 뒤 refine"이라 평가할 때도 하나를 골라야 하는 것처럼 보인다.
  하지만 그건 **제품의 제약이지 평가의 제약이 아니다.** 평가에서는 연산이 공짜이므로
  5개를 다 돌려놓고, 숫자로 분류한 뒤 수상한 것만 눈으로 본다.
  → 사람이 봐야 하는 양: 컷당 이미지 1장 + 플래그 붙은 행 몇 개.

실행:
    # 실 러프 폴더 (rtmlib 필요)
    py -3.12 scripts/eval_refine_batch.py --images "C:/.../conti" --db data/poses.db --outdir eval_refine

    # 모델 없이 회귀(라이브러리 BVH를 쿼리로) — 플러밍·게이트 확인용
    py -3.12 scripts/eval_refine_batch.py --bvh-queries data/bvh --limit 8 --outdir eval_refine

산출물:
    <outdir>/<컷이름>.png    러프 | (후보별) 베이스 · 조정   ← 눈으로 판정
    <outdir>/summary.csv     컷×후보 전 행의 수치와 게이트 사유
    콘솔                     집계 + '눈으로 볼 것' 트리아지 목록

판정 규칙(REFINE_DESIGN.md §6-1):
    · WORSE 는 0건이어야 한다      — refine이 결과를 나쁘게 만드는 경로는 없어야 함
    · BEND  는 0건이어야 한다      — 해부학적으로 깨진 포즈
    · 검색이 틀린 컷은 SKIP(base_mismatch)이 떠야 정상 — 게이트가 일하고 있다는 증거
"""
import argparse
import csv
import glob
import os
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


# ---- 쿼리 소스 --------------------------------------------------------------

def queries_from_images(paths):
    from eval_search import query_from_rtmpose
    for p in paths:
        try:
            kp, sc = query_from_rtmpose(p)
        except Exception as exc:                    # 추출 실패도 데이터다 — 기록하고 계속
            print(f"[skip] {os.path.basename(p)}: {exc}")
            continue
        yield os.path.splitext(os.path.basename(p))[0], kp, sc


def queries_from_bvh(paths, view="front"):
    """모델 없이 도는 회귀 모드. 라이브러리 BVH를 러프 대용으로 투영한다."""
    for p in paths:
        kp3, sc = load_coco17(p)
        kp2 = project_3d_to_2d(kp3, view_angle(view)).copy()
        kp2[:, 1] *= -1
        yield os.path.splitext(os.path.basename(p))[0], kp2.astype(np.float32), sc


# ---- 검색 -------------------------------------------------------------------

def search_topk(entries, feat, k):
    """pose_id별 최선 1개만 남긴 Top-K(=/analyze가 작가에게 보여주는 것과 동일)."""
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


# ---- 렌더 -------------------------------------------------------------------

def _proj(bvh, view):
    kp3, sc = load_coco17(bvh)
    kp2 = project_3d_to_2d(kp3, view_angle(view)).copy()
    kp2[:, 1] *= -1
    return kp2, sc


def contact_sheet(cut, tgt_kp, tgt_sc, rows, out_png):
    """왼쪽에 러프 1장, 오른쪽에 후보별 [베이스 | 조정] — 한 눈에 훑는 용도."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(rows)
    fig = plt.figure(figsize=(9, 2.4 * k + 0.6))
    gs = fig.add_gridspec(k, 3, width_ratios=[1.15, 1, 1])

    ax = fig.add_subplot(gs[:, 0])
    draw(ax, tgt_kp, tgt_sc, "rough (target)", color="#c03020")
    ax.invert_yaxis()

    for i, r in enumerate(rows):
        bk, bs = _proj(r["base_bvh"], r["view"])
        rk, rs = _proj(r["out_bvh"], r["view"])
        a0 = fig.add_subplot(gs[i, 1])
        a1 = fig.add_subplot(gs[i, 2])
        draw(a0, bk, bs, f"#{r['rank']} base  d={r['distance']:.3f}\n{r['pose_id'][:26]}")
        tag = "OK" if r["refined"] else f"SKIP: {r['reason']}"
        draw(a1, rk, rs,
             f"#{r['rank']} refined [{tag}]\n"
             f"all-bone {r['d_all_base']:.3f} -> {r['d_all_ref']:.3f}  {r['flag']}",
             color="#1a6fb5" if r["refined"] else "#888")
        a0.invert_yaxis()
        a1.invert_yaxis()

    fig.suptitle(cut, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ---- 메인 -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help="러프 PNG 폴더 또는 glob")
    ap.add_argument("--bvh-queries", help="모델 없이 회귀: 쿼리로 쓸 BVH 폴더")
    ap.add_argument("--db", default="data/poses.db")
    ap.add_argument("--topk", type=int, default=CFG.top_k_final)
    ap.add_argument("--outdir", default="eval_refine")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개만")
    ap.add_argument("--no-render", action="store_true", help="숫자만(빠름)")
    a = ap.parse_args()

    if not a.images and not a.bvh_queries:
        ap.error("--images 또는 --bvh-queries 중 하나는 필요합니다")

    os.makedirs(a.outdir, exist_ok=True)
    entries = load_entries(a.db)
    print(f"[db] {a.db} — 투영 {len(entries)}개")

    if a.images:
        pat = a.images if glob.has_magic(a.images) else os.path.join(a.images, "*.png")
        src = queries_from_images(sorted(glob.glob(pat)))
    else:
        src = queries_from_bvh(sorted(glob.glob(os.path.join(a.bvh_queries, "*.bvh"))))

    rows_all, t_start = [], time.time()
    for n, (cut, kp, sc) in enumerate(src):
        if a.limit and n >= a.limit:
            break
        feat = normalize_skeleton(kp, sc)
        hits = search_topk(entries, feat, a.topk)
        if not hits:
            print(f"[{cut}] 후보 없음")
            continue

        rows = []
        for rank, (dist, e) in enumerate(hits, 1):
            out_bvh = os.path.join(a.outdir, "bvh", f"{cut}__{rank}.bvh")
            # 게이트를 실사용 그대로 켠 채 돌린다 — 평가 대상은 '게이트 포함 refine'이다
            res = refine_bvh(e.bvh_path, kp, sc, e.view.value, out_path=out_bvh,
                             search_distance=dist)

            tgt_feat = feat
            f_base = pose_to_feature(*(load_coco17(e.bvh_path)[0], e.view.value,
                                       load_coco17(e.bvh_path)[1]))
            f_ref = pose_to_feature(*(load_coco17(res.bvh_path)[0], e.view.value,
                                      load_coco17(res.bvh_path)[1]))
            # 독립 검증: refine이 최소화한 손실(팔다리 8뼈)이 아니라
            # **검색과 같은 척도(전체 12뼈)**로도 나빠지지 않았는지 본다.
            d_all_base = angle_distance(f_base, tgt_feat)
            d_all_ref = angle_distance(f_ref, tgt_feat)

            bend = _bend_degrees(load_coco17(res.bvh_path)[0])
            bad_bend = {k: round(v, 1) for k, v in bend.items()
                        if v < CFG.refine_min_bend_deg}

            flag = ""
            if res.refined and d_all_ref > d_all_base + 1e-6:
                flag = "WORSE"
            elif bad_bend:
                flag = "BEND"
            elif not res.refined:
                flag = "SKIP"
            elif (d_all_base - d_all_ref) / max(d_all_base, 1e-9) > 0.3:
                flag = "BIG"

            rows.append(dict(
                cut=cut, rank=rank, pose_id=e.pose_id, view=e.view.value,
                distance=round(float(dist), 4), refined=res.refined,
                reason=res.reason,
                loss_base=None if np.isnan(res.loss_base) else round(res.loss_base, 4),
                loss_final=None if np.isnan(res.loss_final) else round(res.loss_final, 4),
                d_all_base=round(d_all_base, 4), d_all_ref=round(d_all_ref, 4),
                bad_bend=str(bad_bend) if bad_bend else "",
                flag=flag, base_bvh=e.bvh_path, out_bvh=res.bvh_path,
            ))

        rows_all.extend(rows)
        if not a.no_render:
            try:
                contact_sheet(cut, kp, sc, rows, os.path.join(a.outdir, f"{cut}.png"))
            except ImportError:
                a.no_render = True
                print("[skip] matplotlib 없음 → 숫자만")
        flags = " ".join(f"#{r['rank']}{r['flag']}" for r in rows if r["flag"])
        print(f"[{cut}] Top-{len(rows)}  {flags or 'clean'}")

    # ---- 집계 ----
    csv_path = os.path.join(a.outdir, "summary.csv")
    if rows_all:
        keys = [k for k in rows_all[0] if k not in ("base_bvh", "out_bvh")]
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows_all)

    n = len(rows_all)
    ref = [r for r in rows_all if r["refined"]]
    worse = [r for r in rows_all if r["flag"] == "WORSE"]
    bend = [r for r in rows_all if r["flag"] == "BEND"]
    gains = [(r["d_all_base"] - r["d_all_ref"]) / max(r["d_all_base"], 1e-9) for r in ref]

    print(f"\n{'='*64}\n총 {n}행 ({len(set(r['cut'] for r in rows_all))}컷 × Top-K)  "
          f"{time.time()-t_start:.1f}s")
    print(f"  조정됨      {len(ref)}/{n}")
    print(f"  검색척도 악화 {len(worse)}   ← 0이어야 정상")
    print(f"  굽힘각 위반  {len(bend)}   ← 0이어야 정상")
    if gains:
        print(f"  개선율(전체 12뼈)  median {np.median(gains):+.1%}  "
              f"min {min(gains):+.1%}  max {max(gains):+.1%}")
    hist = {}
    for r in rows_all:
        hist[r["reason"]] = hist.get(r["reason"], 0) + 1
    print("  게이트 사유:", hist)

    triage = worse + bend + [r for r in rows_all if r["flag"] == "SKIP"][:5]
    if triage:
        print("\n눈으로 볼 것(먼저 이것부터):")
        for r in triage[:12]:
            print(f"  {a.outdir}/{r['cut']}.png  #{r['rank']} {r['flag']:5s} "
                  f"{r['reason']:22s} {r['pose_id'][:32]}")
    else:
        print("\n플래그 없음 — 아무 컷이나 2~3장 표본으로 눈 확인하면 충분.")
    print(f"\n표: {csv_path}")


if __name__ == "__main__":
    main()
