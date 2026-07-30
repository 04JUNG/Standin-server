"""
검색 정성평가(4순위): 러프 → 스켈레톤 → Top-K 후보를 한 장에 렌더.
"고쳐 쓸 만한가(유사도 60~80%)"를 눈으로 판정하기 위한 도구.

실행:
    # 실제: 러프 PNG → RTMPose로 스켈레톤 추출 → 검색   (rtmlib 필요)
    py -3.12 scripts/eval_search.py --image rough.png --db data/poses.db --topk 5 --out eval.png
    # 테스트: RTMPose 없이 라이브러리 BVH를 쿼리로(플러밍 확인)
    py -3.12 scripts/eval_search.py --from-bvh data/bvh/Waving_01.bvh --db data/poses.db --topk 5 --out eval.png

패널: [쿼리] | [Top-1] .. [Top-K]  (후보는 매칭된 view로, 거리 표기)
"""
import sys, os, math, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.repo import load_entries
from src.features import normalize_skeleton, pose_distance
from src.search import _dist  # DISTANCE env(pos/angle/hybrid) 존중
from src.bvh import load_coco17
from src.library import VIRTUAL_CAMERAS

VIEW_ANGLE = {v.value: a for v, a in VIRTUAL_CAMERAS.items()}
COCO_EDGES = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
              (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]


def project(kp, angle):
    c, s = math.cos(angle), math.sin(angle)
    out = np.zeros((17, 2)); out[:,0] = c*kp[:,0]+s*kp[:,2]; out[:,1] = kp[:,1]
    return out


def draw(ax, kp2d, scores, title, color="#333"):
    vis = scores > 0
    for a, b in COCO_EDGES:
        if vis[a] and vis[b]:
            ax.plot([kp2d[a,0],kp2d[b,0]],[kp2d[a,1],kp2d[b,1]],"-",lw=2,color=color)
    ax.scatter(kp2d[vis,0], kp2d[vis,1], s=14, color="#e04030", zorder=3)
    xs, ys = kp2d[vis,0], kp2d[vis,1]
    cx, cy = (xs.min()+xs.max())/2, (ys.min()+ys.max())/2
    r = max(xs.max()-xs.min(), ys.max()-ys.min())/2 + 1e-6
    ax.set_xlim(cx-r*1.2, cx+r*1.2); ax.set_ylim(cy-r*1.2, cy+r*1.2)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=8)


def query_from_rtmpose(image_path):
    """러프 PNG → RTMPose Body → 가장 큰 인물의 (kp2d(17,2), scores). 이미지 y-down."""
    from src.pose import RTMPoseModel
    import numpy as np
    model = RTMPoseModel()
    sks = model.estimate(image_path, [], 0, 0)
    if not sks:
        raise RuntimeError("RTMPose가 인물을 못 찾음: " + image_path)
    # 가장 크게 퍼진(=가까운) 인물 선택
    best = max(sks, key=lambda s: np.ptp(s.keypoints[:,1]) + np.ptp(s.keypoints[:,0]))
    return best.keypoints.astype(np.float32), best.scores.astype(np.float32)


def query_from_bvh(bvh_path):
    """테스트용: 라이브러리 BVH를 정면 투영(+y-flip=이미지좌표)해서 쿼리로."""
    kp, sc = load_coco17(bvh_path)
    kp2d = project(kp, VIEW_ANGLE["front"]).copy(); kp2d[:,1] *= -1   # y-down(이미지)
    return kp2d.astype(np.float32), sc


def search(entries, feat, topk):
    scored = sorted(((_dist(feat, e.feature), e) for e in entries),
                    key=lambda t: t[0])
    seen, out = set(), []
    for d, e in scored:
        if e.pose_id in seen: continue
        seen.add(e.pose_id); out.append((d, e))
        if len(out) >= topk: break
    return out



def _load_img(path):
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB"); return im, im.width, im.height
    except Exception:
        return path, 512, 768


def _entry_by(entries, pose_id, view):
    for e in entries:
        if e.pose_id == pose_id and e.view == view:
            return e
    for e in entries:
        if e.pose_id == pose_id:
            return e
    return None


def _render_and_print(a, qdisp, qs, qtitle, hits):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = 1 + len(hits)
    fig, axes = plt.subplots(1, n, figsize=(n*2.2, 2.6))
    draw(axes[0], qdisp, qs, qtitle, color="#1560d0")
    print(f"query: {qtitle}")
    for i, (d, e) in enumerate(hits, 1):
        kp, sc = load_coco17(e.bvh_path)
        draw(axes[i], project(kp, VIEW_ANGLE[e.view.value]), sc,
             f"#{i} {e.pose_id[:16]}\n{e.view.value} d={d:.3f}")
        print(f"  #{i} {e.pose_id:20s} view={e.view.value:13s} dist={d:.3f}")
    fig.suptitle(f"search eval — top {len(hits)} (db={a.db})", fontsize=9)
    fig.tight_layout(rect=[0,0,1,0.92]); fig.savefig(a.out, dpi=120); print("saved", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image"); ap.add_argument("--from-bvh", dest="from_bvh")
    ap.add_argument("--db", default="data/poses.db")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", default="eval.png")
    ap.add_argument("--rankall", action="store_true", help="전체 포즈 거리 순위 출력")
    ap.add_argument("--mark", default=None, help="이 문자열 포함 pose_id의 순위·거리 강조")
    ap.add_argument("--upper-weight", type=float, default=1.0, dest="uw",
                    help="상체(어깨·팔·손) 가중. >1이면 다리 희석 완화(진단용)")
    ap.add_argument("--use-vlm", action="store_true",
                    help="쿼리에 VLM 태그 부여 후 태그필터+view우선 검색(순수 kNN 대신)")
    a = ap.parse_args()

    if a.image:
        qk, qs = query_from_rtmpose(a.image); qtitle = "QUERY (rough→RTMPose)"
        qdisp = qk.copy(); qdisp[:,1] *= -1     # 화면 표시용 upright
    elif a.from_bvh:
        qk, qs = query_from_bvh(a.from_bvh); qtitle = "QUERY (bvh:%s)" % os.path.basename(a.from_bvh)
        qdisp = qk.copy(); qdisp[:,1] *= -1
    else:
        print("--image 또는 --from-bvh 필요"); return

    feat = normalize_skeleton(qk, qs)
    entries = load_entries(a.db)

    if a.use_vlm and a.image:
        # 쿼리 이미지에 VLM 태그 → 태그필터+view우선 검색(search.knn)
        from src.vlm.client import build_vlm_client
        from src.schema import PersonDescriptor, Shot
        from src import search as S
        img, w, h = _load_img(a.image)
        vlm = build_vlm_client().analyze(img, w, h)
        print(f"[vlm] shot={vlm.shot.value} action={vlm.action.value} "
              f"view={vlm.view.value} rel={vlm.relationship.value} n={vlm.num_people}")
        desc = PersonDescriptor(vlm.shot, vlm.action, vlm.view, vlm.relationship,
                                None, feat, None)
        cands = S.search(entries, desc)   # 태그 사전필터 → kNN(view 우선) → rerank
        hits = [(c.distance, _entry_by(entries, c.pose_id, c.view)) for c in cands]
        hits = [(d, e) for d, e in hits if e is not None][:a.topk]
        _render_and_print(a, qdisp, qs, qtitle, hits)
        return

    UPPER = [5, 6, 7, 8, 9, 10]; LOWER = [11, 12, 13, 14, 15, 16]
    def wdist(fa, fb, uw):
        A = fa.reshape(17, 2); B = fb.reshape(17, 2)
        du = np.linalg.norm(A[UPPER] - B[UPPER], axis=1)
        dl = np.linalg.norm(A[LOWER] - B[LOWER], axis=1)
        return float((uw * du.sum() + dl.sum()) / (uw * len(UPPER) + len(LOWER)))

    # 전체 순위(포즈별 최소거리). uw=1.0이면 기본 pose_distance와 동일.
    best = {}
    for e in entries:
        d = wdist(feat, e.feature, a.uw) if a.uw != 1.0 else _dist(feat, e.feature)
        if e.pose_id not in best or d < best[e.pose_id][0]:
            best[e.pose_id] = (d, e)
    ranked = sorted(best.values(), key=lambda t: t[0])
    hits = ranked[:a.topk]

    if a.rankall:
        print("=== 전체 순위 (pose별 최소거리) ===")
        for r, (d, e) in enumerate(ranked, 1):
            print(f"  {r:3d}. {e.pose_id:28s} {e.view.value:13s} d={d:.3f}")
    if a.mark:
        for r, (d, e) in enumerate(ranked, 1):
            if a.mark.lower() in e.pose_id.lower():
                print(f">> MARK '{e.pose_id}' -> 순위 {r}/{len(ranked)}, view={e.view.value}, d={d:.3f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = 1 + len(hits)
    fig, axes = plt.subplots(1, n, figsize=(n*2.2, 2.6))
    draw(axes[0], qdisp, qs, qtitle, color="#1560d0")
    print(f"query: {qtitle}")
    for i, (d, e) in enumerate(hits, 1):
        kp, sc = load_coco17(e.bvh_path)
        draw(axes[i], project(kp, VIEW_ANGLE[e.view.value]), sc,
             f"#{i} {e.pose_id[:16]}\n{e.view.value} d={d:.3f}")
        sim = max(0.0, 1.0 - d)   # 대략 유사도(참고)
        print(f"  #{i} {e.pose_id:20s} view={e.view.value:13s} dist={d:.3f}  ~sim={sim:.2f}")
    fig.suptitle(f"search eval — top {len(hits)} (db={a.db})", fontsize=9)
    fig.tight_layout(rect=[0,0,1,0.92]); fig.savefig(a.out, dpi=120)
    print("saved", a.out)


if __name__ == "__main__":
    main()
