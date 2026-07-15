"""
BVH 폴더 → COCO-17 스틱피겨 컨택트 시트(PNG). 검수용.

검색이 실제로 쓰는 것과 동일하게 그린다:
  · load_coco17로 17개 관절만 추출(손가락 등 여분 본은 안 그림)
  · 인덱스와 동일한 가상 카메라 각도(VIRTUAL_CAMERAS)로 투영
  · score=0(BVH에 없는 결측) 관절은 표시 안 함

실행:
    # 4뷰 감사(축·부호 확인): 포즈당 front/3-4/side/back 한 행에
    python scripts/bvh_contact_sheet.py <bvh_dir> audit.png --views front,three_quarter,side,back --limit 8
    # 단일뷰 전체 격자(keeper 고르기)
    python scripts/bvh_contact_sheet.py <bvh_dir> grid.png --view front --cols 8
"""
import sys, os, glob, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.bvh import load_coco17
from src.library import VIRTUAL_CAMERAS

VIEW_ANGLE = {v.value: a for v, a in VIRTUAL_CAMERAS.items()}

# COCO-17 뼈대 연결(얼굴·팔·몸통·다리). 근사된 얼굴점은 head에 뭉쳐 사실상 점 하나로 보임.
COCO_EDGES = [
    (0,1),(0,2),(1,3),(2,4),                 # 얼굴
    (5,6),                                    # 어깨선
    (5,7),(7,9),(6,8),(8,10),                 # 팔
    (5,11),(6,12),(11,12),                    # 몸통
    (11,13),(13,15),(12,14),(14,16),          # 다리
]


def _project(kp, angle):
    """kp(17,3) → (17,2) : Y축 angle 회전 후 (x,y). 인덱스 투영과 동일 기하. y 위로."""
    c, s = math.cos(angle), math.sin(angle)
    out = np.zeros((17, 2))
    out[:, 0] = c * kp[:, 0] + s * kp[:, 2]
    out[:, 1] = kp[:, 1]
    return out


def draw_pose(ax, kp2d, scores, title):
    vis = scores > 0
    for a, b in COCO_EDGES:
        if vis[a] and vis[b]:
            ax.plot([kp2d[a,0], kp2d[b,0]], [kp2d[a,1], kp2d[b,1]], "-", lw=2, color="#333")
    ax.scatter(kp2d[vis,0], kp2d[vis,1], s=14, color="#e04030", zorder=3)
    xs, ys = kp2d[vis,0], kp2d[vis,1]
    cx, cy = (xs.min()+xs.max())/2, (ys.min()+ys.max())/2
    r = max(xs.max()-xs.min(), ys.max()-ys.min())/2 + 1e-6
    ax.set_xlim(cx-r*1.15, cx+r*1.15); ax.set_ylim(cy-r*1.15, cy+r*1.15)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=7)


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = sys.argv[1:]
    bvh_dir = args[0]
    out = args[1] if len(args) > 1 and not args[1].startswith("--") else "contact_sheet.png"
    frame_i = int(args[args.index("--frame")+1]) if "--frame" in args else 0
    limit = int(args[args.index("--limit")+1]) if "--limit" in args else 0
    if "--views" in args:
        views = args[args.index("--views")+1].split(",")
    else:
        views = [args[args.index("--view")+1]] if "--view" in args else ["front"]
    for v in views:
        if v not in VIEW_ANGLE:
            print("unknown view:", v, "-> 사용가능:", list(VIEW_ANGLE)); return

    files = sorted(glob.glob(os.path.join(bvh_dir, "*.bvh")))
    if limit > 0:
        files = files[:limit]
    if not files:
        print("no .bvh in", bvh_dir); return

    def kp_of(f):
        kp, sc = load_coco17(f, frame_i)
        return kp, sc

    multi = len(views) > 1
    if multi:
        rows, cols = len(files), len(views)
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2.0, rows*2.0), squeeze=False)
        for r, f in enumerate(files):
            try:
                kp, sc = kp_of(f)
                for cix, v in enumerate(views):
                    lbl = os.path.basename(f)[:16] if cix == 0 else v
                    draw_pose(axes[r][cix], _project(kp, VIEW_ANGLE[v]), sc, f"{lbl}\n[{v}]")
            except Exception as e:
                axes[r][0].set_title(f"ERR {os.path.basename(f)[:12]}", fontsize=6, color="red")
        fig.suptitle(f"{bvh_dir}  COCO17 projection audit ({len(files)} poses x {views})", fontsize=9)
    else:
        cols = int(args[args.index("--cols")+1]) if "--cols" in args else 6
        v = views[0]
        rows = (len(files) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols*1.8, rows*2.0))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes: ax.axis("off")
        for k, f in enumerate(files):
            try:
                kp, sc = kp_of(f)
                draw_pose(axes[k], _project(kp, VIEW_ANGLE[v]), sc, os.path.basename(f)[:22])
            except Exception:
                axes[k].set_title(f"ERR {os.path.basename(f)[:14]}", fontsize=6, color="red")
        fig.suptitle(f"{bvh_dir}  COCO17 ({len(files)} poses, view={v})", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out, dpi=110)
    print(f"saved {out}  ({len(files)} poses, COCO17, views={views})")


if __name__ == "__main__":
    main()
