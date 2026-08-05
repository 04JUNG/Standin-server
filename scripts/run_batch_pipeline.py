"""
in/ 폴더의 러프 전체를 실제 파이프라인(Gemini VLM + RTMPose + 검색)으로 돌려
인물별 Top-K 후보를 out/ 폴더에 렌더한다(정성평가용 일회성 배치 스크립트).

실행:
    python scripts/run_batch_pipeline.py
    python scripts/run_batch_pipeline.py --in-dir in --out-dir out --topk 5
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.repo import load_entries
from src.pipeline import Pipeline
from src.bvh import load_coco17
from src.library import VIRTUAL_CAMERAS
from src.config import CFG
from src.refine import refine_bvh

VIEW_ANGLE = {v.value: a for v, a in VIRTUAL_CAMERAS.items()}
COCO_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
              (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def project(kp, angle):
    c, s = math.cos(angle), math.sin(angle)
    out = np.zeros((17, 2))
    out[:, 0] = c * kp[:, 0] + s * kp[:, 2]
    out[:, 1] = kp[:, 1]
    return out


def draw(ax, kp2d, scores, title, color="#333"):
    vis = scores > 0
    if not vis.any():
        ax.axis("off")
        ax.set_title(title, fontsize=8)
        return
    for a, b in COCO_EDGES:
        if vis[a] and vis[b]:
            ax.plot([kp2d[a, 0], kp2d[b, 0]], [kp2d[a, 1], kp2d[b, 1]], "-", lw=2, color=color)
    ax.scatter(kp2d[vis, 0], kp2d[vis, 1], s=14, color="#e04030", zorder=3)
    xs, ys = kp2d[vis, 0], kp2d[vis, 1]
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    r = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + 1e-6
    ax.set_xlim(cx - r * 1.2, cx + r * 1.2)
    ax.set_ylim(cy - r * 1.2, cy + r * 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="in")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--db", default="data/poses.db")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="0=전체")
    ap.add_argument("--only", default=None, help="쉼표구분 파일명 부분일치 필터")
    ap.add_argument("--sleep", type=float, default=0.0, help="컷 사이 대기(초, VLM 무료티어 rate limit 회피)")
    ap.add_argument("--no-refine", dest="refine", action="store_false",
                    help="후보에 refine 적용 끄기(기본은 켜짐)")
    a = ap.parse_args()

    print(f"[config] vlm_provider={CFG.vlm_provider} pose_backend={CFG.pose_backend} "
          f"refine={a.refine}(enabled={CFG.refine_enabled})")

    entries = load_entries(a.db)
    pipeline = Pipeline(entries)

    os.makedirs(a.out_dir, exist_ok=True)
    refine_dir = os.path.join(a.out_dir, "refined_bvh")
    if a.refine:
        os.makedirs(refine_dir, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg"}
    files = sorted(f for f in os.listdir(a.in_dir) if os.path.splitext(f)[1].lower() in exts)
    if a.only:
        needles = [s.strip() for s in a.only.split(",") if s.strip()]
        files = [f for f in files if any(n in f for n in needles)]
    if a.limit:
        files = files[:a.limit]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    for idx, fname in enumerate(files, 1):
        if a.sleep and idx > 1:
            time.sleep(a.sleep)
        path = os.path.join(a.in_dir, fname)
        img = Image.open(path).convert("RGB")
        w, h = img.size
        base = os.path.splitext(fname)[0]
        # 라벨은 ASCII로: 한글 폰트가 없는 환경에서 두부(tofu)가 되면 정성평가용으로 쓸모없다.
        label = f"cut{idx:02d}"
        print(f"\n=== [{label}] {fname} ({w}x{h}) ===")

        try:
            result = pipeline.process_cut(img, w, h)
        except Exception as e:
            print(f"[error] {fname}: {e}")
            continue

        print(f"route={result.route} count_conf={result.count_confidence} "
              f"det={result.detector_count} vlm={result.vlm_count}")
        for note in result.notes:
            print(f"  note: {note}")

        if result.route != "core":
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img)
            ax.set_title(f"{label}\nroute={result.route}", fontsize=9)
            ax.axis("off")
            out_path = os.path.join(a.out_dir, f"{base}.png")
            fig.tight_layout()
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            print(f"[saved] {out_path}")
            continue

        n_people = len(result.person_candidates)
        if n_people == 0:
            print("  (no people found)")
            continue

        cols = 1 + a.topk
        fig, axes = plt.subplots(n_people, cols, figsize=(cols * 2.0, n_people * 2.4), squeeze=False)

        for pi in range(n_people):
            desc = result.descriptors[pi] if pi < len(result.descriptors) else None
            conf = result.person_confidence[pi] if pi < len(result.person_confidence) else "?"
            cands = result.person_candidates[pi][:a.topk]

            ax0 = axes[pi][0]
            if desc is not None and desc.skeleton is not None:
                qkp = desc.skeleton.keypoints.copy()
                qkp[:, 1] *= -1
                qsc = desc.skeleton.scores
                draw(ax0, qkp, qsc, f"person {pi}\nconf={conf}", color="#1560d0")
            else:
                ax0.axis("off")
                ax0.set_title(f"person {pi}\n(no skeleton)", fontsize=8)

            refine_log = []
            for ci in range(cols - 1):
                ax = axes[pi][ci + 1]
                if ci < len(cands):
                    c = cands[ci]
                    kp, sc = load_coco17(c.bvh_path)
                    tag = ""
                    if a.refine and ci == 0 and desc is not None and desc.skeleton is not None:
                        safe_pose = "".join(ch if ch.isalnum() else "_" for ch in c.pose_id)
                        out_bvh = os.path.join(
                            refine_dir, f"{label}_p{pi}_{ci+1:02d}_{safe_pose}.bvh")
                        res = refine_bvh(c.bvh_path, desc.skeleton.keypoints,
                                         desc.skeleton.scores, c.view.value,
                                         out_path=out_bvh, search_distance=c.distance)
                        if res.refined:
                            kp, sc = load_coco17(res.bvh_path)
                            tag = f"\nrefined gain={res.gain:.0%}"
                        else:
                            tag = f"\nbase({res.reason})"
                        refine_log.append(f"#{ci+1}:{res.reason}(gain={res.gain:.0%})")
                    draw(ax, project(kp, VIEW_ANGLE[c.view.value]), sc,
                         f"#{ci + 1} {c.pose_id[:14]}\n{c.view.value} d={c.distance:.3f}{tag}")
                else:
                    ax.axis("off")

            print(f"  person {pi}: conf={conf} " +
                  ", ".join(f"#{i+1} {c.pose_id}({c.view.value},d={c.distance:.3f})"
                            for i, c in enumerate(cands)))
            if refine_log:
                print(f"    refine: " + ", ".join(refine_log))

        fig.suptitle(f"{label}  route={result.route}  det={result.detector_count} vlm={result.vlm_count}",
                     fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out_path = os.path.join(a.out_dir, f"{base}.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
