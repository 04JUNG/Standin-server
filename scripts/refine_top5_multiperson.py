"""
지정한 러프들 → 전체 파이프라인(VLM+검출+검색) → 인물별 Top-5 전부 refine → BVH만 저장.

run_batch_pipeline.py는 Top-1만 refine하고 PNG 시트도 만든다. 이 스크립트는
"컷마다 폴더 하나, 그 안에 인물별 Top-5 refine된 BVH만" 요청에 맞춘 축소판이다.
이미지 렌더링은 안 한다(matplotlib 불필요, 더 빠름).

실행:
    python scripts/refine_top5_multiperson.py --only "124637,131000" --sleep 8
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.repo import load_entries
from src.pipeline import Pipeline
from src.refine import refine_bvh
from src.config import CFG


def safe(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z]", "_", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="in")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--db", default="data/poses.db")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--only", required=True, help="쉼표구분 파일명 부분일치 필터")
    ap.add_argument("--sleep", type=float, default=0.0)
    a = ap.parse_args()

    print(f"[config] vlm_provider={CFG.vlm_provider} pose_backend={CFG.pose_backend} "
          f"refine_limbs={CFG.refine_limbs}")

    entries = load_entries(a.db)
    pipeline = Pipeline(entries)

    exts = {".png", ".jpg", ".jpeg"}
    files = sorted(f for f in os.listdir(a.in_dir) if os.path.splitext(f)[1].lower() in exts)
    needles = [s.strip() for s in a.only.split(",") if s.strip()]
    files = [f for f in files if any(n in f for n in needles)]

    from PIL import Image

    for idx, fname in enumerate(files, 1):
        if a.sleep and idx > 1:
            time.sleep(a.sleep)
        path = os.path.join(a.in_dir, fname)
        img = Image.open(path).convert("RGB")
        w, h = img.size
        base = os.path.splitext(fname)[0]
        print(f"\n=== {fname} ({w}x{h}) ===")

        try:
            result = pipeline.process_cut(img, w, h)
        except Exception as e:
            print(f"[error] {fname}: {e}")
            continue

        print(f"route={result.route} det={result.detector_count} vlm={result.vlm_count}")
        if result.route != "core":
            print(f"  (route={result.route}, 검색 대상 아님)")
            continue

        cut_dir = os.path.join(a.out_dir, base)
        os.makedirs(cut_dir, exist_ok=True)
        n_saved = 0

        for pi, cands in enumerate(result.person_candidates):
            desc = result.descriptors[pi] if pi < len(result.descriptors) else None
            if not cands or desc is None or desc.skeleton is None:
                print(f"  person{pi}: 후보 없음/스켈레톤 없음 - 건너뜀")
                continue
            for rank, c in enumerate(cands[:a.topk], 1):
                out_bvh = os.path.join(
                    cut_dir, f"p{pi}_rank{rank}_{safe(c.pose_id)}.bvh")
                res = refine_bvh(c.bvh_path, desc.skeleton.keypoints,
                                 desc.skeleton.scores, c.view.value,
                                 out_path=out_bvh, search_distance=c.distance)
                if not res.refined:
                    import shutil
                    shutil.copyfile(c.bvh_path, out_bvh)
                print(f"  person{pi} #{rank} {c.pose_id[:30]:30s} "
                      f"{'refined' if res.refined else 'base(' + res.reason + ')'}")
                n_saved += 1

        print(f"  -> {cut_dir}/  ({n_saved}개 bvh)")

    print("\ndone.")


if __name__ == "__main__":
    main()
