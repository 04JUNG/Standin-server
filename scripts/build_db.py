"""라이브러리 → SQLite DB.
  BVH_DIR 지정 시: 실 BVH 폴더 스캔(build_index_from_bvh_dir)
  미지정 시:      합성 포즈(build_synthetic_index)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.library import build_synthetic_index, build_index_from_bvh_dir
from src.repo import build_db

def main():
    db = os.getenv("DB_PATH", "data/poses.db")
    bvh_dir = os.getenv("BVH_DIR")
    if bvh_dir:
        entries = build_index_from_bvh_dir(bvh_dir)
        src = f"BVH dir {bvh_dir}"
    else:
        entries = build_synthetic_index()
        src = "synthetic"
    n = build_db(entries, db)
    poses = len(set(e.pose_id for e in entries))
    print(f"[db] {src}: {poses} poses / {n} projections -> {db}")

if __name__ == "__main__":
    main()
