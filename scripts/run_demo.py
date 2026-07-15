"""
오프라인 데모: API 키·모델 없이 mock으로 파이프라인 전체를 돌린다.

  python scripts/run_demo.py

'image'는 힌트 문자열로 대체(mock가 파일명 대신 힌트를 읽음).
실제로는 PNG 경로/PIL 이미지를 넘긴다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.library import build_synthetic_index, save_index, load_index
from src.config import CFG
from src.pipeline import Pipeline


def _ensure_index():
    if not os.path.exists(CFG.index_path):
        entries = build_synthetic_index()
        save_index(entries, CFG.index_path)
        print(f"[index] 합성 인덱스 {len(entries)}개 생성 → {CFG.index_path}")
    return load_index(CFG.index_path)


class _Img(str):
    """mock용: 힌트 문자열을 담은 가짜 이미지."""
    @property
    def hint(self): return str(self)


def _print(title, res):
    print(f"\n=== {title} ===")
    print(f"route={res.route}  count_conf={res.count_confidence} "
          f"(det={res.detector_count}, vlm={res.vlm_count})")
    for n in res.notes:
        print("  ·", n)
    for i, c in enumerate(res.candidates, 1):
        print(f"  [{i}] {c.pose_id:12s} view={c.view.value:13s} "
              f"dist={c.distance:.3f}  tags={c.tags.get('action')}")


def main():
    entries = _ensure_index()
    pipe = Pipeline(entries)

    cases = [
        ("1인 전신 서기(정면)", _Img("full_half standing front 1p")),
        ("1인 앉기(측면)",     _Img("full_half sitting side 1p")),
        ("1인 뻗기(3/4)",      _Img("full_half reaching three_quarter 1p")),
        ("2인 대화(검출 정상)", _Img("full_half standing front 2p")),
        ("2인, 검출기 1명 놓침", _Img("full_half standing front 2p miss")),
        ("얼굴 클로즈업",       _Img("face front 1p")),
    ]
    for title, img in cases:
        _print(title, pipe.process_cut(img))


if __name__ == "__main__":
    main()
