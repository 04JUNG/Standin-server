"""스모크 테스트: shot/사람수 분기 + 기하 매칭 + 신뢰도 폴백 계약 검증."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.library import build_synthetic_index
from src.pipeline import Pipeline
from src.config import CFG


class _Img(str):
    @property
    def hint(self): return str(self)


def _pipe():
    return Pipeline(build_synthetic_index())


def test_core_route_and_geometric_topk():
    res = _pipe().process_cut(_Img("full_half standing front 1p"))
    assert res.route == "core"
    assert 1 <= len(res.candidates) <= CFG.top_k_final
    # mock 스켈레톤은 서기 → 기하 매칭 1위는 서기 포즈
    assert res.candidates[0].tags["action"] == "standing"


def test_face_skips():
    res = _pipe().process_cut(_Img("face front 1p"))
    assert res.route == "skip"
    assert res.candidates == []


def test_bust_route_skips_search():
    res = _pipe().process_cut(_Img("bust front 1p"))
    assert res.route == "bust"
    assert res.candidates == []


def test_count_mismatch_is_low_conf():
    res = _pipe().process_cut(_Img("full_half standing front 2p miss"))
    assert res.count_confidence == "low"     # 검출 1 vs VLM 2
    assert res.vlm_count == 2


def test_count_match_is_high_conf():
    res = _pipe().process_cut(_Img("full_half standing front 2p"))
    assert res.count_confidence == "high"


def test_person_confidence_high_on_good_match():
    res = _pipe().process_cut(_Img("full_half standing front 1p"))
    # 서기 쿼리 ≈ 서기 라이브러리 → 거리 작음 → high
    assert res.person_confidence and res.person_confidence[0] == "high"


def test_two_person_two_results():
    res = _pipe().process_cut(_Img("full_half standing front 2p"))
    assert len(res.person_candidates) == 2


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
