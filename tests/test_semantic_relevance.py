"""Explicit feedback is bounded; unseen actions must match the requested role."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_semantic_search import _index, _write_library, POSES
from src.semantic_search import normalize_text, semantic_search, SemanticIndex
from src.semantic_relevance import load_review_preferences, relevance_constraints


def test_handshake_is_not_any_reaching_pose():
    assert relevance_constraints('상대에게 악수를 청하는 자세', '꽃을 앞으로 건네는 자세', set())
    assert not relevance_constraints('상대에게 악수를 청하는 자세', '한 손을 내밀어 악수를 청하는 자세', set())


def test_pushup_is_not_swimming_or_crawling():
    for desc in ['엎드려 수영하는 자세', '몸을 낮춰 포복하는 자세']:
        assert relevance_constraints('팔굽혀펴기를 하는 사람', desc, set())
    assert not relevance_constraints('팔굽혀펴기를 하는 사람', '손과 발끝으로 지지하며 팔굽혀펴기', set())


def test_helper_is_not_the_person_getting_up():
    query = '쓰러진 사람을 일으켜 세우는 자세'
    assert relevance_constraints(query, '몸을 일으켜 걷는 자세', set())
    assert relevance_constraints(query, '상대를 바라보며 서 있는 자세', set())
    assert not relevance_constraints(query, '상대의 팔을 잡아 일으켜 세우는 자세', set())


def test_vertical_sword_guard_is_not_tip_forward():
    q = '칼끝을 상대에게 겨누는 자세'
    assert relevance_constraints(q, '낮게 앉아 검을 세우고 겨누는 자세', {('action', 'aim')})
    assert not relevance_constraints(q, '검을 전방으로 뻗어 겨누는 자세', {('object', 'sword')})
    assert relevance_constraints(q, '권총을 전방으로 뻗어 겨누는 자세', {('object', 'pistol')})


def test_feedback_is_exact_query_only_and_preserves_raw_score():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _write_library(tmp, poses=POSES + [
            ('pitch_alt', POSES[0][1], '한 팔로 야구공을 멀리 던지는 투수')])
        idx = SemanticIndex.load(**paths, excluded_pose_ids={'quarantined_pose'})
        query = '야구공을 던지는 투수'
        before = semantic_search(idx, query, min_score=0)
        # A second eligible family is used solely to test ordering mechanics.
        chosen = before['candidates'][1]['pose_id']
        raw_score = before['candidates'][1]['score']
        idx.review_preferences[normalize_text(query)] = {'preferred_pose_id': chosen}
        after = semantic_search(idx, query, min_score=0)
        assert after['candidates'][0]['pose_id'] == chosen
        assert after['candidates'][0]['score'] == raw_score
        assert after['candidates'][0]['score_components']['human_preference'] == 1
        assert 'review_preference' not in semantic_search(idx, '다른 야구공 투구 문장')['query']


def test_preference_cannot_bypass_threshold_or_constraints():
    with tempfile.TemporaryDirectory() as tmp:
        idx = _index(tmp)
        q = '야구공을 던지는 투수'
        idx.review_preferences[normalize_text(q)] = {'preferred_pose_id': 'chair_rest'}
        result = semantic_search(idx, q)
        assert result['candidates'][0]['pose_id'].startswith('pitch_01')
        assert not any(c['pose_id'] == 'chair_rest' for c in result['candidates'])


def test_preference_loader_rejects_quarantined_or_missing_id():
    with tempfile.TemporaryDirectory() as tmp:
        idx = _index(tmp)
        path = Path(tmp) / 'preferences.json'
        path.write_text(json.dumps({'schema_version': 1, 'entries': [
            {'query': '야구공을 던져', 'preferred_pose_id': 'quarantined_pose'}]}))
        try:
            load_review_preferences(idx, path, normalize_text)
        except ValueError:
            return
        raise AssertionError('excluded preference accepted')


if __name__ == '__main__':
    import traceback
    funcs = [v for k,v in sorted(globals().items()) if k.startswith('test_')]
    passed = 0
    for function in funcs:
        try:
            function()
            print('PASS', function.__name__)
            passed += 1
        except Exception:
            traceback.print_exc()
    print(f'{passed}/{len(funcs)} passed')
    raise SystemExit(0 if passed == len(funcs) else 1)
