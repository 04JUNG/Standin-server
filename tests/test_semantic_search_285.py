"""시맨틱(자연어) 검색 계약 — 04J-285.

작은 합성 라이브러리(SQLite + 태그 v2 + 어휘)로 돈다. 실제 1,981개 라이브러리 검사는
``SEMANTIC_REAL_DATA_DIR``(pose-library-viewer 폴더)을 지정했을 때만 돈다.

    python tests/test_semantic_search.py          # 자체 러너
    SEMANTIC_REAL_DATA_DIR=~/Desktop/.../pose-library-viewer python tests/test_semantic_search.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import semantic_search_285 as ss
from src.semantic_search_285 import (ErrorCode, SemanticIndex, SemanticSearchService,
                                 SemanticSettings, body_clauses, semantic_search)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEXICON = os.path.join(ROOT, "config", "semantic_lexicon.v1.json")

TAXONOMY = {
    "version": "2.0",
    "groups": [
        {"key": "pose_action", "fields": ["action", "posture"]},
        {"key": "scene", "fields": ["domain", "activity"]},
        {"key": "objects", "fields": ["object"]},
        {"key": "meaning", "fields": ["emotion", "intent"]},
        {"key": "description", "fields": []},
    ],
    "vocabulary": {
        "posture": {"standing": "서기", "sitting": "앉기", "kneeling": "무릎 꿇기"},
        "action": {"throw": "던지기", "kick": "발차기", "hold": "들기·잡기", "rest": "쉬기",
                   "block": "막기", "reach": "뻗기", "raise_hand": "손 들기",
                   "defend": "방어", "prepare": "준비하기", "bend": "숙이기",
                   "lift": "들어 올리기", "fly": "비행·부양", "punch": "주먹 공격",
                   "hug": "안기", "pray": "기도", "wave": "손 흔들기", "lean": "기울이기·기대기",
                   "stretch": "스트레칭", "balance": "균형 잡기", "talk": "말하기·통화",
                   "guard": "가드", "slash": "베기", "strike": "내리치기", "pass": "패스",
                   "catch": "받기", "evade": "회피", "land": "착지", "dribble": "드리블", "run": "달리기"},
        "domain": {"sports": "스포츠", "combat": "전투", "daily": "일상"},
        "activity": {"baseball": "야구", "soccer": "축구", "swordsmanship": "검술",
                     "goalkeeping": "골키퍼 선방", "basketball": "농구", "volleyball": "배구"},
        "object": {"ball": "공", "sword": "검", "chair": "의자", "phone": "휴대폰", "hammer": "망치",
                   "spear": "창", "water": "물"},
        "intent": {"attack": "공격", "guard": "경계", "defense": "방어"},
        "emotion": {"tired": "피곤함"},
    },
}

POSES = [
    # pose_id, groups(field,value…), description
    ("pitch_01", [("action", "throw"), ("posture", "standing"), ("domain", "sports"),
                  ("activity", "baseball"), ("object", "ball")],
     "팔을 머리 위로 휘둘러 야구공을 던지는 자세"),
    ("pitch_01_mirror", [("action", "throw"), ("posture", "standing"), ("domain", "sports"),
                         ("activity", "baseball"), ("object", "ball")],
     "팔을 머리 위로 휘둘러 야구공을 던지는 미러 자세"),
    ("keeper_01", [("action", "block"), ("domain", "sports"), ("activity", "goalkeeping"),
                   ("activity", "soccer"), ("object", "ball"), ("intent", "defense")],
     "다리를 넓게 벌리고 몸을 낮춰 페널티킥을 막을 준비를 하는 골키퍼 자세"),
    ("sword_guard", [("action", "hold"), ("posture", "standing"), ("domain", "combat"),
                     ("activity", "swordsmanship"), ("object", "sword"), ("intent", "guard")],
     "양손으로 검을 쥐고 다리를 벌려 경계하는 자세"),
    ("chair_rest", [("action", "rest"), ("posture", "sitting"), ("domain", "daily"),
                    ("object", "chair"), ("emotion", "tired")],
     "의자에 앉아 등을 기대고 쉬는 자세"),
    ("arms_up_knee", [("action", "raise_hand"), ("posture", "standing")],
     "양팔을 위로 들고 한쪽 무릎을 굽혀 드는 자세"),
    ("kneel_bow", [("posture", "kneeling"), ("action", "bend")],
     "한쪽 무릎을 꿇고 상체를 앞으로 숙인 자세"),
    ("quarantined_pose", [("action", "throw"), ("activity", "baseball")],
     "야구공을 던지는 격리된 자세"),
]


def _write_library(tmp: str, *, poses=POSES, db_ids=None, tags_db_sha=None) -> dict:
    db = os.path.join(tmp, "poses.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE poses (pose_id TEXT PRIMARY KEY, bvh_path TEXT, source TEXT, "
                "license TEXT, shot TEXT, action TEXT, relationship TEXT, set_id TEXT, "
                "set_role TEXT, meta_json TEXT)")
    for pid in (db_ids if db_ids is not None else [p[0] for p in poses]):
        con.execute("INSERT INTO poses VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (pid, f"data/bvh/{pid}.bvh", "cmu", "CMU", "full_half", "other", "solo",
                     None, None, json.dumps({"category": "테스트", "review_status": "accepted"})))
    con.commit()
    con.close()
    sha = hashlib.sha256(open(db, "rb").read()).hexdigest()
    labels = {(f, v): lab for f, vals in TAXONOMY["vocabulary"].items() for v, lab in vals.items()}
    group_of = {f: g["key"] for g in TAXONOMY["groups"] for f in g["fields"]}
    entries = []
    for pid, tags, desc in poses:
        groups = {"pose_action": [], "scene": [], "objects": [], "meaning": []}
        for f, v in tags:
            groups[group_of[f]].append({"field": f, "value": v, "label": labels[(f, v)]})
        entries.append({"pose_id": pid, "version": "2.0", "status": "reviewed",
                        "groups": groups, "description": {"text": desc, "source": "test"}})
    tags_path = os.path.join(tmp, "search-v2.json")
    json.dump({"version": "2.0", "source_db_sha256": tags_db_sha or sha, "poses": entries},
              open(tags_path, "w"), ensure_ascii=False)
    tax_path = os.path.join(tmp, "search-taxonomy-v2.json")
    json.dump(TAXONOMY, open(tax_path, "w"), ensure_ascii=False)
    os.makedirs(os.path.join(tmp, "thumbs"), exist_ok=True)
    open(os.path.join(tmp, "thumbs", "pitch_01__front.jpg"), "wb").write(b"jpg")
    return {"tags_path": tags_path, "taxonomy_path": tax_path, "lexicon_path": LEXICON,
            "db_path": db, "thumbs_dir": os.path.join(tmp, "thumbs")}


def _index(tmp: str, **overrides) -> SemanticIndex:
    paths = _write_library(tmp)
    paths.update(overrides)
    allowed = {p[0] for p in POSES} - {"quarantined_pose"}
    return SemanticIndex.load(allowed_pose_ids=allowed, library_version="lib-test", **paths)


# ───────────── ok: 후보 + 유효 JSON + ID 일치 ─────────────


def test_ok_returns_candidates_with_matching_pose_json():
    with tempfile.TemporaryDirectory() as tmp:
        idx = _index(tmp)
        r = semantic_search(idx, "야구공을 던지는 투수", source="user")
        assert r["status"] == "ok", r
        assert r["error"] is None
        top = r["candidates"][0]
        assert top["pose_id"] in {"pitch_01", "pitch_01_mirror"}
        for c in r["candidates"]:
            assert c["pose"]["pose_id"] == c["pose_id"]
            assert c["pose"]["schema"] == "standin-pose-record-v1"
            assert c["pose"]["bvh_url"] == f"/pose/{c['pose_id']}/bvh"
            assert c["refine_allowed"] is False
            assert c["score"] >= r["threshold"]
        json.dumps(r, ensure_ascii=False)  # 직렬화 가능
        labels = {m["label"] for m in top["evidence"]["matched_tags"]}
        assert {"던지기", "야구", "공"} <= labels


def test_quarantined_pose_is_never_returned():
    with tempfile.TemporaryDirectory() as tmp:
        idx = _index(tmp)
        r = semantic_search(idx, "야구공을 던지는 자세", top_k=20, min_score=0.0)
        ids = {c["pose_id"] for c in r["candidates"]}
        assert "quarantined_pose" not in ids
        assert idx.identity["excluded_pose_count"] == 1


def test_mirror_pair_is_collapsed_into_one_candidate():
    with tempfile.TemporaryDirectory() as tmp:
        idx = _index(tmp)
        r = semantic_search(idx, "야구공을 던지는 투수", top_k=5, min_score=0.0)
        ids = [c["pose_id"] for c in r["candidates"]]
        assert not ({"pitch_01", "pitch_01_mirror"} <= set(ids))
        assert r["candidates"][0]["mirror_alternates"]


def test_thumbnail_urls_only_for_existing_files():
    with tempfile.TemporaryDirectory() as tmp:
        idx = _index(tmp)
        pose = idx.pose_json("pitch_01")
        assert pose["thumbnail_urls"] == {"front": "/pose/pitch_01/thumbnail?view=front"}
        assert idx.pose_json("keeper_01")["thumbnail_urls"] == {}


def test_body_description_query_uses_upper_and_lower_clauses():
    with tempfile.TemporaryDirectory() as tmp:
        idx = _index(tmp)
        r = semantic_search(idx, "한쪽 무릎을 꿇고 상체를 앞으로 숙인 자세")
        assert r["status"] == "ok"
        assert r["candidates"][0]["pose_id"] == "kneel_bow"
        body = r["candidates"][0]["evidence"]["body"]
        assert set(body) == {"upper", "lower"}


def test_vlm_source_uses_its_own_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        idx = _index(tmp)
        settings = SemanticSettings(min_score_user=0.9, min_score_vlm=0.1)
        r_user = semantic_search(idx, "의자에 앉아 쉬는 자세", source="user", settings=settings)
        r_vlm = semantic_search(idx, "의자에 앉아 쉬는 자세", source="vlm", settings=settings)
        assert r_user["threshold"] == 0.9 and r_vlm["threshold"] == 0.1
        assert r_vlm["status"] == "ok"


# ───────────── no_result: 정상 검색, 임계값 미달 ─────────────


def test_no_result_is_not_an_error():
    with tempfile.TemporaryDirectory() as tmp:
        idx = _index(tmp)
        r = semantic_search(idx, "우주선 조종석에서 레버를 당기는 조종사")
        assert r["status"] == "no_result", r["candidates"]
        assert r["error"] is None
        assert r["candidates"] == []
        assert len(r["near_misses"]) >= 1  # 디버깅용 임계값 아래 최상위


# ───────────── error: 원인 코드 ─────────────


def test_error_invalid_query_and_source():
    with tempfile.TemporaryDirectory() as tmp:
        idx = _index(tmp)
        assert semantic_search(idx, "   ")["error"]["code"] == ErrorCode.INVALID_QUERY
        assert semantic_search(idx, "!!!")["error"]["code"] == ErrorCode.INVALID_QUERY
        assert semantic_search(idx, "야구", source="bff")["error"]["code"] == ErrorCode.INVALID_SOURCE
        assert semantic_search(idx, "가" * 600)["error"]["code"] == ErrorCode.INVALID_QUERY


def test_error_index_not_loaded():
    r = semantic_search(None, "야구공을 던지는 투수")
    assert r["status"] == "error" and r["error"]["code"] == ErrorCode.INDEX_NOT_LOADED


def test_error_index_version_mismatch_via_service():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _write_library(tmp, tags_db_sha="0" * 64)
        service = SemanticSearchService.from_paths(SemanticSettings(), **paths)
        assert not service.ready
        assert service.health()["error_code"] == ErrorCode.INDEX_VERSION_MISMATCH
        r = service.search("야구공을 던지는 투수")
        assert r["status"] == "error"
        assert r["error"]["code"] == ErrorCode.INDEX_VERSION_MISMATCH
        # 잘못된 쿼리는 색인 상태보다 먼저 알린다
        assert service.search("")["error"]["code"] == ErrorCode.INVALID_QUERY


def test_error_tags_reference_pose_missing_from_db():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _write_library(tmp, db_ids=[p[0] for p in POSES if p[0] != "keeper_01"])
        service = SemanticSearchService.from_paths(SemanticSettings(),
                                                   verify_db_sha256=False, **paths)
        assert service.health()["error_code"] == ErrorCode.INDEX_INCONSISTENT


def test_error_missing_files():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _write_library(tmp)
        paths["tags_path"] = os.path.join(tmp, "nope.json")
        service = SemanticSearchService.from_paths(SemanticSettings(), **paths)
        assert service.health()["error_code"] == ErrorCode.INDEX_LOAD_FAILED


def test_error_pose_id_mismatch_is_surfaced_not_dropped():
    with tempfile.TemporaryDirectory() as tmp:
        idx = _index(tmp)
        original = idx.pose_json

        def corrupted(pose_id):
            pose = original(pose_id)
            if pose and pose_id.startswith("pitch_01"):
                pose = dict(pose, pose_id="someone_else")
            return pose

        idx.pose_json = corrupted
        r = semantic_search(idx, "야구공을 던지는 투수")
        assert r["status"] == "error"
        assert r["error"]["code"] == ErrorCode.POSE_ID_MISMATCH
        assert r["candidates"] == []


def test_error_pose_json_missing():
    with tempfile.TemporaryDirectory() as tmp:
        idx = _index(tmp)
        idx.pose_records.pop("keeper_01")
        r = semantic_search(idx, "페널티킥을 막는 골키퍼")
        assert r["error"]["code"] == ErrorCode.POSE_JSON_MISSING


def test_error_encoder_failure():
    with tempfile.TemporaryDirectory() as tmp:
        idx = _index(tmp)

        class Broken:
            name = "broken"

            def encode(self, text):
                raise TimeoutError("embedding backend timeout")

        idx.encoder = Broken()
        r = semantic_search(idx, "야구공을 던지는 투수")
        assert r["error"]["code"] == ErrorCode.ENCODER_FAILED
        assert "timeout" in r["error"]["message"]


def test_lexicon_keys_missing_from_taxonomy_are_reported_not_fatal():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _write_library(tmp)
        lex = os.path.join(tmp, "lex.json")
        json.dump({"version": "t", "field_weights": {"action": 1.0},
                   "synonyms": {"action:teleport": ["순간이동"], "action:throw": ["투척"]}},
                  open(lex, "w"), ensure_ascii=False)
        paths["lexicon_path"] = lex
        idx = SemanticIndex.load(**paths)
        assert idx.identity["lexicon_skipped_keys"] == ["action:teleport"]
        assert any(c.surface == "투척" for c in idx.lexicon.extract("투척하는 자세"))


def test_real_lexicon_matches_real_taxonomy_when_available():
    root = os.getenv("SEMANTIC_REAL_DATA_DIR")
    if not root:
        print("  skip: SEMANTIC_REAL_DATA_DIR not set")
        return
    tax = json.load(open(os.path.join(root, "dist/data/search-taxonomy-v2.json")))
    lex = ss.ConceptLexicon(tax, json.load(open(LEXICON)))
    assert lex.skipped_keys == []


# ───────────── 확정 태그 입력(VLM 구조화 출력) ─────────────


def test_tags_only_query_uses_taxonomy_values():
    with tempfile.TemporaryDirectory() as tmp:
        idx = _index(tmp)
        r = semantic_search(idx, "", source="vlm", tags=["action:throw", "activity:baseball"])
        assert r["status"] == "ok", r
        assert r["candidates"][0]["pose_id"].startswith("pitch_01")
        assert [c["surface"] for c in r["query"]["concepts"]][:2] == [
            "tag:action:throw", "tag:activity:baseball"]
        # 같은 필드 값은 하나의 OR 개념
        r2 = semantic_search(idx, "", tags=["action:throw", "action:kick"])
        assert [c["surface"] for c in r2["query"]["concepts"]] == ["tag:action:throw|action:kick"]


def test_tags_override_overlapping_text_concepts():
    with tempfile.TemporaryDirectory() as tmp:
        idx = _index(tmp)
        r = semantic_search(idx, "야구공을 던지는 투수", tags=["action:throw"])
        surfaces = [c["surface"] for c in r["query"]["concepts"]]
        assert "tag:action:throw" in surfaces and "던지는" not in surfaces


def test_invalid_tags_are_rejected_with_code():
    with tempfile.TemporaryDirectory() as tmp:
        idx = _index(tmp)
        r = semantic_search(idx, "야구", tags=["action:teleport"])
        assert r["error"]["code"] == ErrorCode.INVALID_TAGS
        assert r["error"]["detail"]["unknown"] == ["action:teleport"]
        assert semantic_search(idx, "", tags=["throw"])["error"]["code"] == ErrorCode.INVALID_TAGS


def test_conjugations_cover_common_endings():
    assert {"젖힌", "젖혀", "젖히고"} <= ss.conjugations("젖히")
    assert {"젓는", "젓고"} <= ss.conjugations("젓")
    assert "휘둘러" in ss.conjugations("휘두르")


def test_service_counts_statuses_for_health():
    with tempfile.TemporaryDirectory() as tmp:
        idx = _index(tmp)
        service = SemanticSearchService(idx, SemanticSettings())
        service.search("야구공을 던지는 투수")
        service.search("우주선 조종석에서 레버를 당기는 조종사")
        service.search("", source="vlm")
        stats = service.health()["stats"]
        assert stats["status.ok"] == 1 and stats["status.no_result"] == 1
        assert stats["status.error"] == 1 and stats["error.invalid_query"] == 1
        assert stats["source.user"] == 2 and stats["source.vlm"] == 1


def test_opponents_causal_action_is_not_the_requested_action():
    poses = [
        ("attack", [("action", "punch")], "한 주먹을 앞으로 날리는 자세"),
        ("guard", [("action", "guard"), ("action", "defend")], "두 손으로 얼굴을 가리고 버티는 자세"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        idx = SemanticIndex.load(**_write_library(tmp, poses=poses))
        r = semantic_search(idx, "상대가 주먹을 날리니까 얼굴을 가리고 버티는 사람")
        assert r["candidates"][0]["pose_id"] == "guard", r
        assert r["query"]["foreground_text"] == "얼굴을 가리고 버티는 사람"
        assert all(("action", "punch") not in c.alternatives
                   for c in idx.lexicon.extract(r["query"]["foreground_text"]))
        active = semantic_search(idx, "상대에게 주먹을 날리는 사람")
        assert "foreground_text" not in active["query"]
        assert active["candidates"][0]["pose_id"] == "attack", active


def test_twohand_sword_strike_does_not_return_onehand_or_hammer():
    poses = [
        ("twohand", [("action", "strike"), ("object", "sword")], "양손으로 검을 머리 위로 들어 내리치는 자세"),
        ("onehand", [("action", "slash"), ("object", "sword")], "한 손으로 검을 머리 위에서 내려베는 자세"),
        ("hammer", [("action", "strike"), ("object", "hammer")], "망치를 두 손으로 머리 위에서 내리치는 자세"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        idx = SemanticIndex.load(**_write_library(tmp, poses=poses))
        r = semantic_search(idx, "양손으로 검을 머리 위에서 내려베는 자세")
        assert [c["pose_id"] for c in r["candidates"]] == ["twohand"], r


def test_basketball_compound_does_not_match_every_ball_sport():
    poses = [
        ("basket", [("action", "pass"), ("activity", "basketball"), ("object", "ball")], "농구공을 가슴 앞에서 패스하는 자세"),
        ("volley", [("action", "pass"), ("activity", "volleyball"), ("object", "ball")], "배구공을 가슴 앞에서 패스하는 자세"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        idx = SemanticIndex.load(**_write_library(tmp, poses=poses))
        r = semantic_search(idx, "농구공을 가슴 앞에서 패스하는 자세")
        assert [c["pose_id"] for c in r["candidates"]] == ["basket"], r


def test_prior_sitting_state_is_separated_from_stretching():
    with tempfile.TemporaryDirectory() as tmp:
        idx = _index(tmp)
        r = semantic_search(idx, "오래 앉아 있다가 일어나 몸을 쭉 펴는 사람")
        assert r["query"]["foreground_text"] == "일어나 몸을 쭉 펴는 사람"
        assert "앉아" not in r["query"]["foreground_text"]
        concepts = idx.lexicon.extract(r["query"]["foreground_text"])
        assert any(("action", "stretch") in c.alternatives for c in concepts)


def test_negative_contrast_prefers_catching_not_throwing():
    poses = [
        ('catch', [('action','catch'),('object','ball')], '두 손을 내밀어 공을 받는 자세'),
        ('throw', [('action','throw'),('object','ball')], '두 손을 내밀어 공을 던지는 자세'),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        idx = SemanticIndex.load(**_write_library(tmp, poses=poses))
        r = semantic_search(idx, '공을 던지는 것이 아니라 잡으려는 자세')
        assert r['candidates'][0]['pose_id'] == 'catch', r
        assert 'action:throw' in r['query']['negated_actions']
        assert all(c['pose_id'] != 'throw' for c in r['candidates']), r


def test_incoming_kick_is_not_the_evading_persons_action():
    poses = [
        ('evade', [('action','evade')], '상체를 뒤로 젖혀 공격을 피하는 자세'),
        ('kick', [('action','kick')], '상체를 뒤로 젖혀 발차기를 하는 자세'),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        idx = SemanticIndex.load(**_write_library(tmp, poses=poses))
        r = semantic_search(idx, '날아오는 발차기를 피하려고 상체를 뒤로 젖히는 자세')
        assert r['candidates'][0]['pose_id'] == 'evade', r
        assert all(c['pose_id'] != 'kick' for c in r['candidates']), r


def test_incoming_sword_does_not_require_holding_a_sword():
    poses = [
        ('evade', [('action','evade')], '몸을 옆으로 빼며 회피하는 자세'),
        ('sword', [('action','strike'),('object','sword')], '몸을 옆으로 빼며 검을 내리치는 자세'),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        idx = SemanticIndex.load(**_write_library(tmp, poses=poses))
        r = semantic_search(idx, '칼이 날아오자 몸을 옆으로 빼는 사람')
        assert r['candidates'][0]['pose_id'] == 'evade', r
        assert '칼' not in r['query']['foreground_text']


def test_explicit_spear_does_not_return_sword():
    poses = [
        ('spear', [('action','strike'),('object','spear')], '창을 앞으로 뻗어 내려치는 자세'),
        ('sword', [('action','strike'),('object','sword')], '검을 앞으로 뻗어 내려치는 자세'),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        idx = SemanticIndex.load(**_write_library(tmp, poses=poses))
        r = semantic_search(idx, '창을 앞으로 뻗어 내려치는 자세')
        assert [c['pose_id'] for c in r['candidates']] == ['spear'], r


def test_hand_on_floor_does_not_match_only_knee_on_floor():
    poses = [
        ('hand', [('action','guard')], '한 손은 바닥에 짚고 다른 손으로 얼굴을 가리는 자세'),
        ('knee', [('action','guard')], '한 무릎으로 바닥을 짚고 다른 손으로 얼굴을 가리는 자세'),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        idx = SemanticIndex.load(**_write_library(tmp, poses=poses))
        r = semantic_search(idx, '한 손은 바닥에 짚고 다른 손으로 얼굴을 가리는 자세')
        assert [c['pose_id'] for c in r['candidates']] == ['hand'], r


def test_single_character_water_is_not_found_inside_obstacle():
    with tempfile.TemporaryDirectory() as tmp:
        idx = _index(tmp)
        assert not any(('object','water') in c.alternatives for c in idx.lexicon.extract('장애물이 보이는 자세'))
        assert any(('object','water') in c.alternatives for c in idx.lexicon.extract('물에 들어가는 자세'))


def test_bouncing_a_ball_requires_dribble_not_running():
    poses = [
        ('dribble', [('action','dribble'),('activity','basketball'),('object','ball')], '농구공을 낮게 드리블하는 자세'),
        ('run', [('action','run'),('activity','basketball'),('object','ball')], '농구공을 들고 달리는 자세'),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        idx = SemanticIndex.load(**_write_library(tmp, poses=poses))
        r = semantic_search(idx, '농구공을 바닥에 튀기는 자세')
        assert [c['pose_id'] for c in r['candidates']] == ['dribble'], r


def test_showing_palm_does_not_require_floor_contact():
    poses = [('palm', [('action','wave')], '한 손을 높이 들어 손바닥을 펴고 좌우로 흔드는 자세')]
    with tempfile.TemporaryDirectory() as tmp:
        idx = SemanticIndex.load(**_write_library(tmp, poses=poses))
        r = semantic_search(idx, '한 손을 높이 들어 손바닥을 펴고 좌우로 흔드는 자세')
        assert r['candidates'][0]['pose_id'] == 'palm', r
        assert r['candidates'][0]['score_components']['requested_constraint_match'] == 1.0


def test_reaching_to_floor_does_not_require_support_contact():
    poses = [('reach', [('action','reach')], '바닥에 있는 물건을 향해 손을 뻗는 자세')]
    with tempfile.TemporaryDirectory() as tmp:
        idx = SemanticIndex.load(**_write_library(tmp, poses=poses))
        r = semantic_search(idx, '바닥에 있는 물건을 향해 손을 뻗는 자세')
        assert r['candidates'][0]['pose_id'] == 'reach', r


# ───────────── 텍스트 처리 단위 ─────────────


def test_body_clauses_keep_modifier_with_its_body_part():
    out = body_clauses("양팔을 위로 들고 한쪽 무릎을 굽힌 자세")
    assert out == {"upper": ["양팔을위로들고"], "lower": ["한쪽무릎을굽힌"]}
    assert body_clauses("왼팔을 앞으로 뻗은")["upper"] == ["왼팔을앞으로뻗은"]
    assert body_clauses("발레를 하는 등반가") == {"upper": [], "lower": []}


def test_single_char_object_needs_particle_boundary():
    with tempfile.TemporaryDirectory() as tmp:
        idx = _index(tmp)
        surfaces = {c.surface for c in idx.lexicon.extract("공격하는 자세")}
        assert "공" not in surfaces and "공격" in surfaces
        assert "공" in {c.surface for c in idx.lexicon.extract("공을 차는")}


def test_settings_validation():
    for bad in (SemanticSettings(top_k=0), SemanticSettings(min_score_user=1.5),
                SemanticSettings(weight_concept=0, weight_body=0, weight_text=0)):
        try:
            bad.validate()
        except ValueError:
            continue
        raise AssertionError(f"{bad} should be invalid")


# ───────────── HTTP 경계(fastapi가 있을 때) ─────────────


def test_http_route_status_codes():
    import pytest
    pytest.skip("Source direct-search HTTP endpoint is not imported; routing adapter is tested separately")
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("  skip: fastapi not installed")
        return
    from api import app as app_module

    with tempfile.TemporaryDirectory() as tmp:
        idx = _index(tmp)
        app_module.STATE["semantic"] = SemanticSearchService(idx, SemanticSettings())
        try:
            client = TestClient(app_module.app)
            ok = client.post("/semantic-search", json={"query": "야구공을 던지는 투수"})
            assert ok.status_code == 200 and ok.json()["status"] == "ok"
            assert ok.json()["candidates"][0]["pose"]["schema"] == "standin-pose-record-v1"
            nr = client.post("/semantic-search", json={"query": "우주선 조종석 레버"})
            assert nr.status_code == 200 and nr.json()["status"] == "no_result"
            bad = client.post("/semantic-search", json={"query": "  "})
            assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_query"
            app_module.STATE["semantic"] = SemanticSearchService(
                None, SemanticSettings(),
                ss.SemanticSearchError(ErrorCode.INDEX_NOT_LOADED, "missing"))
            down = client.post("/semantic-search", json={"query": "야구"})
            assert down.status_code == 503 and down.json()["status"] == "error"
        finally:
            app_module.STATE.pop("semantic", None)


# ───────────── 실제 라이브러리(선택) ─────────────


def test_real_library_integrity_when_available():
    root = os.getenv("SEMANTIC_REAL_DATA_DIR")
    if not root:
        print("  skip: SEMANTIC_REAL_DATA_DIR not set")
        return
    idx = SemanticIndex.load(
        tags_path=os.path.join(root, "dist/data/search-v2.json"),
        taxonomy_path=os.path.join(root, "dist/data/search-taxonomy-v2.json"),
        lexicon_path=LEXICON,
        db_path=os.path.join(root, "source_bundle/poses.db"),
    )
    assert idx.pose_count == len(json.load(open(os.path.join(root, "dist/data/search-v2.json")))["poses"])
    for pose_id in idx.pose_records:
        assert idx.pose_json(pose_id)["pose_id"] == pose_id


def test_dense_nonfinite_query_is_error_not_no_result():
    import numpy as np
    with tempfile.TemporaryDirectory() as tmp:
        index = _index(tmp)
        index.dense_encoder = type("BrokenDense", (), {
            "encode": lambda self, text: np.array([[float("nan"), 1.]])})()
        index.dense_matrix = np.ones((len(index.docs), 2))
        result = semantic_search(index, "공을 던지는", settings=SemanticSettings(weight_dense=.3))
        assert result["status"] == "error" and result["error"]["code"] == "encoder_failed"


def test_explicit_exclusions_never_appear_in_any_response_list():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _write_library(tmp)
        excluded = {"pitch_01", "pitch_01_mirror"}
        index = SemanticIndex.load(excluded_pose_ids=excluded, **paths)
        assert excluded.isdisjoint(d.pose_id for d in index.docs)
        for query in ("공을 던지는 자세", "야구 투수", "존재하지 않는 동작"):
            result = semantic_search(index, query, min_score=0)
            ids = {c["pose_id"] for c in result["candidates"]}
            ids.update(a["pose_id"] for c in result["candidates"] for a in c["mirror_alternates"])
            ids.update(c["pose_id"] for c in result["near_misses"])
            assert excluded.isdisjoint(ids), (query, ids)


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
