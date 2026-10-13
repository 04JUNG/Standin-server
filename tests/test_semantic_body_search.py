from src.body_language import (
    compare_observations, observations_from_text, query_observations,
)
from src.search_routing import PoseDescription, RoutingConfig, assess_semantic_evidence
from src.semantic_search import SemanticQuery, TagSemanticSearcher
from src.vlm.client import parse_pose_descriptions
from src.vlm import prompts


def record(pose_id, text, *, postures=(), actions=("jump",), activities=(), domains=()):
    return {"pose_id": pose_id, "description": {"text": text}, "groups": {
        "pose_action": [{"field": "action", "value": a} for a in actions]
                       + [{"field": "posture", "value": p} for p in postures],
        "scene": [{"field": "activity", "value": a} for a in activities]
                 + [{"field": "domain", "value": d} for d in domains],
    }}


def test_query_preserves_observations_and_explicit_unknown():
    desc = PoseDescription(
        body_details="상체를 뒤로 젖히고 두 무릎을 굽힌다",
        body_observations={"knees": "bent", "torso": "unknown"},
        search_text="양 무릎을 굽힌 자세")
    query = SemanticQuery.from_description(desc)
    assert query.body_details == desc.body_details
    assert query_observations(query.body_details, query.body_observations) == {"knees": "bent"}


def test_body_match_outweighs_speculative_scene_and_generic_standing():
    index = TagSemanticSearcher([
        record("body_match", "두 무릎을 굽히고, 다리를 넓게 벌리고, 두 팔을 내린 자세"),
        record("scene_match", "다리를 앞뒤로 뻗고, 두 팔을 올린 자세",
               postures=["standing"], activities=["hero_action"], domains=["fantasy"]),
    ])
    query = SemanticQuery(
        actions=["jump"], postures=["standing"], activities=["hero_action"],
        domains=["fantasy"], body_observations={
            "knees": "bent", "legs": "apart", "arms": "lowered"})
    result = index.search(query)
    assert result.hits[0].pose_id == "body_match"
    components = result.hits[0].matched["score_components"]
    assert components["body_reward"] == 8
    assert components["shape_tags"] == 3  # standing adds no points
    assert components["context"] <= 0.5


def test_standing_and_many_action_guesses_cannot_accumulate_score():
    index = TagSemanticSearcher([record("p", "", postures=["standing"],
                                       actions=["jump", "fall", "lean_back"])])
    query = SemanticQuery(postures=["standing"], actions=["jump", "fall", "lean_back"])
    hit = index.search(query).hits[0]
    assert hit.score == 3 and hit.matched["score_components"]["shape_tags"] == 3


def test_unknown_library_attributes_are_neutral_and_conflicts_are_explicit():
    neutral = compare_observations({"arms": "lowered"}, {})
    assert neutral["reward"] == neutral["penalty"] == 0
    conflict = compare_observations({"knees": "bent"}, {"knees": "straight"})
    assert conflict["penalty"] == 3
    assert conflict["conflicts"]["knees"] == {"query": "bent", "candidate": "straight"}
    assert compare_observations({"arms": "unknown"}, {"arms": "raised"})["penalty"] == 0


def test_specific_body_can_search_when_action_is_unknown():
    desc = PoseDescription(
        body_details="두 무릎을 굽히고 다리를 좌우로 벌린 자세",
        search_text="두 무릎을 굽혀 다리를 넓게 벌린 자세",
        body_observations={"knees": "bent", "legs": "apart"}, certainty="medium")
    assert assess_semantic_evidence(desc, RoutingConfig()).valid
    index = TagSemanticSearcher([
        record("observed", "양 무릎을 굽혀 다리를 넓게 벌린 자세", actions=[]),
        record("unrelated", "한 다리를 앞으로 뻗는 자세", actions=["kick"]),
    ])
    hits = index.search(SemanticQuery.from_description(desc)).hits
    assert [h.pose_id for h in hits] == ["observed"]


def test_plain_standing_body_is_still_too_generic():
    desc = PoseDescription(
        postures=["standing"], body_details="곧게 서서 두 팔을 내리고 다리를 모은다",
        search_text="두 팔을 내리고 곧게 선 자세", certainty="medium",
        body_observations={"knees": "straight", "legs": "together",
                           "arms": "lowered", "torso": "upright"})
    assert not assess_semantic_evidence(desc, RoutingConfig()).valid


def test_parser_separates_ungrounded_context_and_rejects_invalid_body_values():
    payload = {"people": [{
        "actions": ["jump"], "domains": ["fantasy"], "activities": ["hero_action"],
        "body_observations": {"knees": "bent", "legs": "flying", "x": 0.5},
        "body_details": "두 무릎을 굽혀 다리를 벌린다", "search_text": "무릎을 굽힌 도약 자세",
        "certainty": "medium",
    }]}
    desc = parse_pose_descriptions(payload, 1)[0]
    assert desc.actions == ["jump"] and desc.domains == desc.activities == []
    assert desc.body_observations == {"knees": "bent"}
    assert desc.dropped["ungrounded_domains"] == ["fantasy"]
    assert "legs" in desc.dropped["body_observations"] and "x" in desc.dropped["body_observations"]
    payload["people"][0]["context_evidence"] = "몸 배치가 공중 도약의 활용 맥락과 연결됨"
    assert parse_pose_descriptions(payload, 1)[0].domains == ["fantasy"]


def test_text_observations_do_not_treat_negation_as_evidence():
    result = observations_from_text("무릎을 굽히지 않고, 양팔을 아래로 내린다")
    assert "knees" not in result
    assert result["arms"] == "lowered"


def test_limb_verbs_are_not_borrowed_from_other_body_parts():
    result = observations_from_text("무릎을 굽힌 채 손을 낮게 벌리고 팔을 앞으로 뻗는다")
    assert result["knees"] == "bent" and "legs" not in result
    assert observations_from_text("다리를 벌리고 두 팔을 위로 올린다").get("knees") is None


def test_scene_alone_never_unlocks_body_search():
    index = TagSemanticSearcher([record("p", "", actions=[], activities=["athletics"])],
                                min_score=0)
    assert index.search(SemanticQuery(activities=["athletics"])).hits == []
    assert index.search(SemanticQuery(actions=["jump"]), top_k=0).hits == []


def test_body_only_minimum_is_configurable_and_shared_with_query():
    desc = PoseDescription(
        body_details="두 무릎을 굽혀 다리를 넓게 벌린 자세",
        search_text="두 무릎을 굽혀 다리를 넓게 벌린 자세", certainty="medium",
        body_observations={"knees": "bent", "legs": "apart"})
    config = RoutingConfig(semantic_min_body_observations=3)
    assert not assess_semantic_evidence(desc, config).valid
    assert config.semantic_thresholds()["min_body_observations"] == 3
    index = TagSemanticSearcher([
        record("p", "양 무릎을 굽혀 다리를 넓게 벌린 자세", actions=[])])
    query = SemanticQuery.from_description(desc, min_body_matches=3)
    assert index.search(query).status == "no_result"


def test_describe_prompt_is_versioned_and_requests_observation_not_coordinates():
    prompt = prompts.pose_describe_prompt([(0, 0, 1, 1)])
    assert prompts.POSE_DESCRIBE_VERSION == "d2-observed-body"
    assert '"body_observations"' in prompt and '"context_evidence"' in prompt
    assert "jump와 fall을 동시에" in prompt and "좌표·각도는 출력하지 마라" in prompt
