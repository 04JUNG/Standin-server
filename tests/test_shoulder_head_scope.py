from src.body_scope import BodyScope, ScopeDetection, parse_person_scopes, refine_scope_from_description, promote_for_observed_legs, mesh_scope
from api.models import ScopeDetectionOut
from converter.framing import retained_roles


def test_explicit_067_description_narrows_bust_without_pose_confidence():
    d = refine_scope_from_description(ScopeDetection(BodyScope.BUST, 'vlm_person'), {
        'visible':False, 'certainty':'low',
        'body_details':'머리와 목, 어깨선 일부만 클로즈업되어 있어 팔과 다리의 배치를 관찰할 수 없다.'})
    assert d == ScopeDetection(BodyScope.SHOULDER_HEAD, 'vlm_description')
    assert ScopeDetectionOut(**d.__dict__).model_dump(mode='json')['detected']=='shoulder_head'
    assert ScopeDetectionOut(**d.__dict__).model_dump(mode='json')['mesh_scope']=='shoulder_head'
    assert mesh_scope(BodyScope.BUST)=='half'


def test_missing_joints_or_generic_upper_body_do_not_invent_a_tight_crop():
    original = ScopeDetection(BodyScope.HALF, 'vlm_person')
    for text in ['상반신만 보이며 팔은 가려짐', '머리를 만지고 어깨에 손을 올림', '']:
        assert refine_scope_from_description(original, {'body_details':text}) == original


def test_observed_legs_and_full_frame_win_over_prose():
    original = ScopeDetection(BodyScope.FULL, 'observed_legs')
    assert refine_scope_from_description(original, {'body_details':'머리와 목 어깨만 클로즈업'}) == original
    assert promote_for_observed_legs(ScopeDetection(BodyScope.SHOULDER_HEAD,'vlm_person'),('left_leg',)).detected == BodyScope.FULL


def test_new_scope_parses_without_shifting_people_and_drops_arm_roles():
    assert parse_person_scopes(['shoulder_head','half','full'],3) == [BodyScope.SHOULDER_HEAD,BodyScope.HALF,BodyScope.FULL]
    roles = retained_roles('shoulder_head')
    assert {'head','neck','shoulder.L','shoulder.R'} <= roles
    assert not roles & {'hips','upleg.L','upperarm.L','forearm.L','hand.L'}


def test_scope_prompt_has_a_new_identity_and_preserves_legacy_templates():
    from src.vlm import prompts
    assert 'shoulder_head' in prompts.user_template('p3-output-scope')
    assert 'shoulder_head' not in prompts.user_template('p1-scope')
    assert prompts.DEFAULT_PROMPT_VERSION == 'p3-output-scope'
