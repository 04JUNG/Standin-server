import pytest

from src.experimental.search_recall.context_slot_probe import ContextSlots


def slot_index():
    return ContextSlots([
        dict(pose_id=pose_id, groups={'pose_action': [dict(field='action', value=action, label=action)]})
        for pose_id, action in [('throw_side', 'throw'), ('throw_front', 'throw'), ('reach', 'gesture')]
    ])


@pytest.mark.parametrize(('actions', 'certainty', 'fit_ids', 'expected'), [
    (['throw'], 'high', ['throw_side', 'reach'], 'throw_side'),
    (['throw'], 'low', ['throw_side'], None),
    (['throw'], 'high', ['reach'], None),
    (['throw'], 'high', ['throw_side', 'throw_front'], None),
    (['gesture'], 'high', ['reach'], None),
])
def test_specific_action_needs_clear_text_and_pose_support(actions, certainty, fit_ids, expected):
    slots = slot_index()
    description = dict(visible=True, certainty=certainty, visible_scope='full', actions=actions)
    text = {'both': {'top': [('reach', .50), ('throw_side', .42), ('throw_front', .40)]}}
    models = {'original': {'profiles': {'full_xy': {
        'geometry': {'top': [(pose_id, .15) for pose_id in fit_ids]},
    }}}}

    proposal = slots.fallback(description, text, models)

    assert (proposal and proposal['pose_id']) == expected


def test_specific_action_does_not_propose_pose_missing_from_text_results():
    slots = slot_index()
    description = dict(visible=True, certainty='medium', visible_scope='full', actions=['throw'])
    text = {'both': {'top': [('reach', .50)]}}
    models = {'original': {'profiles': {'full_xy': {'geometry': {'top': [('throw_side', .15)]}}}}}

    assert slots.fallback(description, text, models) is None


@pytest.mark.parametrize(('scope', 'details', 'action_supported', 'expected'), [
    ('half', '턱에 손 · 한 팔 내리기', True, 'chin_rest'),
    ('full', '턱에 손', True, None),
    ('half', '턱에 손 · 반대쪽 팔 가려짐', False, None),
    ('half', '턱에 손이 가려져 추정', True, None),
])
def test_visible_contact_needs_an_exact_tag_and_both_clause_rankings(scope, details, action_supported, expected):
    slots = ContextSlots([
        dict(pose_id='chin_rest', groups={'pose_action': [dict(field='body_configuration', label='턱에 손')]}),
        dict(pose_id='head_rest', groups={'pose_action': [dict(field='body_configuration', label='머리에 손')]}),
    ])
    description = dict(visible=True, certainty='medium', visible_scope=scope, body_details=details)
    text = {
        'dense_clauses': {'top': [('head_rest', .51), ('chin_rest', .48)]},
        'dense_action_tags': {'top': [('chin_rest', .46)] if action_supported else [('head_rest', .52)]},
    }

    proposal = slots.fallback(description, text, {})

    assert (proposal and proposal['pose_id']) == expected
