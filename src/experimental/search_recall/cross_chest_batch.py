"""Cross-chest recovery from a partial arm chain."""

from . import service_context as arm
from src.semantic_shapes import cross_chest_hint

def replay(engine, person):
    previous = person['dynamic_search']
    joint = arm.JointEvidence(**previous['geometry_evidence'])
    semantic = arm.SemanticEvidence(**previous['description_evidence'])
    scope = person['output_scope']
    evidence = dict(keypoints=person['keypoints'],
                    joint_mask=person['quality_trace']['search_valid_joint_mask'],
                    coverage_class=person['coverage_class'], output_scope=scope['detected'],
                    output_scope_source=scope['source'],
                    hand_presets={side: person['hands'][side]['preset'] for side in ('left', 'right')})
    if not (joint.sufficient and not joint.policy_block and cross_chest_hint(semantic, evidence)):
        return None
    report = engine.search(arm.decide_route(joint, semantic), evidence,
                           review_geometry=arm.integrated_enabled(arm.CFG))
    geometry = report['branches']['geometry']
    screen = report.get('hand_skin_screen') or {}
    if report['status'] in ('error', 'partial_error') or screen.get('unverified'):
        raise RuntimeError((person['index'], report['status'], report['reason'], screen.get('unverified')))
    assert geometry.get('cross_chest_hint') and report['branches']['semantic']['status'] == 'not_run'
    person['dynamic_search'] = report
    person['pose_presentation'] = arm.build_pose_presentation(
        report, person_index=person['index'], output_scope=scope['mesh_scope'],
        resolve_library=arm._library_preview_asset).model_dump(mode='json')
    return report
