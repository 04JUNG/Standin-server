"""Opt-in private frozen-proposal replay: real IK/BVH/audit/converter mesh.

No model calls by default. The binding is copied only from a separately saved
observation, never from grip/active_hand. This is declared use intent, not proof
of prop contact. Never commits inputs, BVHs, FBX files or generated receipts.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('experiment', 'rig', 'taxonomy', 'character', 'blender', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--case-id', default='p_rough_279')
    parser.add_argument('--proposal-index', type=int, default=2)
    parser.add_argument('--live-model', help='Optional one fresh generation using the frozen text; requires GEMINI_API_KEY')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.is_relative_to(ROOT) or out.exists():
        parser.error('private output must be a new directory outside the checkout')
    os.environ['POSE_GENERATION_RIG'] = str(args.rig.resolve())
    from fastapi.testclient import TestClient
    from converter_api.app import create_app
    from converter_api.registry import CharacterRegistry
    from converter_api.runner import BlenderRunner, RunnerSettings
    from src.hand_skin_screen import HandSkinClient
    from src.natural_pose import NaturalLanguagePoseGenerator, RIG_SHA256, file_hash
    from src.natural_pose_schema import validate_proposal
    from src.natural_pose_worker import solve_proposal
    from src.pose_requirements import audit_generated
    from src.search_routing import PoseDescription

    if file_hash(args.rig) != RIG_SHA256:
        parser.error('rig hash mismatch')
    baseline = json.loads(args.experiment.read_text())
    case = next(c for c in baseline['cases'] if c['test_id'] == args.case_id)
    report = case['arms']['after']['response']['people'][0]['dynamic_search']
    description = report['description_evidence']['description']
    observed = {o: s for o, s in description.get('held_object_hands', {}).items()
                if s in ('left', 'right', 'both')}
    if not observed:
        parser.error('no explicit independently saved object/hand observation')
    required = deepcopy(report['candidate_validation']['requirements'])
    required['object_hands'] = observed
    raw = deepcopy(report['generation']['diagnostics']['attempts'][args.proposal_index]['proposal'])
    raw['body_parts'].pop('held_object_hands', None)
    after_raw = deepcopy(raw)
    after_raw['body_parts']['held_object_hands'] = observed
    taxonomy = json.loads(args.taxonomy.read_text())
    text = '. '.join(dict.fromkeys(v for v in
        (description.get('search_text'), description.get('body_details')) if v))
    results = [solve_proposal(validate_proposal(p, taxonomy), text) for p in (raw, after_raw)]
    if not all(r['passed'] for r in results):
        raise AssertionError({'unmodified_geometry_failed': [r.get('reasons') for r in results]})
    before, after = (r['candidate'] for r in results)
    before_audit, after_audit = (audit_generated(required, c) for c in (before, after))
    assert before_audit['status'] == 'needs_review'
    assert set(before_audit['unknown']) == {
        key + ':' + obj for obj in observed for key in ('hand_count', 'held_hand')}
    assert after_audit['status'] == 'passed'
    assert before['bvh_sha256'] == after['bvh_sha256']
    assert before['bvh'] == after['bvh']
    assert after_audit['actual_mesh_contact_validated'] is False
    # Empty, unknown and wrong-side intent cannot be promoted to a match.
    wrong_side = {o: 'right' if s == 'left' else 'left' for o, s in observed.items()}
    for bindings in ({}, dict.fromkeys(observed, 'unknown'), wrong_side):
        changed = deepcopy(after)
        changed['meta_json']['body_parts']['held_object_hands'] = bindings
        assert audit_generated(required, changed)['status'] != 'passed'

    registry = CharacterRegistry(ROOT / 'config/characters.example.json',
        environ={'STANDIN_MASTER_V2_URI': str(args.character.resolve())})
    runner = BlenderRunner(RunnerSettings(blender_binary=str(args.blender.resolve()),
        worker_path=ROOT / 'converter/worker.py', timeout_seconds=15.))
    app = TestClient(create_app(registry=registry, runner=runner))

    class Response:
        def __init__(self, response):
            self.response, self.status = response, response.status_code
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit): return self.response.content[:limit]

    def opener(request, timeout):
        return Response(app.post('/screen-hand-contact', content=request.data,
            headers={'Content-Type': request.get_header('Content-type')}))

    client = HandSkinClient('http://local-converter', 'standin-master-v2',
        hashlib.sha256(args.character.read_bytes()).hexdigest(), opener=opener)
    screen = client.check(after['bvh'], None, timeout=15.)
    assert screen['status'] == 'clear_screen', screen
    out.mkdir(parents=True, exist_ok=False)
    (out / 'pose.bvh').write_text(after['bvh'], encoding='utf-8')
    summary = dict(case_id=args.case_id, comparison='frozen_proposal_explicit_observation_binding_only',
        new_image_inference=False, frozen_replay_model_calls=0, uploaded=False,
        observation_binding=observed, geometry_identical=True, bvh_sha256=after['bvh_sha256'],
        before=before_audit, after=after_audit, hand_skin_screen=screen,
        validation=results[1]['validation'], review_required=True,
        actual_mesh_validated=False, prop_contact_validated=False,
        lookup_allowed=False, auto_apply_allowed=False, refine_allowed=False,
        release_allowed=False, export_allowed=False,
        limitations=['Frozen 2D observation and IK proposal, not a fresh image-to-pose accuracy test.',
                     'Object-hand intent is not physical prop contact or load-bearing certification.',
                     'Skin screen covers opposite hands only, not whole-body/prop collisions.'])
    if args.live_model:
        generator = NaturalLanguagePoseGenerator(args.rig, args.taxonomy, model=args.live_model)
        live = generator.generate(PoseDescription(**description))
        summary['fresh_text_generation'] = {k: v for k, v in live.to_dict().items() if k != 'candidates'}
        summary['fresh_generation_uses_model'] = True
        if live.status == 'preview_ready':
            candidate = live.candidates[0]
            live_audit = audit_generated(required, candidate)
            live_screen = client.check(candidate['bvh'], None, timeout=15.)
            summary['fresh_text_generation'].update(object_side_audit=live_audit, hand_skin_screen=live_screen)
            if live_audit['status'] == 'passed' and live_screen['status'] == 'clear_screen':
                (out / 'fresh-generated.bvh').write_text(candidate['bvh'], encoding='utf-8')
        # A fresh LLM failure is still evidence; do not relabel the frozen replay as a live success.
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({'case': args.case_id, 'before': before_audit['status'],
        'after': after_audit['status'], 'geometry_identical': True,
        'skin_status': screen['status'], 'intersection_pairs': screen['intersection_pairs'],
        'fresh_status': summary.get('fresh_text_generation', {}).get('status'), 'out': str(out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
