"""Opt-in local frozen-case verification: real IK -> real converter HTTP -> skin.

Inputs are private local assets. No model calls, service deployment or upload.
Results must be outside this checkout; never commit roughs or BVH payloads.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from fastapi.testclient import TestClient
    from converter_api.app import create_app
    from converter_api.registry import CharacterRegistry
    from converter_api.runner import BlenderRunner, RunnerSettings
    from src.hand_skin_screen import HandSkinClient, HandSkinGate
    from src.hand_spacing_repair import GeneratedHandSpacingRepair
    from src.natural_pose import NaturalLanguagePoseGenerator
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('experiment', 'source-bvh', 'rig', 'taxonomy', 'character', 'blender', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--case-id', default='p_rough_192')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.is_relative_to(ROOT):
        parser.error('private result directory must be outside the checkout')
    baseline = json.loads(args.experiment.read_text())
    case = next(c for c in baseline['cases'] if c['test_id'] == args.case_id)
    report = case['arms']['after']['response']['people'][0]['dynamic_search']
    generation = deepcopy(report['generation'])
    source = deepcopy(generation['candidates'][0])
    source['bvh'] = args.source_bvh.read_text(encoding='utf-8')
    source['bvh_sha256'] = hashlib.sha256(source['bvh'].encode()).hexdigest()
    required = report['candidate_validation']['requirements']
    registry = CharacterRegistry(ROOT/'config/characters.example.json',
        environ={'STANDIN_MASTER_V2_URI': str(args.character.resolve())})
    runner = BlenderRunner(RunnerSettings(blender_binary=str(args.blender.resolve()),
        worker_path=ROOT/'converter/worker.py', timeout_seconds=15.))
    app = TestClient(create_app(registry=registry, runner=runner))

    class Response:
        def __init__(self, response):
            self.response, self.status = response, response.status_code
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self, limit): return self.response.content[:limit]

    def opener(request, timeout):
        return Response(app.post('/screen-hand-contact', content=request.data,
            headers={'Content-Type': request.get_header('Content-type')}))

    client = HandSkinClient('http://local-converter', 'standin-master-v2',
        hashlib.sha256(args.character.read_bytes()).hexdigest(), opener=opener)
    gate = HandSkinGate(client, args.rig.parent, {})
    generator = NaturalLanguagePoseGenerator(args.rig, args.taxonomy)
    repair = GeneratedHandSpacingRepair(generator.repair_hand_spacing)
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(case_id=args.case_id, model_calls=0, uploaded=False,
        comparison='frozen_description_proposal_and_original_BVH_not_live_rough_inference', cases=[])
    for name, hands in (('source_phone_fingers', None),
                        ('overridden_relaxed_fingers', {'left': 'relaxed', 'right': 'relaxed'})):
        session = gate.session(hands)
        rows, diagnostic = repair.filter([source], generation, required, session)
        if not rows:
            raise AssertionError(dict(case=name, repair=diagnostic, screen=session.report()))
        [candidate] = rows
        original = session.cache[source['bvh_sha256']]
        assert original['status'] == 'intersection'
        assert candidate['hand_skin_screen']['status'] == 'clear_screen'
        assert candidate['validation']['whole_pose_conditions']['status'] == 'passed'
        assert candidate['validation']['non_arm_channels_preserved'] is True
        assert not candidate['release_allowed'] and not candidate['refine_allowed']
        folder = out/name
        folder.mkdir(exist_ok=True)
        (folder/'pose.bvh').write_text(candidate['bvh'], encoding='utf-8')
        result = dict(case=name, source_bvh_sha256=source['bvh_sha256'],
            result_bvh_sha256=candidate['bvh_sha256'],
            original_intersection_pairs=original['intersection_pairs'],
            repaired_intersection_pairs=candidate['hand_skin_screen']['intersection_pairs'],
            delta_cm=candidate['hand_spacing_repair']['delta_cm'],
            hand_presets=hands, repair=diagnostic, screen=session.report(),
            validation=candidate['validation'], automatic_export_allowed=False)
        (folder/'receipt.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        summary['cases'].append(result)
        print(json.dumps({k: result[k] for k in ('case', 'original_intersection_pairs',
            'repaired_intersection_pairs', 'delta_cm', 'result_bvh_sha256')}, ensure_ascii=False), flush=True)
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print('PASS bounded offline repair, original conditions, real mesh and closed export gates')


if __name__ == '__main__':
    main()
