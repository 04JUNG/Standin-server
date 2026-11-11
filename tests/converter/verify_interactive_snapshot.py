"""Local approved asset audit. Blender 5.2; never calls a paid service.

--character /local/approved.fbx --bvh /local/approved.bvh --out /private/tmp/run
All model/source/output bytes stay local. Commit only sanitized evidence JSON.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from converter.protocol import *
from converter.hand_library import load_library
from converter.interactive_adapter import VERSION, prepare_body, solve_from_body, load_body, shape_binding, runtime_revision
from converter.interactive_scene import capture, compare
from converter.worker import load_job, run_job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--character', type=Path, required=True)
    parser.add_argument('--bvh', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shape-recipe', type=Path)
    parser.add_argument('--preset', default='relaxed')
    parser.add_argument('--skip-png', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ['STANDIN_LAB_PREVIEW_TRACE'] = '1'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    recipe = json.loads(args.shape_recipe.read_text()) if args.shape_recipe else None
    def job(name, **extra):
        directory = args.out/name; directory.mkdir()
        source = directory/'input.bvh'; source.write_bytes(args.bvh.read_bytes())
        value = dict(schema_version=JOB_SCHEMA_VERSION, conversion_id=name, temp_dir=str(directory),
            bvh_path=str(source), character_fbx=str(args.character), character_id='local-approved-fixture',
            character_sha256=sha(args.character), output_path=str(directory/'artifact.fbx'),
            report_path=str(directory/'report.json'), frame=FRAME, mirror=False, output_mode=OUTPUT_MODE,
            apply_root_translation=False, embed_textures=False, force_exact_v324=False, thumbnail=None,
            output_scope='full', **extra)
        if recipe: value['shape_recipe'] = recipe
        path = directory/'job.json'; path.write_text(json.dumps(value))
        return path
    eye = [float(r==c) for r in range(4) for c in range(4)]
    identity = dict(ownerBinding='a'*64, selection=dict(operation='local-audit', revision=1, person='fixture', candidate='fixture'),
        source=dict(originalBvhSha256=sha(args.bvh), refinedBvhSha256=None, refineVersion='none'),
        body=dict(characterId='local-approved-fixture', characterSha256=sha(args.character), rigProfile='mixamo'),
        shape=shape_binding(recipe),
        hands=dict(left=None if args.preset == 'source' else args.preset, right=None,
            librarySha256=load_library().sha256, rulesVersion='hand-bake-v1'),
        scope='full', mirror=False,
        camera=dict(projection='orthographic', viewMatrix=eye, projectionMatrix=eye, distanceTorso=4.,
            torsoReferenceSha256=sha(args.character), baked=False),
        solver=dict(version=SOLVER_VERSION, manifestSha256=SOLVER_MANIFEST_SHA256, adapterVersion=VERSION, runtimeSha256=runtime_revision()))
    started = time.monotonic()
    base = prepare_body(job('body'), identity, args.out/'base')
    evidence = dict(schema='interactive-local-audit-v1', characterSha256=sha(args.character),
        sourceBvhSha256=sha(args.bvh), blenderBuild=EXPECTED_BLENDER_BUILD_HASH,
        bodyPrepareMs=(time.monotonic()-started)*1000, initialFbxRequired=True, cases=[])
    from converter.interactive_revision import LatestRevisionGate
    gate = LatestRevisionGate()
    for index, scope in enumerate(['full', 'half', 'bust', 'shoulder_head', 'head']):
        current = copy.deepcopy(identity); current['scope']=scope; current['selection']['revision']=index+1
        started = time.monotonic()
        state = solve_from_body(args.out/'base', current, args.out/f'solved-{scope}', checkpoint=gate.select(current))
        evidence['cases'].append(dict(scope=scope, stateHash=state.state_hash,
            solveSnapshotMs=(time.monotonic()-started)*1000, stateBytes=len(state.canonical_bytes)))
    # Legacy body->bake_hands->FBX validation and PNG baseline. Strict compare
    # against the new scene boundary, including topology/skin/rig and vertices.
    for png in ([False] if args.skip_png else [False, True]):
        path = job('legacy-png' if png else 'legacy-fbx', hands={'left':identity['hands']['left'],'right':None},
                   preview_view='front' if png else None)
        started = time.monotonic(); report=run_job(load_job(path))
        if png:
            from converter import retarget as rt
            rt.reset_scene();rt.import_character(str(path.parent/'artifact.fbx'))
        from converter.solved_state import SolvedPoseState
        expected = SolvedPoseState((args.out/'solved-full/state.canonical.json').read_bytes()).payload['model']
        actual = capture(expected['coordinates']['originBlender'])
        try:
            correspondence = dict(passed=True, **compare(expected, actual))
        except ValueError as exc:
            correspondence = dict(passed=False, reason=str(exc))
        evidence['cases'].append(dict(baseline='png' if png else 'fbx-only',
            wallMs=(time.monotonic()-started)*1000, stages=report.get('lab_stage_timings'),
            correspondence=correspondence, artifactSha256=report['artifact_sha256']))
    (args.out/'evidence.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))
    if any(case.get('correspondence',{}).get('passed') is False for case in evidence['cases']):
        raise ValueError('baseline correspondence failed; evidence is non-authoritative')

if __name__ == '__main__':
    main()
