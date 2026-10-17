"""Build private, executable contract examples, NOT recorded live search results.

Real library/generated BVH bytes must not be committed to the public repository.
The caller supplies authorized local files; this script makes no cloud/VLM calls.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.models import CutResultOut
from api.pose_presentation import PreviewAssetOut, build_pose_presentation
from src.bvh import parse_bvh, fk, coco17_from_fk


def digest(data):
    return hashlib.sha256(data).hexdigest()


def make_examples(library, generated, output):
    library, generated, output = map(lambda p: Path(p).resolve(), (library, generated, output))
    if output == ROOT or ROOT in output.parents:
        raise ValueError('private_BVH_examples_must_be_outside_repository')
    payloads = [p.read_bytes() for p in (library, generated)]
    for path in (library, generated):
        joints, frames = parse_bvh(str(path))
        points, scores = coco17_from_fk(joints, fk(joints, frames[0]))
        if len(joints) < 15 or not all(scores[i] > 0 for i in (5,6,11,12)):
            raise ValueError('example_requires_real_mapped_humanoid_BVH')
    text = payloads[1].decode('utf-8')
    if text.encode('utf-8') != payloads[1]:
        raise ValueError('generated_BVH_must_roundtrip_as_UTF8')
    pid = library.stem
    sha = [digest(p) for p in payloads]
    from urllib.parse import quote
    def resolve(pose_id):
        if pose_id != pid:
            raise ValueError('unknown_fixture_pose')
        return PreviewAssetOut(transport='library_bvh', bvh_url=f'/pose/{quote(pid,safe="")}/bvh',
            bvh_sha256=sha[0], library_lookup_supported=True)
    def row(**extra):
        return dict(pose_id=pid, family=pid, rank_score=.6/61+.4/62, evidence={},
            camera=dict(yaw=45.,pitch=0.,distance=4.), refine_allowed=False,
            release_allowed=False, actual_mesh_validated=False, **extra)
    def report(status='ok', **extra):
        return dict(version='fullbody-dynamic-v1', route='hybrid', status=status,
            selected=[], ranking='weighted_rrf_k60_geometry_0.4_semantic_0.6', elapsed_ms=0., **extra)
    def generation(status, **extra):
        return dict(status=status, refine_allowed=False, release_allowed=False,
            actual_mesh_validated=False, candidates=[], **extra)
    gen = generation('preview_ready')
    gen['candidates']=[dict(kind='generated_preview', bvh=text, bvh_sha256=sha[1],
        refine_allowed=False, release_allowed=False, actual_mesh_validated=False,
        validation={'fixture_note':'Existing local generated BVH; this example does not rerun or certify generation/skin gates.'})]
    normal = report(); normal['selected']=[row()]
    near = report('no_result', near_candidates=[row(match_level='near_match')],
        generation=generation('not_requested',reason='generation_disabled'))
    generated_report = report('no_result', generation=gen)
    empty = report('no_result', generation=generation('no_result',reason='offline_fixture_successful_empty_generation'))
    error = report('error',reason='geometry_search:ValueError',generation=generation('not_requested',reason='search_error'))
    failed = report('no_result',generation=generation('error',reason='generation_worker_failed'))
    rejected = report('no_result',generation=generation('rejected',reason='generated_observation_conditions_failed'))
    basic = report(basic_model=dict(kind='basic_shoulder_head',template='standin-master-v2',
        output_scope='shoulder_head',status='planned',artifact_status='not_rendered',
        search_match=False,generated_pose=False,refine_allowed=False,release_allowed=False))
    basic['route']='basic_model'

    def person(index, raw, scope='full'):
        presentation=build_pose_presentation(raw,person_index=index,output_scope=scope,resolve_library=resolve)
        return dict(index=index,box=None,tags={},candidates=[],dynamic_search=raw,
            pose_presentation=presentation.model_dump(),output_scope={'detected':scope,'source':'vlm_description'})
    def cut(people):
        raw=dict(route='core',count_confidence='high',detector_count=len(people),vlm_count=len(people),
            people=people,notes=['OFFLINE CONTRACT FIXTURE: serialized by real API models; not a recorded Gemini/search request.'],
            image={'width':512,'height':768},inference_metadata=dict(deployment_version='offline-contract-fixture',
                vlm_provider='not_called',vlm_model='not_called',pose_backend='fixture',
                pose_model_version='not_called',pose_library_version='local_file_hash_bound',feature_version=1))
        return CutResultOut.model_validate(raw).model_dump(mode='json')

    cases = {
        '01-library.json':cut([person(0,normal)]),
        '02-near-only.json':cut([person(0,near)]),
        '03-generated.json':cut([person(0,generated_report,'half')]),
        '04-empty.json':cut([person(0,empty)]),
        '05-search-error.json':cut([person(0,error)]),
        '06-generation-error.json':cut([person(0,failed)]),
        '07-generation-rejected.json':cut([person(0,rejected)]),
        '08-mixed-people.json':cut([person(0,normal),person(1,generated_report,'half'),person(2,error)]),
        '09-basic-plan.json':cut([person(0,basic,'shoulder_head')]),
        '10-legacy-null.json':cut([dict(index=0,box=None,tags={},candidates=[])]),
    }
    legacy=deepcopy(cases['10-legacy-null.json']); del legacy['people'][0]['pose_presentation']
    cases['11-legacy-omitted.json']=legacy
    for value in cases.values():
        CutResultOut.model_validate(value)
    output.mkdir(parents=True,exist_ok=True)
    (output/'assets').mkdir(exist_ok=True)
    (output/'assets/library.bvh').write_bytes(payloads[0])
    (output/'assets/generated.bvh').write_bytes(payloads[1])
    for name, value in cases.items():
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'openapi.json').write_text(json.dumps(CutResultOut.model_json_schema(),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'CONTRACT.md').write_text((ROOT/'docs/POSE_PRESENTATION_HANDOFF.md').read_text(encoding='utf-8'),encoding='utf-8')
    (output/'README.txt').write_text(
        'PRIVATE TEAM FIXTURES — do not upload library BVH or inline-BVH JSON to public Git.\n'
        'These are executable offline contract cases, not measured live search results.\n'
        'Library URL is an inference-internal route, not a public link. Fetch only through authenticated BFF.\n'
        'For offline viewer testing map 01-library asset.bvh_url to assets/library.bvh; SHA-256 must match.\n'
        '03-generated contains the exact UTF-8 text of assets/generated.bvh; generated pose_id is null.\n'
        'Read render_camera for orientation/projection; use output_scope; never apply camera twice.\n'
        'Do not enable automatic apply, refine or export. Numeric/skin safety is not certified by these fixtures.\n'
        'Basic closeup is only a plan, not a rendered 3D file. Legacy files test null/omitted handling.\n'
        'Generation no_result case is a synthetic boundary fixture, not a claim the current generator emitted it.\n',encoding='utf-8')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    manifest=dict(version='private-pose-presentation-examples-v1',fixture_kind='offline_contract_not_live_capture',
        source_revision=revision,working_tree_dirty=dirty,library_pose_id=pid,
        assets=dict(library=dict(file='assets/library.bvh',sha256=sha[0]),
                    generated=dict(file='assets/generated.bvh',sha256=sha[1])),
        files={name:digest((output/name).read_bytes()) for name in sorted([
            *cases, 'assets/library.bvh', 'assets/generated.bvh', 'openapi.json', 'CONTRACT.md', 'README.txt'])})
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted([*manifest['files'], 'manifest.json']):
            z.write(output/name,arcname=name)
    return dict(directory=str(output),archive=str(archive),cases=len(cases),assets_sha256=sha)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',required=True,type=Path)
    parser.add_argument('--generated',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(make_examples(args.library,args.generated,args.output),ensure_ascii=False))
