"""Run the deployed staging image in an isolated ECS task; never route to production."""
import base64
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

from .http_runner import _multipart, _request_json, _fetch_binary
from .matching import match_people
from .rubric_package import _extract
from .rubric_judge import judge_bundle, GeminiRubricJudge
from .rubric_v32 import RUBRIC
from .staging_compare import report
from .util import read_json, read_jsonl, write_json, write_jsonl, sha256_file, hash_json

TARGET = 'http://127.0.0.1:8000'


def assignments(case, people):
    refs = case['expected_person_ids']
    if len(refs) == len(people) == 1:
        return {refs[0]: people[0]}
    boxes = {b['reference_person_id']: b['box'] for b in case.get('source_person_boxes', [])}
    gt = [{'person_id': p, 'bbox_xyxy': boxes[p]} for p in refs if p in boxes]
    pred = [{'prediction_id': str(i), 'box_xyxy': p.get('box')} for i, p in enumerate(people)]
    matched = match_people(case['case_id'], gt, pred, 'core', 'core')
    return {r['person_id']: people[int(r['prediction_id'])] for r in matched if r['match_status'] == 'matched'}


def candidate(person):
    dynamic = person.get('dynamic_search')
    if dynamic is not None:
        pool = dynamic.get('selected', [])
        if not pool:
            return None
        selected = pool[0]
        if selected.get('kind', 'library_original') != 'library_original':
            return None
        bound = next((c for c in person.get('refine_candidates', []) if c['pose_id'] == selected['pose_id']), None)
        # Do not quietly substitute a legacy candidate when dynamic search is empty.
        return {**selected, **(bound or {}), '_refine_bound': bound is not None,
                'view': (bound or {}).get('view', (selected.get('camera') or {}).get('display_view', 'front'))}
    pool = person.get('candidates', [])
    return {**pool[0], '_refine_bound': True} if pool else None


def refine_payload(person, chosen):
    fields = ('keypoints', 'scores', 'refine_allowed', 'refinable_limbs', 'lower_body_observed',
              'skeleton_state', 'coverage_class', 'slot_origin', 'skeleton_source', 'search_stability',
              'distance_metric', 'confidence_threshold', 'refine_v3')
    return {**{k: person[k] for k in fields if k in person}, 'pose_id': chosen['pose_id'],
            'view': chosen['view'], 'camera': chosen.get('camera'),
            'search_distance': chosen.get('distance'), 'render_thumbnail': False}


def execute(package, output, revision, model, previous=None):
    from src.config import CFG
    from src.refine_thumbnail import ConverterThumbnailRenderer, ThumbnailSource
    native = read_json(package/'dataset-manifest.json')
    if len(native['cases']) != 50:
        raise ValueError('fixed 50 cuts required')
    renderer = ConverterThumbnailRenderer(CFG.thumbnail_converter_url, character_id=CFG.thumbnail_character_id,
        timeout_seconds=60, image_format='png', reuse_library=False)
    arms = {a: {'cases': []} for a in ('search', 'refine')}
    operations = []
    for case in native['cases']:
        cid = case['case_id']
        source = package/case['input']['path']
        if sha256_file(source) != case['input']['sha256']:
            raise ValueError('input hash changed')
        dest = output/'inputs'/f'{cid}.png'; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        ref_boxes = {x['reference_person_id']: x['box'] for x in case.get('source_person_boxes', [])}
        base = {'case_id': cid, 'input_image': dest.relative_to(output).as_posix(), 'input_sha256': sha256_file(dest),
                'reference_person_ids': case['expected_person_ids'], 'framing': case['framing'],
                'evaluation_scope': 'upper_body' if case['framing'] == '01_상반신' else 'full_body',
                'response_completed': False, 'execution_error': None, 'candidates': [], 'selections': []}
        per_arm = {a: copy.deepcopy(base) for a in arms}
        op = {'case_id': cid, 'refine_attempts': 0, 'refine_errors': 0, 'artifact_errors': 0}
        start = time.monotonic()
        try:
            body, content_type = _multipart(source)
            status, _, response = _request_json(TARGET+'/analyze', method='POST', data=body,
                content_type=content_type, timeout=180)
            if status != 200 or not response:
                raise ValueError('analyze failed')
            if response['inference_metadata']['deployment_version'] != revision:
                raise ValueError('deployment revision mismatch')
            write_json(output/'responses'/f'{cid}.json', response)
            assigned = assignments(case, response['people'])
            op.update(returned_slots=len(response['people']), matched_slots=len(assigned), analyze_error=False)
            for a in arms:
                per_arm[a]['response_completed'] = True
            for pid, person in assigned.items():
                chosen = candidate(person)
                if chosen is None:
                    continue
                try:
                    pose_id = chosen['pose_id']
                    from urllib.parse import quote
                    status, raw = _fetch_binary(TARGET+'/pose/'+quote(pose_id, safe='')+'/bvh', timeout=60)
                    if status != 200:
                        raise ValueError('missing BVH')
                    bvh = raw.decode('utf-8')
                    rotation = (chosen.get('camera') or {}).get('rotation')
                    rendered = renderer.render(ThumbnailSource(view=chosen['view'], bvh_text=bvh,
                                               pose_id=pose_id, camera_rotation=rotation))
                    search_image = output/'images'/f'{cid}-{pid}-search.png'
                    search_image.parent.mkdir(exist_ok=True)
                    search_image.write_bytes(rendered.data)
                    final_image = search_image
                    refine_status = 'not_allowed'
                    if chosen['_refine_bound'] and person.get('refine_allowed') and person.get('keypoints'):
                        op['refine_attempts'] += 1
                        try:
                            status, _, refined = _request_json(TARGET+'/refine', method='POST',
                                data=json.dumps(refine_payload(person, chosen)).encode(), content_type='application/json', timeout=180)
                            if status != 200 or not refined:
                                raise ValueError('refine failed')
                            write_json(output/'responses'/f'{cid}-{pid}-refine.json', refined)
                            refine_status = 'timeout' if refined.get('reason') == 'timeout' else 'returned'
                            if refined.get('refined'):
                                if not refined.get('bvh'):
                                    raise ValueError('refined output missing')
                                rendered = renderer.render(ThumbnailSource(view=chosen['view'], bvh_text=refined['bvh'],
                                    pose_id=pose_id, refined=True, camera_rotation=rotation))
                                final_image = output/'images'/f'{cid}-{pid}-refine.png'
                                final_image.write_bytes(rendered.data)
                        except Exception as exc:
                            op['refine_errors'] += 1
                            refine_status = 'failed'  # Report error separately; base is explicit fallback.
                            write_json(output/'errors'/f'{cid}-{pid}-refine.json', {'error_type': type(exc).__name__})
                    for arm, image in [('search', search_image), ('refine', final_image)]:
                        ident = pose_id+':'+chosen['view']
                        per_arm[arm]['candidates'].append({'reference_person_id': pid, 'candidate_id': ident, 'rank': 1,
                            'final_image': image.relative_to(output).as_posix(), 'final_image_sha256': sha256_file(image),
                            'refine_status': 'not_allowed' if arm == 'search' else refine_status})
                        per_arm[arm]['selections'].append({'reference_person_id': pid, 'candidate_id': ident, 'rank': 1,
                            **({'reference_bbox_xyxy': ref_boxes[pid]} if pid in ref_boxes else {})})
                except Exception as exc:
                    op['artifact_errors'] += 1
                    write_json(output/'errors'/f'{cid}-{pid}.json', {'error_type': type(exc).__name__})
        except Exception as exc:
            op['analyze_error'] = True
            for a in arms:
                per_arm[a]['execution_error'] = type(exc).__name__
        op['seconds'] = time.monotonic()-start
        for a in arms:
            row = per_arm[a]; n = len(row['selections'])
            row['result_status'] = ('execution_error' if row['execution_error'] else
                'missing_artifacts' if op['artifact_errors'] else 'returned_none' if not n else
                'returned_all' if n == len(row['reference_person_ids']) else 'returned_partial')
            arms[a]['cases'].append(row)
        operations.append(op)
        write_json(output/'operations.json', operations)
        print(json.dumps({'case': cid, 'completed': len(operations), 'total': 50}), flush=True)
    config = {'run_id': revision, 'rubric_version': '3.2', 'rubric_id': RUBRIC['rubric_id'],
              'mode': 'live_e2e', 'review_status': 'ai_assisted_unreviewed'}
    write_json(output/'config.json', config)
    frozen = (previous or {}).get('targets', [])
    summaries = {}; scored = {}
    for arm, manifest in arms.items():
        write_json(output/f'{arm}-manifest.json', manifest)
        summaries[arm] = judge_bundle(output/f'{arm}-manifest.json', output/'config.json', output/arm, output,
            judge=GeminiRubricJudge(model, 60), model=model, max_calls=120, frozen_targets=frozen)
        new_targets = read_jsonl(output/arm/'targets.jsonl')
        merged = {(t['case_id'],t['reference_person_id']):t for t in frozen}
        merged.update({(t['case_id'],t['reference_person_id']):t for t in new_targets if 'target' in t})
        frozen = list(merged.values())
        scored[arm] = read_json(output/arm/'computed-scores.json')
    current = {'revision': revision, 'contract': {'dataset': os.environ['EVAL_DATA_SHA256'],
               'judge_model': model, 'rubric': hash_json(RUBRIC), 'runner': os.environ['EVAL_CONTRACT_VERSION']},
               'arms': scored, 'targets': frozen, 'operations': {'cuts': len(operations),
               'analyze_errors': sum(r['analyze_error'] for r in operations),
               'artifact_errors': sum(r['artifact_errors'] for r in operations),
               'refine_errors': sum(r['refine_errors'] for r in operations),
               'refine_attempts': sum(r['refine_attempts'] for r in operations),
               'total_seconds': sum(r['seconds'] for r in operations)},
               'summaries': summaries}
    write_json(output/'result.json', current)
    report(output, current, previous)
    # Broken infrastructure/judge runs are diagnostic artifacts, never the next baseline.
    failures = current['operations']['analyze_errors']+current['operations']['artifact_errors']+current['operations']['refine_errors']
    failures += sum(read_json(output/a/'judge-summary.json')['failed_calls'] for a in arms)
    write_json(output/'completion.json', {'complete': failures == 0, 'failures': failures})
    return failures == 0


def main():
    root = Path('/tmp/standin-staging-eval'); root.mkdir(exist_ok=False)
    output = root/'report'; output.mkdir()
    # URLs are presigned and must never appear in logs or reports.
    urllib.request.urlretrieve(os.environ['EVAL_DATA_URL'], root/'data.zip')
    if sha256_file(root/'data.zip') != os.environ['EVAL_DATA_SHA256']:
        raise ValueError('dataset ZIP checksum mismatch')
    _extract(root/'data.zip', root/'dataset')
    manifest = list((root/'dataset').rglob('dataset-manifest.json'))
    if len(manifest) != 1:
        raise ValueError('one dataset manifest required')
    previous = None
    if os.environ.get('EVAL_PREVIOUS_URL'):
        with urllib.request.urlopen(os.environ['EVAL_PREVIOUS_URL'], timeout=60) as f:
            previous = json.load(f)
    env = {k:v for k,v in os.environ.items() if not k.startswith(('EVAL_', 'DISCORD_'))}
    server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'api.app:app', '--host', '127.0.0.1', '--port', '8000'], env=env)
    success = False
    try:
        deadline = time.monotonic()+900
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError('server exited during startup')
            try:
                status, _, health = _request_json(TARGET+'/healthz', timeout=10)
                if status == 200:
                    if health['refine']['deployment_version'] != os.environ['EVAL_REVISION']:
                        raise ValueError('wrong staging image revision')
                    write_json(output/'health.json', health)
                    break
            except OSError:
                pass
            time.sleep(5)
        else:
            raise RuntimeError('staging startup timeout')
        success = execute(manifest[0].parent, output, os.environ['EVAL_REVISION'], os.environ['EVAL_JUDGE_MODEL'], previous)
    except Exception as exc:
        write_json(output/'failure.json', {'error_type': type(exc).__name__})
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill(); server.wait()
        archive = shutil.make_archive(str(root/'report'), 'zip', output)
        request = urllib.request.Request(os.environ['EVAL_RESULT_URL'], data=Path(archive).read_bytes(), method='PUT')
        with urllib.request.urlopen(request, timeout=120) as f:
            f.read()
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
