"""Portable native ZIP regressions: no Downloads files, models or network required."""
import copy
import shutil
import zipfile

import pytest

from standin_eval.rubric_package import import_package
from standin_eval.rubric_v32 import CONTRACT, RUBRIC, summarize
from standin_eval.util import read_json, write_json, write_jsonl, sha256_file
from tests.test_rubric_v32 import record, bundle


def native_package(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    for name in ('rubric.json', 'review-record.schema.json'):
        shutil.copyfile(CONTRACT / name, root / name)
    (root / 'README.md').write_text('Synthetic native package; not model quality.')
    # Deliberately invalid program: the importer must never execute attachments.
    (root / 'score_reviews.py').write_text('raise RuntimeError("DO NOT EXECUTE")')
    (root / 'image.png').write_bytes(b'synthetic image fixture')
    (root / 'model.fbx').write_bytes(b'synthetic FBX fixture')
    def asset(name):
        return {'path': name, 'sha256': sha256_file(root / name), 'bytes': (root / name).stat().st_size}
    r = record()
    r['people'][0]['final_image_sha256'] = asset('image.png')['sha256']
    candidate = {'reference_person_id': 'P1', 'candidate_id': 'candidate', 'original_rank': 1}
    selection = {**candidate, 'service_person_index': 0, 'final_image': asset('image.png'),
                 'fbx': asset('model.fbx'), 'final_bvh': None, 'refine_status': 'timeout'}
    case = {'case_id': 'C1', 'job_id': 'job1', 'input': asset('image.png'),
            'expected_person_ids': ['P1'], 'expected_people': 1, 'returned_people': 1,
            'missing_expected_person_ids': [], 'final_selections': [selection],
            'source_execution_evidence': 'evidence/C1/execution.json',
            'returned_candidates': [candidate], 'result_status': 'returned_all',
            'framing': 'full_body', 'evaluation_scope': 'full_body'}
    write_json(root / 'dataset-manifest.json', {'cases': [case]})
    write_json(root / 'evidence/C1/execution.json', {'case_id': 'C1', 'job_id': 'job1',
        'job_status': {'status': 'completed', 'error': None}, 'candidates': [candidate]})
    write_json(root / 'evidence/C1/P1-choice.json', {'candidate_id': 'candidate', 'rank': 1})
    write_json(root / 'evidence/C1/P1-refine-receipt.json',
               {'candidate_id': 'candidate', 'rank': 1, 'job_id': 'job1', 'person_index': 0})
    write_json(root / 'evidence/C1/P1-fbx-receipt.json', {'candidate_id': 'candidate',
        'chosen_rank': 1, 'job_id': 'job1', 'fbx_sha256': asset('model.fbx')['sha256']})
    write_json(root / 'run-config.json', {'run_id': r['run_id'], 'rubric_id': RUBRIC['rubric_id'],
        'rubric_version': '3.2', 'mode': 'replay_review', 'review_status': 'ai_assisted_unreviewed',
        'active_contract': {'rubric_sha256': sha256_file(root / 'rubric.json'),
                            'schema_sha256': sha256_file(root / 'review-record.schema.json')}})
    write_json(root / 'missing-artifacts.json', {'core_artifacts_complete': True,
        'blocked_review_case_ids': [], 'entries': [{'severity': 'optional', 'kind': 'metadata'}]})
    write_jsonl(root / 'reviews.jsonl', [r])
    computed, summary = summarize([r])
    summary.update(selected_final_images=1, verified_fbx_count=1, missing_core_artifact_cases=[],
                   refine_timeout_with_valid_fallback_cases=['C1'])
    write_json(root / 'computed-scores.json', computed)
    write_json(root / 'summary.json', summary)
    return root


def seal(root):
    write_json(root / 'artifact-hashes.json', {'files': [
        {'path': p.relative_to(root).as_posix(), 'sha256': sha256_file(p), 'bytes': p.stat().st_size}
        for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'artifact-hashes.json']})


def pack(root, prefix='package/'):
    archive = root.parent / 'fixture.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for p in root.rglob('*'):
            if p.is_file():
                z.write(p, prefix + p.relative_to(root).as_posix())
    return archive


@pytest.mark.parametrize('prefix', ['', 'package/'])
def test_native_zip_roundtrip(tmp_path, prefix):
    root = native_package(tmp_path)
    seal(root)
    result = import_package(pack(root, prefix), tmp_path / 'accepted', expected_count=1)
    assert result['status'] == 'valid', result
    assert result['summary_differences'] == {}
    assert result['computed_scores_equal']
    assert result['optional_missing_artifacts'] == 1
    assert result['source_programs_executed'] is False
    assert (tmp_path / 'accepted/scored/report.html').is_file()


@pytest.mark.parametrize('fault,expected', [
    ('missing', 'coverage'), ('tampered', 'hash mismatch'), ('size', 'byte size'),
    ('choice', 'choice receipt'), ('refine', 'refine receipt'), ('person', 'refine person'),
    ('fbx', 'FBX receipt'), ('execution', 'execution identity'), ('contract', 'schema hash'),
    ('count', 'case count'), ('schema', 'missing fields'), ('traversal', 'escapes root'),
])
def test_native_zip_rejects_invalid_evidence(tmp_path, fault, expected):
    root = native_package(tmp_path)
    seal(root)
    if fault == 'missing':
        (root / 'image.png').unlink()
    elif fault == 'tampered':
        (root / 'image.png').write_bytes(b'altered')
    else:
        path, key, value = {
            'size': ('artifact-hashes.json', None, None),
            'choice': ('evidence/C1/P1-choice.json', 'candidate_id', 'other'),
            'refine': ('evidence/C1/P1-refine-receipt.json', 'job_id', 'other'),
            'person': ('evidence/C1/P1-refine-receipt.json', 'person_index', 1),
            'fbx': ('evidence/C1/P1-fbx-receipt.json', 'fbx_sha256', '0'*64),
            'execution': ('evidence/C1/execution.json', 'job_id', 'other'),
            'contract': ('run-config.json', 'active_contract', {'rubric_sha256': sha256_file(root/'rubric.json'), 'schema_sha256': '0'*64}),
            'count': ('dataset-manifest.json', 'cases', []),
            'schema': ('reviews.jsonl', None, None),
            'traversal': ('dataset-manifest.json', None, None),
        }[fault]
        if fault == 'schema':
            r = record(); r.pop('evaluation_scope')
            write_jsonl(root / path, [r])
        else:
            data = read_json(root / path)
            if fault == 'size':
                data['files'][0]['bytes'] += 1
            elif fault == 'traversal':
                data['cases'][0]['source_execution_evidence'] = '../outside.json'
            else:
                data[key] = value
            write_json(root / path, data)
        if fault != 'size':
            seal(root)  # Valid inventory hashes must not bypass semantic validation.
    output = tmp_path / 'rejected'
    result = import_package(pack(root), output, expected_count=1)
    assert result['status'] == 'invalid', result
    assert expected in ' '.join(result['errors'])
    assert not (output / 'scored').exists()
    assert not (output / 'final-review.jsonl').exists()


def test_native_zip_reports_submitted_score_disagreement(tmp_path):
    root = native_package(tmp_path)
    s = read_json(root / 'summary.json'); s['N'] = 999
    write_json(root / 'summary.json', s)
    write_json(root / 'computed-scores.json', [])
    seal(root)
    result = import_package(pack(root), tmp_path / 'accepted', expected_count=1)
    assert result['status'] == 'valid'
    assert result['summary_differences']['N'] == {'submitted': 999, 'computed': 1}
    assert result['computed_scores_equal'] is False
    assert read_json(tmp_path / 'accepted/scored/summary.json')['N'] == 1


def test_score_cannot_switch_to_another_returned_candidate(tmp_path):
    from standin_eval.rubric_score import score_bundle
    root, r, manifest, config = bundle(tmp_path)
    other = copy.deepcopy(manifest['cases'][0]['candidates'][0])
    other.update(candidate_id='other', rank=2)
    manifest['cases'][0]['candidates'].append(other)
    r['people'][0].update(candidate_id='other', chosen_rank=2)
    write_json(tmp_path / 'manifest.json', manifest)
    write_jsonl(tmp_path / 'reviews.jsonl', [r])
    with pytest.raises(ValueError, match='frozen selections'):
        score_bundle(tmp_path/'manifest.json', tmp_path/'reviews.jsonl', tmp_path/'config.json', tmp_path/'out', root)


def test_native_execution_error_is_preserved(tmp_path):
    from standin_eval.rubric_v32_io import normalize_manifest
    root = native_package(tmp_path)
    m = read_json(root / 'dataset-manifest.json')
    m['cases'][0].update(result_status='execution_error')
    ev = read_json(root / 'evidence/C1/execution.json')
    ev['job_status'] = {'status': 'failed', 'error': 'backend timeout'}
    write_json(root / 'evidence/C1/execution.json', ev)
    assert normalize_manifest(m, root)['cases'][0]['execution_error'] == 'backend timeout'
