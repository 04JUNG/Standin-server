import copy
import json
from pathlib import Path

import pytest

from standin_eval.rubric_score import CONTRACT, aggregate, contract_check, score_bundle, score_case
from standin_eval.util import sha256_file, write_json, write_jsonl

FIXTURES = json.loads((CONTRACT / 'contract-examples.json').read_text())['cases']


@pytest.mark.parametrize('fixture', FIXTURES, ids=[f['input']['case_id'] for f in FIXTURES])
def test_handoff_contract(fixture):
    actual = score_case(fixture['input'])
    assert {k: actual[k] for k in fixture['expected']} == fixture['expected']


def test_fixed_denominators():
    records = [f['input'] for f in FIXTURES]
    result = aggregate(records, [r['case_id'] for r in records])
    assert result['total_cases'] == 12
    assert result['eligible_cases'] == 5
    assert result['full_body_mean'] == 70
    assert result['full_body_100'] == {'numerator': 1, 'denominator': 12, 'rate': 1/12}
    assert result['threshold_among_eligible']['numerator'] == 3
    assert result['threshold_among_eligible']['denominator'] == 5
    assert result['fixed_set_threshold']['denominator'] == 12
    assert result['completed_cases'] == 11
    assert result['any_result_rate']['numerator'] == 10
    assert result['all_people_returned_rate']['numerator'] == 8
    assert aggregate(list(reversed(records)), [r['case_id'] for r in records]) == result


@pytest.mark.parametrize('indices', [[], [4, 9], [7, 8]])
def test_no_eligible_cases(indices):
    rows = [FIXTURES[i]['input'] for i in indices]
    result = aggregate(rows, [r['case_id'] for r in rows])
    assert result['full_body_mean'] is None
    assert result['threshold_among_eligible']['rate'] is None
    if not indices:
        assert result['fixed_set_threshold']['rate'] is None


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(returned_people=0),
    lambda r: r.update(result_status='returned_partial'),
    lambda r: r.update(execution_error='unexpected'),
    lambda r: r.update(result_status='execution_error'),
    lambda r: r.update(expected_people=True),
    lambda r: r.update(case_score=100),
    lambda r: r['people'][0].update(chosen_rank=6),
    lambda r: r['people'][0].update(final_image_sha256='not-a-hash'),
    lambda r: r['people'][0]['criteria']['head'].update(reason=' '),
    lambda r: r['people'][0]['criteria']['head'].update(status='unknown'),
    lambda r: r['people'][0]['criteria'].pop('action'),
])
def test_invalid_record(mutation):
    r = copy.deepcopy(FIXTURES[0]['input'])
    mutation(r)
    with pytest.raises(ValueError):
        score_case(r)


def test_duplicate_people():
    r = copy.deepcopy(FIXTURES[10]['input'])
    r['people'][1]['reference_person_id'] = r['people'][0]['reference_person_id']
    with pytest.raises(ValueError, match='duplicate'):
        score_case(r)


def test_missing_duplicate_mixed_runs():
    r = copy.deepcopy(FIXTURES[0]['input'])
    with pytest.raises(ValueError):
        aggregate([r], [r['case_id'], 'missing'])
    with pytest.raises(ValueError):
        aggregate([r, r], [r['case_id']])
    other = copy.deepcopy(FIXTURES[1]['input'])
    other['run_id'] = 'different'
    with pytest.raises(ValueError, match='mixed'):
        aggregate([r, other], [r['case_id'], other['case_id']])


def make_bundle(tmp_path):
    root = tmp_path / 'assets'
    root.mkdir()
    (root / 'image.png').write_bytes(b'fixed-test-image')
    r = copy.deepcopy(FIXTURES[1]['input'])
    p = r['people'][0]
    p.update(final_image='image.png', final_image_sha256=sha256_file(root / 'image.png'))
    manifest = {'dataset_version': 'test-v1', 'cases': [{
        'case_id': r['case_id'], 'input_image': 'image.png', 'input_sha256': p['final_image_sha256'],
        'reference_person_ids': ['P1'], 'response_completed': True,
        'candidates': [{'reference_person_id': 'P1', 'candidate_id': p['candidate_id'], 'rank': 1,
                        'final_image': p['final_image'], 'final_image_sha256': p['final_image_sha256'],
                        'refine_status': p['refine_status']}]}]}
    config = {'run_id': r['run_id'], 'mode': 'replay_review', 'review_origin': 'ai_assisted_unreviewed',
              'threshold': 90, 'official_comparison': False}
    return root, r, manifest, config


def run_bundle(tmp_path, root, record, manifest, config):
    write_json(tmp_path / 'manifest.json', manifest)
    write_json(tmp_path / 'config.json', config)
    write_jsonl(tmp_path / 'review.jsonl', [record])
    return score_bundle(tmp_path / 'manifest.json', tmp_path / 'review.jsonl', tmp_path / 'config.json',
                        tmp_path / 'result', root)


def test_bound_offline_scoring_and_html_escaping(tmp_path):
    root, r, m, c = make_bundle(tmp_path)
    r['people'][0]['criteria']['head']['reason'] = '<script>alert(1)</script>'
    result = run_bundle(tmp_path, root, r, m, c)
    assert result['full_body_mean'] == 95
    assert result['review_origin'] == 'ai_assisted_unreviewed'
    assert not result['comparison_ready']
    assert len(result['pending_run_config']) == 5
    report = (tmp_path / 'result/report.html').read_text()
    assert '<script>' not in report
    assert '&lt;script&gt;' in report
    with pytest.raises(FileExistsError):
        run_bundle(tmp_path, root, r, m, c)


@pytest.mark.parametrize('mutation', [
    lambda r, m, c: m['cases'][0].update(input_sha256='0'*64),
    lambda r, m, c: m['cases'][0].update(input_image='../escape.png'),
    lambda r, m, c: m['cases'][0].update(reference_person_ids=['other']),
    lambda r, m, c: m['cases'][0]['candidates'][0].update(rank=2),
    lambda r, m, c: m['cases'][0]['candidates'][0].update(final_image_sha256='0'*64),
    lambda r, m, c: m['cases'][0]['candidates'].append(m['cases'][0]['candidates'][0].copy()),
    lambda r, m, c: c.update(official_comparison=True),
    lambda r, m, c: c.update(run_id='wrong'),
    lambda r, m, c: c.update(threshold=float('nan')),
    lambda r, m, c: c.update(manual_pose_edit_allowed=True),
])
def test_bundle_rejects_unbound_or_unsafe_inputs(tmp_path, mutation):
    root, r, m, c = make_bundle(tmp_path)
    mutation(r, m, c)
    with pytest.raises(ValueError):
        run_bundle(tmp_path, root, r, m, c)
    assert not (tmp_path / 'result').exists()


def test_no_result_requires_response_evidence(tmp_path):
    root, r, m, c = make_bundle(tmp_path)
    r.update(result_status='returned_none', returned_people=0, people=[])
    m['cases'][0]['candidates'] = []
    m['cases'][0]['response_completed'] = False
    with pytest.raises(ValueError, match='completed response'):
        run_bundle(tmp_path, root, r, m, c)


def test_contract_cli(tmp_path):
    from standin_eval.cli import main
    assert main(['run', 'rubric-contract-check', '--output', str(tmp_path / 'contract')]) == 0
    result = json.loads((tmp_path / 'contract/summary.json').read_text())
    assert result['synthetic'] is True
    assert result['acceptance_cases_passed'] == 12

@pytest.mark.parametrize('failure', [None, 'timeout', 'invalid_schema'])
def test_ai_judge_to_report_offline(tmp_path, failure):
    from standin_eval.rubric_judge import judge_bundle
    root, r, m, c = make_bundle(tmp_path)
    case = m['cases'][0]
    case['result_status'] = 'returned_all'
    case['selections'] = [{'reference_person_id': 'P1', 'candidate_id': r['people'][0]['candidate_id'],
                            'rank': 1, 'target_description': '원본 중앙의 유일한 인물'}]
    write_json(tmp_path / 'm.json', m)
    write_json(tmp_path / 'c.json', c)
    calls = []
    def fake_judge(prompt, original, final):
        calls.append((prompt, original, final))
        if failure == 'timeout':
            raise TimeoutError('secret-key-must-not-be-logged')
        if failure == 'invalid_schema':
            return {'criteria': {'score': 100}}
        if final is None:
            return {'criteria': {k: {'description': '원본만 해석한 가상 기준', 'visibility': 'observable'}
                                 for k in r['people'][0]['criteria']}, 'actual_model': 'fake-test-only'}
        return {'criteria': r['people'][0]['criteria'], 'actual_model': 'fake-test-only'}
    result = judge_bundle(tmp_path / 'm.json', tmp_path / 'c.json', tmp_path / 'result', root,
                          judge=fake_judge, model='fake-test-only', max_calls=2)
    assert len(calls) == (2 if failure is None else 1)
    assert result['review_origin'] == 'ai_assisted_unreviewed'
    assert result['full_body_mean'] == (95 if failure is None else None)
    assert result['execution_error_rate']['numerator'] == 0
    traces = (tmp_path / 'result/judge-traces.jsonl').read_text()
    assert 'secret-key' not in traces
    assert (tmp_path / 'result.pending/reviews.jsonl').exists()
    assert len(list((tmp_path / 'result/images').iterdir())) == 1


def test_ai_budget_fails_before_call(tmp_path):
    from standin_eval.rubric_judge import judge_bundle
    root, r, m, c = make_bundle(tmp_path)
    case = m['cases'][0]
    case['result_status'] = 'returned_all'
    case['selections'] = [{'reference_person_id': 'P1', 'candidate_id': r['people'][0]['candidate_id'],
                          'rank': 1, 'target_description': '중앙 인물'}]
    write_json(tmp_path / 'm.json', m)
    write_json(tmp_path / 'c.json', c)
    def forbidden(*args):
        pytest.fail('must not call provider')
    with pytest.raises(ValueError, match='budget'):
        judge_bundle(tmp_path / 'm.json', tmp_path / 'c.json', tmp_path / 'result', root,
                     judge=forbidden, model='fake', max_calls=0)
    assert not (tmp_path / 'result.pending').exists()


def packaged(tmp_path, count=50):
    import zipfile
    from standin_eval.rubric_score import RUBRIC
    root, r, m, c = make_bundle(tmp_path)
    cases, records = [], []
    for i in range(count):
        case = copy.deepcopy(m['cases'][0]); case['case_id'] = f'cut-{i:03}'
        record = copy.deepcopy(r); record['case_id'] = case['case_id']
        cases.append(case); records.append(record)
    write_json(root / 'dataset-manifest.json', {'cases': cases})
    write_json(root / 'run-config.json', c)
    write_json(root / 'rubric.json', RUBRIC)
    write_json(root / 'missing-artifacts.json', [])
    write_jsonl(root / 'reviews.jsonl', records)
    write_json(root / 'summary.json', {'full_body_mean': 100})
    (root / 'README.md').write_text('Synthetic package for tests only')
    archive = tmp_path / 'received.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for p in root.iterdir():
            z.write(p, 'handoff/' + p.name)
    return archive


def test_fifty_zip_intake_and_final_review(tmp_path):
    from standin_eval.rubric_package import import_package, review_queue, finalize_review
    from standin_eval.util import read_json, read_jsonl
    archive = packaged(tmp_path)
    received = tmp_path / 'received'
    receipt = import_package(archive, received)
    assert receipt['status'] == 'valid'
    assert receipt['summary_differences']['full_body_mean'] == {'submitted': 100, 'computed': 95}
    assert read_json(received / 'scored/summary.json')['total_cases'] == 50
    queue = tmp_path / 'decisions.jsonl'
    assert review_queue(received / 'scored', queue)['auto_approved'] == 0
    decisions = read_jsonl(queue)
    with pytest.raises(ValueError, match='explicit'):
        finalize_review(received / 'scored', queue, received / 'package', tmp_path / 'final')
    for d in decisions:
        d.update(decision='approve', reviewer='human-test-reviewer', reviewed_at='2026-10-28T15:00:00+09:00')
    replacement = read_jsonl(received / 'scored/reviews.jsonl')[0]
    replacement['people'][0]['criteria']['head'] = {'status': 'match', 'reason': '사람 최종 수정'}
    decisions[0].update(decision='replace', replacement=replacement)
    write_jsonl(queue, decisions)
    result = finalize_review(received / 'scored', queue, received / 'package', tmp_path / 'final')
    assert result['review_origin'] == 'human_reviewed'
    assert result['full_body_mean'] == 95.1
    feedback = read_json(tmp_path / 'final/review-feedback.json')
    assert len(feedback['changes']) == 1
    from standin_eval.rubric_package import audit_judge
    audit = audit_judge(tmp_path / 'final', received / 'scored', tmp_path / 'audit')
    assert audit['judgements'] == 350
    assert len(audit['disagreements']) == 1
    assert read_json(received / 'scored/summary.json')['full_body_mean'] == 95


@pytest.mark.parametrize('bad_name', ['../escape', '/absolute', 'a\\b', 'C:/file', 'a/../file'])
def test_zip_path_rejection(tmp_path, bad_name):
    import zipfile
    from standin_eval.rubric_package import import_package
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(bad_name, 'bad')
    result = import_package(archive, tmp_path / 'output')
    assert result['status'] == 'invalid'
    assert not (tmp_path / 'output/scored').exists()


def test_zip_wrong_count_missing_artifacts_and_link(tmp_path):
    import zipfile, stat
    from standin_eval.rubric_package import import_package
    archive = packaged(tmp_path, count=49)
    assert import_package(archive, tmp_path / 'wrong-count')['status'] == 'invalid'
    with zipfile.ZipFile(archive, 'a') as z:
        info = zipfile.ZipInfo('handoff/link')
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(info, '/etc/passwd')
    result = import_package(archive, tmp_path / 'link')
    assert result['status'] == 'invalid'
    assert 'links' in result['errors'][0]


def test_original_target_uncertainty_cannot_become_match(tmp_path):
    from standin_eval.rubric_judge import judge_bundle
    root, r, m, c = make_bundle(tmp_path)
    case = m['cases'][0]; case['result_status'] = 'returned_all'
    case['selections'] = [{'reference_person_id': 'P1', 'candidate_id': r['people'][0]['candidate_id'], 'rank': 1}]
    write_json(tmp_path / 'm.json', m); write_json(tmp_path / 'c.json', c)
    calls = []
    def judge(prompt, original, final):
        calls.append(final)
        if final is None:
            return {'criteria': {k: {'description': '원본에서 가림', 'visibility': 'unobservable'}
                                 for k in r['people'][0]['criteria']}}
        return {'criteria': {k: {'status': 'match', 'reason': '출력만 보면 좋음'} for k in r['people'][0]['criteria']}}
    result = judge_bundle(tmp_path / 'm.json', tmp_path / 'c.json', tmp_path / 'result', root,
                          judge=judge, model='fake', max_calls=2)
    assert calls[0] is None and calls[1] is not None
    assert result['eligible_cases'] == 0
    assert (tmp_path / 'result/final-review.jsonl').exists()


def test_zip_duplicate_and_declared_gaps(tmp_path):
    import zipfile
    from standin_eval.rubric_package import import_package
    archive = packaged(tmp_path)
    with zipfile.ZipFile(archive, 'a') as z:
        z.writestr('handoff/IMAGE.PNG', b'collision')
    result = import_package(archive, tmp_path / 'duplicate')
    assert result['status'] == 'invalid'
    assert not (tmp_path / 'duplicate/scored').exists()
    root = tmp_path / 'assets'
    write_json(root / 'missing-artifacts.json', [{'case_id': 'cut-001', 'missing': 'output'}])
    archive2 = tmp_path / 'gaps.zip'
    with zipfile.ZipFile(archive2, 'w') as z:
        for p in root.iterdir():
            z.write(p, p.name)
    assert import_package(archive2, tmp_path / 'gaps')['status'] == 'invalid'


def test_stale_final_review_is_rejected(tmp_path):
    from standin_eval.rubric_package import review_queue, finalize_review
    from standin_eval.util import read_jsonl
    root, r, m, c = make_bundle(tmp_path)
    run_bundle(tmp_path, root, r, m, c)
    review_queue(tmp_path / 'result', tmp_path / 'queue.jsonl')
    rows = read_jsonl(tmp_path / 'queue.jsonl')
    rows[0].update(review_sha256='0'*64, decision='approve', reviewer='human', reviewed_at='today')
    write_jsonl(tmp_path / 'queue.jsonl', rows)
    with pytest.raises(ValueError, match='stale'):
        finalize_review(tmp_path / 'result', tmp_path / 'queue.jsonl', root, tmp_path / 'approved')
