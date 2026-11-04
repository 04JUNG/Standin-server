"""Rubric 3.1 accounting over frozen visual judgements, never inferred from PCK.

The supplied rubric has no calibrated visual decision thresholds. This module
therefore automates validation, deductions and aggregation, not visual truth.
"""
from __future__ import annotations

from collections import Counter
import html
import math
from pathlib import Path
import re
import shutil

from .util import atomic_write_text, hash_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl

CONTRACT = Path(__file__).resolve().parent.parent / 'evaluation/rubrics/standin-v3.1'
RUBRIC = read_json(CONTRACT / 'rubric.json')
WEIGHTS = RUBRIC['weights']
REVIEW_SCHEMA = read_json(CONTRACT / 'review-record.schema.json')
ASSESSED = {'match', 'mismatch'}


def _schema(value, schema, path='review'):
    """Validate the small, bundled schema without a new runtime dependency.

    Supports only the keywords used in this pinned schema, not arbitrary schemas.
    """
    types = schema.get('type', [])
    types = [types] if isinstance(types, str) else types
    matches = {'object': type(value) is dict, 'array': type(value) is list,
               'string': type(value) is str, 'integer': type(value) is int,
               'null': value is None}
    if types and not any(matches[t] for t in types):
        raise ValueError(f'{path}: expected {types}')
    if 'const' in schema and value != schema['const']:
        raise ValueError(f'{path}: wrong rubric')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError(f'{path}: invalid status')
    if isinstance(value, dict):
        props = schema.get('properties', {})
        if set(schema.get('required', [])) - value.keys():
            raise ValueError(f'{path}: missing fields')
        if schema.get('additionalProperties') is False and value.keys() - props.keys():
            raise ValueError(f'{path}: unexpected fields')
        for key in value.keys() & props.keys():
            _schema(value[key], props[key], f'{path}.{key}')
    if isinstance(value, list) and 'items' in schema:
        for i, item in enumerate(value):
            _schema(item, schema['items'], f'{path}[{i}]')
    if isinstance(value, str):
        if len(value.strip()) < schema.get('minLength', 0):
            raise ValueError(f'{path}: empty text')
        if 'pattern' in schema and not re.search(schema['pattern'], value):
            raise ValueError(f'{path}: invalid hash')
    if type(value) is int:
        if value < schema.get('minimum', value) or value > schema.get('maximum', value):
            raise ValueError(f'{path}: out of range')
    for child in schema.get('allOf', []):
        _schema(value, child, path)
    if 'if' in schema:
        try:
            _schema(value, schema['if'], path)
        except ValueError:
            branch = 'else'
        else:
            branch = 'then'
        if branch in schema:
            _schema(value, schema[branch], path)


def score_case(record):
    _schema(record, REVIEW_SCHEMA)
    n, expected, status = record['returned_people'], record['expected_people'], record['result_status']
    people = record['people']
    if n != len(people) or n > expected:
        raise ValueError('returned_people must equal people length and not exceed expected_people')
    ids = [p['reference_person_id'] for p in people]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate reference_person_id')
    valid_count = {'returned_all': n == expected, 'returned_partial': 0 < n < expected,
                   'returned_none': n == 0, 'execution_error': True}
    if not valid_count[status]:
        raise ValueError('result_status conflicts with people count')
    error = record['execution_error']
    if status == 'execution_error':
        if not isinstance(error, str) or not error.strip():
            raise ValueError('execution_error requires a reason')
    elif error is not None:
        raise ValueError('successful execution cannot have execution_error')
    losses = {k: w if any(p['criteria'][k]['status'] == 'mismatch' for p in people) else 0
              for k, w in WEIGHTS.items()}
    diagnostic = 100 - sum(losses.values()) if people and status != 'execution_error' else None
    complete = bool(people) and all(p['criteria'][k]['status'] in ASSESSED for p in people for k in WEIGHTS)
    reasons = sorted({p['criteria'][k]['status'] for p in people for k in WEIGHTS
                      if p['criteria'][k]['status'] not in ASSESSED})
    if status == 'execution_error':
        score, classification = None, 'execution_error'
    elif status == 'returned_none':
        score, classification = 0, 'no_result'
    elif not complete:
        score, classification = None, 'needs_review_or_out_of_scope'
    elif status == 'returned_partial':
        score, classification = min(diagnostic, RUBRIC['partial_people_cap']), 'missing_people'
    else:
        score, classification = diagnostic, 'full_body'
    if status != 'returned_all':
        reasons.append(status)
    eligible = classification == 'full_body'
    return dict(case_id=record['case_id'], diagnostic_score=diagnostic, case_score=score,
                full_body_eligible=eligible, full_body_100=eligible and score == 100,
                classification=classification, deductions=losses, exclusion_reasons=reasons,
                review_sha256=hash_json(record))


def _rate(n, d):
    return dict(numerator=n, denominator=d, rate=n / d if d else None)


def aggregate(records, expected_case_ids, *, threshold=90):
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 100:
        raise ValueError('threshold must be finite and in 0..100')
    ids = [r['case_id'] for r in records]
    if len(expected_case_ids) != len(set(expected_case_ids)) or len(ids) != len(set(ids)):
        raise ValueError('duplicate case IDs')
    if set(ids) != set(expected_case_ids):
        raise ValueError('missing or unexpected review records; fixed denominator cannot shrink')
    if len({r['run_id'] for r in records}) > 1:
        raise ValueError('mixed runs')
    records = sorted(records, key=lambda r: r['case_id'])
    computed = [score_case(r) for r in records]
    eligible = [r for r in computed if r['full_body_eligible']]
    n, e = len(records), len(eligible)
    h = sum(r['full_body_100'] for r in computed)
    p = sum(r['case_score'] >= threshold for r in eligible)
    statuses = Counter(r['result_status'] for r in records)
    parts = {}
    for k in WEIGHTS:
        assessed_people = [person['criteria'][k]['status'] for r in records
                           if r['result_status'] != 'execution_error' for person in r['people']
                           if person['criteria'][k]['status'] in ASSESSED]
        # A cut denominator requires every returned person to be assessed for this part.
        assessed_cuts = [r for r in records if r['result_status'] != 'execution_error' and r['people']
                         and all(person['criteria'][k]['status'] in ASSESSED for person in r['people'])]
        parts[k] = {'people_error_rate': _rate(assessed_people.count('mismatch'), len(assessed_people)),
                    'cut_error_rate': _rate(sum(any(person['criteria'][k]['status'] == 'mismatch'
                                                   for person in r['people']) for r in assessed_cuts), len(assessed_cuts))}
    return dict(rubric_id=RUBRIC['rubric_id'], rubric_version=RUBRIC['rubric_version'],
                rubric_sha256=sha256_file(CONTRACT / 'rubric.json'), threshold=threshold,
                threshold_status='configured_not_official_target', total_cases=n, eligible_cases=e,
                completed_cases=n - statuses['execution_error'],
                full_body_mean=sum(r['case_score'] for r in eligible) / e if e else None,
                full_body_100=_rate(h, n), threshold_among_eligible=_rate(p, e),
                fixed_set_threshold=_rate(p, n), evaluation_coverage=_rate(e, n),
                any_result_rate=_rate(statuses['returned_all'] + statuses['returned_partial'], n),
                all_people_returned_rate=_rate(statuses['returned_all'], n),
                no_result_rate=_rate(statuses['returned_none'], n),
                execution_error_rate=_rate(statuses['execution_error'], n),
                exclusions=dict(Counter(r['classification'] for r in computed if not r['full_body_eligible'])),
                exclusion_reasons=dict(Counter(reason for r in computed for reason in r['exclusion_reasons'])),
                part_errors=parts, cases=computed)


def _verified_file(root, relative, digest):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('artifact must be a relative local path')
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError('artifact escapes root')
    if not re.fullmatch('[a-f0-9]{64}', str(digest)) or sha256_file(path) != digest:
        raise ValueError(f'artifact hash mismatch: {relative}')


def score_bundle(manifest_path, reviews_path, config_path, output, artifacts_root, *, validate_only=False):
    """Offline rescore. Sidecar binds review to fixed inputs and returned candidates."""
    manifest, config = read_json(manifest_path), read_json(config_path)
    if config.get('rubric_version') == '3.2':
        from .rubric_v32_io import score_bundle_v32
        return score_bundle_v32(manifest_path, reviews_path, config_path, output, artifacts_root,
                                validate_only=validate_only)
    records = read_jsonl(reviews_path)
    root, output = Path(artifacts_root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError('use a new output directory to preserve review history')
    if config.get('mode') not in {'replay_review', 'live_e2e'}:
        raise ValueError('mode must identify the captured source run')
    if config.get('review_origin') not in {'human_reviewed', 'ai_assisted_unreviewed'}:
        raise ValueError('explicit review_origin required')
    if not isinstance(config.get('run_id'), str) or not config['run_id'].strip():
        raise ValueError('run_id required')
    if any(r['run_id'] != config['run_id'] for r in records):
        raise ValueError('run_id mismatch')
    if 'threshold' not in config:
        raise ValueError('explicit diagnostic threshold required')
    pending = [k for k in RUBRIC['pending_run_config'] if config.get(k) is None]
    if config.get('official_comparison') and pending:
        raise ValueError(f'unresolved run settings: {pending}')
    for k in ('refine_attempt_limit', 'refine_time_limit_seconds', 'retry_limit', 'timeout_seconds'):
        v = config.get(k)
        if v is not None and (type(v) not in (int, float) or not math.isfinite(v) or v < 0):
            raise ValueError(f'invalid run setting: {k}')
    for k in ('refine_attempt_limit', 'retry_limit'):
        if config.get(k) is not None and type(config[k]) is not int:
            raise ValueError(f'{k} must be an integer')
    if config.get('selection_before_or_after_refine') not in (None, 'before_refine', 'after_refine'):
        raise ValueError('invalid selection timing')
    if config.get('official_comparison'):
        if config['review_origin'] != 'human_reviewed':
            raise ValueError('AI-assisted review is not a confirmed comparison')
        for k in ('pipeline_version', 'model_version', 'library_version', 'prompt_version',
                  'refine_version', 'camera_policy', 'fallback_policy', 'selection_method'):
            if not isinstance(config.get(k), str) or not config[k].strip():
                raise ValueError(f'comparison requires {k}')
    if config.get('manual_pose_edit_allowed', False) is not False:
        raise ValueError('manual pose editing is outside rubric 3.1')
    cases = manifest['cases']
    result = aggregate(records, [c['case_id'] for c in cases], threshold=config['threshold'])
    by_id = {c['case_id']: c for c in cases}
    for r in records:
        c = by_id[r['case_id']]
        if 'selections' in c:
            selected_keys = [(p['reference_person_id'], p['candidate_id'], p['rank']) for p in c['selections']]
            reviewed_keys = [(p['reference_person_id'], p['candidate_id'], p['chosen_rank']) for p in r['people']]
            if sorted(selected_keys) != sorted(reviewed_keys):
                raise ValueError('review differs from frozen selections')
        if c.get('result_status') is not None and c['result_status'] != r['result_status']:
            raise ValueError('review result_status conflicts with frozen product status')
        _verified_file(root, c['input_image'], c['input_sha256'])
        refs = c['reference_person_ids']
        if len(refs) != len(set(refs)) or len(refs) != r['expected_people']:
            raise ValueError('expected person identity mismatch')
        if not {p['reference_person_id'] for p in r['people']} <= set(refs):
            raise ValueError('unknown reference person')
        candidates = c['candidates']
        keys = [(x['reference_person_id'], x['rank']) for x in candidates]
        identities = [(x['reference_person_id'], x['candidate_id']) for x in candidates]
        if len(keys) != len(set(keys)) or len(identities) != len(set(identities)):
            raise ValueError('duplicate returned candidate identity or rank')
        if any(type(rank) is not int or not 1 <= rank <= 5 or person not in refs for person, rank in keys):
            raise ValueError('invalid Top-5 rank or candidate person')
        if r['result_status'] == 'returned_none' and (c.get('response_completed') is not True or candidates):
            raise ValueError('no_result requires completed response with no valid candidates')
        for p in r['people']:
            matches = [x for x in candidates if x['reference_person_id'] == p['reference_person_id']
                       and x['candidate_id'] == p['candidate_id'] and x['rank'] == p['chosen_rank']]
            if len(matches) != 1:
                raise ValueError('selected candidate/rank not uniquely present in returned candidates')
            candidate = matches[0]
            if (candidate['final_image'], candidate['final_image_sha256'], candidate['refine_status']) != (
                    p['final_image'], p['final_image_sha256'], p['refine_status']):
                raise ValueError('review does not match frozen final output')
            _verified_file(root, p['final_image'], p['final_image_sha256'])
    result.update(run_id=config['run_id'], mode=config['mode'], review_origin=config['review_origin'],
                  synthetic=config.get('data_origin') == 'synthetic',
                  comparison_ready=bool(config.get('official_comparison')) and not pending,
                  pending_run_config=pending,
                  metric_kind='visual_rubric_score_not_joint_accuracy',
                  manifest_sha256=hash_json(manifest), run_config_sha256=hash_json(config),
                  reviews_sha256=hash_json(sorted(records, key=lambda r: r['case_id'])))
    if validate_only:
        return result
    write_json(output / 'summary.json', result)
    write_json(output / 'run-config.json', config)
    write_json(output / 'dataset-manifest.json', manifest)
    write_jsonl(output / 'reviews.jsonl', records)
    write_jsonl(output / 'computed.jsonl', result['cases'])
    images = {}
    for r in records:
        c = by_id[r['case_id']]
        paths = [c['input_image']] + [p['final_image'] for p in r['people']]
        images[r['case_id']] = []
        for relative in paths:
            source = root / relative
            suffix = source.suffix.lower()
            if suffix not in {'.png', '.jpg', '.jpeg', '.webp'}:
                continue
            target = Path('images') / (sha256_file(source) + suffix)
            (output / 'images').mkdir(exist_ok=True)
            shutil.copyfile(source, output / target)
            images[r['case_id']].append(target.as_posix())
    write_report(output, result, records, images)
    return result


def write_report(output, result, records, images=None):
    e = html.escape
    title = '가상 계약 검증 · 실제 성능 아님' if result.get('synthetic') else 'Standin 3.1 자동 집계'
    rows = []
    for r, computed in zip(sorted(records, key=lambda r: r['case_id']), result['cases']):
        pictures = ''.join(f'<img src="{e(path, quote=True)}" alt="원본 또는 동결된 최종 결과" loading="lazy" style="max-width:220px;max-height:240px">'
                           for path in (images or {}).get(r['case_id'], []))
        reasons = '<br>'.join(e(f"{p['reference_person_id']} · {k}: {v['status']} — {v['reason']}")
                             for p in r['people'] for k, v in p['criteria'].items())
        score = computed['case_score']
        rows.append(f"<tr><td>{e(r['case_id'])}{pictures}</td><td>{score if score is not None else '평가 대상 제외'}</td>"
                    f"<td>{e(computed['classification'])}</td><td><details><summary>인물별 판정 근거</summary>{reasons}</details></td></tr>")
    body = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>{title}</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#182334}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd4dd;padding:12px;text-align:left}}pre{{white-space:pre-wrap}}summary{{cursor:pointer}}</style>
<h1>{title}</h1><p>머리·목 5 / 전체 동작 35 · 배점 합계 100 · 관절 정확도와 별도 지표</p>
<p>판정 출처: {e(result.get('review_origin', 'synthetic'))}. AI 보조 판정은 검수 전 추정입니다. 가림·보류를 정답으로 계산하지 않습니다.</p>
<p>전체 {result['total_cases']}컷 / 전신 평가 가능 {result['eligible_cases']}컷 / 전신 평균 {result['full_body_mean']}점</p>
<p>100점 {result['full_body_100']['numerator']}컷 / 전체 {result['total_cases']}컷. 기준 {result['threshold']}점 이상 {result['fixed_set_threshold']['numerator']}컷.</p>
<p>미확정 실행 설정: {e(', '.join(result.get('pending_run_config', []))) or '없음'}</p>
<table><thead><tr><th>컷</th><th>점수</th><th>상태</th><th>근거</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p><a href="summary.json">분모·제외·실패·부위 오류 전체 지표 JSON</a></p></html>'''
    atomic_write_text(Path(output) / 'report.html', body)


def contract_check(output):
    """Bundled synthetic examples bypass artifact checks only in this isolated mode."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('use a new output directory')
    fixtures = read_json(CONTRACT / 'contract-examples.json')['cases']
    for f in fixtures:
        actual = score_case(f['input'])
        if any(actual[k] != v for k, v in f['expected'].items()):
            raise AssertionError(f['input']['case_id'])
    records = [f['input'] for f in fixtures]
    result = aggregate(records, [r['case_id'] for r in records])
    result.update(synthetic=True, review_origin='synthetic', acceptance_cases_passed=len(fixtures))
    write_json(output / 'summary.json', result)
    write_report(output, result, records)
    return result


def prepare_pose_review(dataset, output):
    """Prepare the sealed recognition set without inventing final-output reviews."""
    from .pose_dataset import load_pose_dataset
    frozen, cuts, persons = load_pose_dataset(dataset)
    output = Path(output)
    if output.exists():
        raise FileExistsError('use a new preparation directory')
    worklist = []
    for cut in cuts:
        people = [p for p in persons if p['cut_id'] == cut['cut_id']]
        worklist.append({'case_id': cut['cut_id'], 'input_image': cut['image_path'],
                         'input_sha256': cut['image_sha256'], 'expected_people': cut['person_count'],
                         'source_gt_status': cut['gt_status'], 'visible_ranges': cut['visible_ranges'],
                         'artifact_status': 'final_selected_output_not_bound',
                         'people': [{'reference_person_id': p['person_id'],
                                     'criteria': {k: {'status': 'not_run', 'reason': '최종 후보 이미지와 시각 판정 필요'}
                                                  for k in WEIGHTS}} for p in people]})
    summary = {'rubric_id': RUBRIC['rubric_id'], 'dataset_seal': frozen['seal_sha256'],
               'cuts': len(cuts), 'people': len(persons), 'part_judgements_pending': len(persons) * 7,
               'status': 'preparation_only_not_a_scored_run',
               'missing': ['frozen Top-5 and selections', 'final output images and hashes',
                           'run protocol', 'seven-part visual judgements']}
    write_jsonl(output / 'worklist.jsonl', worklist)
    write_json(output / 'preparation.json', summary)
    return summary
