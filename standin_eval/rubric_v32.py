"""Versioned v3.2 scope/interval arithmetic, adapted from reviewed handoff code.
Source score_reviews.py SHA256: 218e2f9e4ad2a88f9e0f25f7c429552d25f8ded0705544d57a54093e15baf86e.
No attachment code is executed. Scope schema and semantics are validated here.
"""
import copy
from collections import Counter
from pathlib import Path
from .rubric_score import _schema
from .util import read_json
CONTRACT = Path(__file__).resolve().parent.parent / 'evaluation/rubrics/standin-v3.2'
RUBRIC = read_json(CONTRACT / 'rubric.json')
REVIEW_SCHEMA = read_json(CONTRACT / 'review-record.schema.json')

def ratio(a, b):
    return a / b if b else None

def check_semantics(record, rubric):
    schema = copy.deepcopy(REVIEW_SCHEMA)
    schema['properties']['review_status'] = {'enum': ['ai_assisted_unreviewed', 'human_reviewed']}
    _schema(record, schema)
    people = record['people']
    n, expected = record['returned_people'], record['expected_people']
    if n != len(people) or n > expected:
        raise ValueError('person count mismatch')
    ids = [p['reference_person_id'] for p in people]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate person ID')
    status = record['result_status']
    if status not in ('execution_error', 'missing_artifacts'):
        actual = 'returned_none' if n == 0 else 'returned_all' if n == expected else 'returned_partial'
        if status != actual:
            raise ValueError('result/count mismatch')


def score_case(record, rubric=RUBRIC, blocked_case_ids=()):
    check_semantics(record, rubric)
    scope = rubric['scopes'][record['evaluation_scope']]
    keys, maximum = scope['active_keys'], scope['score_max']
    people, rs = record['people'], record['result_status']
    # A criterion costs at most once per cut, regardless of person count.
    mismatch = [k for k in keys if any(p['criteria'][k]['status'] == 'mismatch' for p in people)]
    uncertain = [k for k in keys if k not in mismatch and any(p['criteria'][k]['status'] == 'uncertain' for p in people)]
    unobservable = [k for k in keys if any(p['criteria'][k]['status'] == 'unobservable' for p in people)]
    not_run = [k for k in keys if any(p['criteria'][k]['status'] == 'not_run' for p in people)]
    D = sum(rubric['weights'][k] for k in mismatch)
    U = sum(rubric['weights'][k] for k in uncertain)
    # Even when another person mismatches the same key, any person's unobservable
    # or not_run blocks scope metrics: all expected people must be assessable.
    all_definitive = bool(people) and all(p['criteria'][k]['status'] in ('match', 'mismatch') for p in people for k in keys)
    reason = None
    if record['case_id'] in blocked_case_ids or rs == 'missing_artifacts':
        reason = 'missing_core_artifacts'
    elif rs == 'execution_error':
        reason = 'execution_error'
    elif rs == 'returned_none':
        reason = 'normal_no_result'
    elif rs == 'returned_partial':
        reason = 'missing_expected_people'
    elif unobservable:
        reason = 'unobservable_required_parts'
    elif not_run:
        reason = 'not_run_required_parts'
    eligible = reason is None
    low = high = value = None
    kind = 'not_scored'
    if eligible:
        low, high = maximum - D - U, maximum - D
        kind = 'exact' if low == high else 'interval'
        value = low if low == high else None
    elif reason == 'normal_no_result':
        low = high = value = 0
        kind = 'normal_no_result'
    elif reason == 'missing_expected_people' and all_definitive:
        # Diagnostic only: never enter all-person full/upper metrics.
        low = high = value = min(maximum - D, rubric['partial_people_cap'])
        kind = 'partial_people_capped'
    return {
        'case_id': record['case_id'], 'evaluation_scope': record['evaluation_scope'],
        'score_max': maximum, 'score_low': low, 'score_high': high, 'exact_score': value,
        'score_kind': kind, 'eligible_for_scope_metrics': eligible,
        'exclusion_reason': reason, 'all_active_parts_definitive': all_definitive,
        'confirmed_loss': D, 'uncertain_loss': U,
        'mismatch_keys': mismatch, 'uncertain_keys': uncertain,
        'unobservable_keys': unobservable, 'not_run_keys': not_run,
        'inactive_keys': scope['inactive_keys'],
        'confirmed_perfect': eligible and all_definitive and low == high == maximum,
        'result_status': rs, 'expected_people': record['expected_people'],
        'returned_people': record['returned_people'],
    }

def summarize(records, rubric=RUBRIC, blocked_case_ids=()):
    if len({r['case_id'] for r in records}) != len(records):
        raise ValueError('duplicate case ID')
    if len({r['run_id'] for r in records}) > 1 or len({r['review_status'] for r in records}) > 1:
        raise ValueError('mixed runs or review provenance')
    computed = [score_case(r, rubric, blocked_case_ids) for r in records]
    N = len(records)
    result = {
        'run_id': records[0]['run_id'] if records else None,
        'rubric_id': rubric['rubric_id'], 'rubric_version': rubric['rubric_version'],
        'mode': 'replay_review', 'review_status': records[0]['review_status'] if records else 'ai_assisted_unreviewed',
        'human_reviewed': bool(records) and all(r['review_status'] == 'human_reviewed' for r in records),
        'N': N, 'review_record_count': N, 'rates_are_fractions': True,
        'threshold': rubric['threshold'], 'scopes': {},
        'excluded_primary_counts': dict(Counter(c['exclusion_reason'] for c in computed if not c['eligible_for_scope_metrics'])),
        'excluded_case_ids_by_reason': {},
    }
    for reason in result['excluded_primary_counts']:
        result['excluded_case_ids_by_reason'][reason] = [c['case_id'] for c in computed if c['exclusion_reason'] == reason]
    for name, settings in rubric['scopes'].items():
        rows = [c for c in computed if c['evaluation_scope'] == name and c['eligible_for_scope_metrics']]
        E = len(rows)
        exact = [c for c in rows if c['all_active_parts_definitive']]
        H = sum(c['confirmed_perfect'] for c in rows)
        item = {
            'score_max': settings['score_max'], 'classified_count': sum(c['evaluation_scope'] == name for c in computed),
            'E': E, 'strict_definitive_E': len(exact), 'interval_count': sum(c['score_kind'] == 'interval' for c in rows),
            'case_ids': [c['case_id'] for c in rows],
            'mean_low': ratio(sum(c['score_low'] for c in rows), E),
            'mean_high': ratio(sum(c['score_high'] for c in rows), E),
            'mean_exact': ratio(sum(c['exact_score'] for c in rows), E) if E and all(c['score_kind'] == 'exact' for c in rows) else None,
            'strict_subset_mean': ratio(sum(c['exact_score'] for c in exact), len(exact)),
            'confirmed_perfect_count': H, 'confirmed_perfect_over_N': ratio(H, N),
            'confirmed_perfect_over_E': ratio(H, E),
            'upper_bound_perfect_count': sum(c['score_high'] == settings['score_max'] for c in rows),
        }
        if name == 'full_body':
            T = rubric['threshold']['value']
            P = sum(c['score_low'] >= T for c in rows)
            possible = sum(c['score_high'] >= T for c in rows)
            item.update({'threshold_value': T, 'P_confirmed': P, 'P_possible_inclusive': possible,
                         'P_confirmed_over_E': ratio(P, E), 'P_confirmed_over_N': ratio(P, N),
                         'P_possible_over_E': ratio(possible, E), 'P_possible_over_N': ratio(possible, N)})
        else:
            item.update({'threshold_value': None, 'threshold_note': '45점 척도에 90점 기준이나 자동 환산 합격선을 적용하지 않음.'})
        result['scopes'][name] = item
    for field, predicate in [('result_return', lambda c: c['returned_people'] > 0 and c['result_status'] in ('returned_all','returned_partial')),
                             ('complete_person_return_by_case', lambda c: c['result_status'] == 'returned_all'),
                             ('normal_no_result', lambda c: c['result_status'] == 'returned_none' and c['exclusion_reason'] == 'normal_no_result'),
                             ('execution_error', lambda c: c['result_status'] == 'execution_error')]:
        count = sum(predicate(c) for c in computed)
        result[field] = {'cases': count, 'denominator': N, 'rate': ratio(count, N)}
    count, expected = sum(c['returned_people'] for c in computed), sum(c['expected_people'] for c in computed)
    result['person_return'] = {'people': count, 'expected_people': expected, 'rate': ratio(count, expected)}
    result['scored_scope_cases'] = sum(c['eligible_for_scope_metrics'] for c in computed)
    result['notes'] = [
        'mean_low/high는 보류가 모두 불일치/일치로 해소될 때의 평균 경계다. 중간값 또는 확률 추정이 아니다.',
        '전신 E는 v3.2 범위 평가 가능 수. strict_definitive_E만 전원 7항목 확정된 수다. 두 지표를 혼동하지 않는다.',
        '100점과 45점 척도 평균을 합산하거나 상반신을 전신 100점으로 자동 환산하지 않는다.',
        'uncertain은 논리 범위에 반영한다. unobservable/not_run은 활성 항목이면 범위 집계에서 제외한다.',
        '상한이 만점이어도 보류가 있으면 확정 만점이 아니다. AI 판정은 사람 검수 전이다.',
    ]
    if result['human_reviewed']:
        result['notes'][-1] = '상한이 만점이어도 보류가 있으면 확정 만점이 아니다. 모든 컷에 사람 검수 기록이 있다.'
    return computed, result
