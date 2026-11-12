"""Matched fixed-set comparisons; uncertainty and missingness cannot become gains."""
import html
from pathlib import Path
from .util import write_json, atomic_write_text


def compare(current, previous=None):
    if previous is not None and current['contract'] != previous['contract']:
        raise ValueError('evaluation contract changed; create a new baseline series')
    result = {'status': 'baseline_created' if previous is None else 'compared',
              'revision': current['revision'], 'previous_revision': previous['revision'] if previous else None,
              'contract': current['contract'], 'arms': {}, 'human_verified': False}
    for arm, rows in current['arms'].items():
        old = {r['case_id']: r for r in previous['arms'][arm]} if previous else {}
        if previous and set(old) != {r['case_id'] for r in rows}:
            raise ValueError('fixed case set changed')
        changes, scopes = [], {}
        for row in rows:
            before = old.get(row['case_id'])
            eligible = row['eligible_for_scope_metrics']
            change = 'baseline'
            if before:
                if row['evaluation_scope'] != before['evaluation_scope']:
                    raise ValueError('scope changed')
                was = before['eligible_for_scope_metrics']
                if not was or not eligible:
                    change = 'became_evaluable' if eligible else 'became_excluded' if was else 'both_excluded'
                elif row['score_low'] > before['score_high']:
                    change = 'improved'
                elif row['score_high'] < before['score_low']:
                    change = 'regressed'
                elif row['score_kind'] == before['score_kind'] == 'exact' and row['exact_score'] == before['exact_score']:
                    change = 'unchanged'
                else:
                    change = 'overlapping_uncertainty'
            changes.append({'case_id': row['case_id'], 'scope': row['evaluation_scope'], 'change': change,
                            'before': [before['score_low'], before['score_high']] if before else None,
                            'after': [row['score_low'], row['score_high']], 'exclusion': row['exclusion_reason']})
        for scope in ('full_body', 'upper_body'):
            eligible = [r for r in rows if r['evaluation_scope'] == scope and r['eligible_for_scope_metrics']]
            pairs = [r for r in eligible if r['case_id'] in old and old[r['case_id']]['eligible_for_scope_metrics']]
            def mean(values):
                return sum(values)/len(values) if values else None
            scopes[scope] = {'eligible': len(eligible), 'paired': len(pairs),
                'mean_low': mean([r['score_low'] for r in eligible]), 'mean_high': mean([r['score_high'] for r in eligible]),
                'paired_delta_low': mean([r['score_low']-old[r['case_id']]['score_high'] for r in pairs]),
                'paired_delta_high': mean([r['score_high']-old[r['case_id']]['score_low'] for r in pairs])}
        result['arms'][arm] = {'scopes': scopes, 'changes': changes,
            'counts': {kind: sum(c['change'] == kind for c in changes) for kind in sorted({c['change'] for c in changes})}}
    result['operations'] = current.get('operations', {})
    result['previous_operations'] = previous.get('operations', {}) if previous else None
    return result


def report(output, current, previous=None):
    result = compare(current, previous)
    within = compare({**current, 'arms': {'refine': current['arms']['refine']}},
                     {**current, 'arms': {'refine': current['arms']['search']}})
    result['refine_vs_search_same_run'] = within['arms']['refine']
    output = Path(output)
    write_json(output/'comparison.json', result)
    text = ['# 스테이징 고정 50컷 평가', '', f"상태: {result['status']}",
            f"버전: {result['previous_revision']} → {result['revision']}", '',
            'AI 잠정 판정. 범위가 겹치면 개선/악화를 확정하지 않습니다. Top-5 정답률·메시 안전 인증이 아닙니다.', '',
            '리파인 arm은 해당 실행의 동일 Top-1 후보에 리파인을 적용한 결과입니다. 버전 간 후보가 바뀌면 전체 파이프라인 비교입니다.', '']
    text += ['## 실행 지표', f"이전: {result['previous_operations']}",
             f"현재: {result['operations']}", '', '## 동일 후보 리파인 전후',
             str(within['arms']['refine']['counts']), '']
    for arm, data in result['arms'].items():
        text += [f'## {arm}', str(data['counts']), '', '| 범위 | 평가 가능 | 동일 평가 가능 컷 | 평균 범위 | 대응 컷 변화 범위 |', '|---|---:|---:|---|---|']
        for scope, s in data['scopes'].items():
            text.append(f"| {scope} | {s['eligible']} | {s['paired']} | {s['mean_low']}–{s['mean_high']} | {s['paired_delta_low']}–{s['paired_delta_high']} |")
        text += ['', '| 컷 | 변화 | 이전 | 이후 | 제외 사유 |', '|---|---|---|---|---|']
        for c in data['changes']:
            text.append(f"| {c['case_id']} | {c['change']} | {c['before']} | {c['after']} | {c['exclusion']} |")
        text += ['', f'[원본·출력·항목별 근거]({arm}/report.html)', '']
    markdown = '\n'.join(text)
    atomic_write_text(output/'report.md', markdown)
    links = ''.join(f'<p><a href="{a}/report.html">{a} 이미지·판정</a></p>' for a in result['arms'])
    atomic_write_text(output/'report.html', '<!doctype html><meta charset="utf-8"><title>Staging evaluation</title>'+links+'<pre style="white-space:pre-wrap">'+html.escape(markdown)+'</pre>')
    return result
