#!/usr/bin/env python3
"""Replay phase-1 plans without VLM calls, retrieval, library mutation or promotion."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experimental.rough_router import Observation, POLICY_VERSION, plan_search
from src.vlm.rough_slots import PersonSemantics, parse_slots


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_cases(path):
    protocol = json.loads(path.read_text())
    for case in protocol['cases']:
        for kind in ('image', 'response'):
            if digest(case[kind]) != case[kind + '_sha256']:
                raise ValueError('frozen input changed: ' + case['id'] + ':' + kind)
        response = json.loads(Path(case['response']).read_text())
        index = int(case['id'].rsplit(':p', 1)[1])
        candidates = [p for p in response['people'] if p['index'] == index]
        if len(candidates) != 1:
            raise ValueError('ambiguous frozen person: ' + case['id'])
        source = candidates[0]
        # Existing snapshot has no per-person semantic extension. Do not synthesize it
        # from legacy cut tags or from the reviewer's labels.
        person = PersonSemantics(index, issues=('semantic_extension_missing',))
        obs = Observation(case['keypoints'], case['mask'],
                          ownership_valid=case['owner_verified'] is True,
                          state=source.get('skeleton_state', 'missing'))
        yield case['id'], person, obs, 'frozen_real_rough'


def fixture_cases(path):
    payload = json.loads(path.read_text())
    if payload.get('provenance') != 'synthetic_policy_fixture':
        raise ValueError('fixture input must explicitly declare synthetic_policy_fixture')
    for c in payload['cases']:
        parsed = parse_slots(c['vlm_raw'], c['num_people'])
        person = parsed.get(c['person_index']) or PersonSemantics(c['person_index'], issues=parsed.issues)
        o = c.get('observation', {})
        yield c['id'], person, Observation(o.get('keypoints'), o.get('valid_mask'),
            o.get('ownership_valid', True), o.get('state', 'missing')), payload['provenance']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--protocol',type=Path)
    inputs.add_argument('--fixtures',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    source=args.protocol or args.fixtures
    cases=frozen_cases(source) if args.protocol else fixture_cases(source)
    rows=[]
    for case_id,person,obs,provenance in cases:
        plan=plan_search(person,obs)
        rows.append(dict(case_id=case_id,provenance=provenance,plan_id=plan.plan_id,**plan.to_dict()))
    summary={
        'policy_version':POLICY_VERSION,'created_at':datetime.now(timezone.utc).isoformat(),
        'input_sha256':digest(source),'case_count':len(rows),
        'routes':dict(Counter(r['route'] for r in rows)),
        'semantic_channel_cases':sum(any(c['kind']=='S' for c in r['channels']) for r in rows),
        'missing_extension_cases':sum('semantic_extension_missing' in r['reasons'] for r in rows),
        'actual_semantic_search_executions':0,'actual_compositions':0,
        'retrieval_quality_evaluated':False,'candidates_changed':False,
    }
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'plans.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows))
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (args.out/'REPORT.md').write_text(
        '# 라우터 경로 재현\n\n'
        f'- 사례: {len(rows)}\n- 정책: {POLICY_VERSION}\n'
        f'- 경로 집계: {summary["routes"]}\n'
        f'- 의미 채널 계획이 있는 사례: {summary["semantic_channel_cases"]}\n'
        f'- 확장 의미 슬롯 누락: {summary["missing_extension_cases"]}\n\n'
        '이 실행은 경로 계획만 재현했다. 시맨틱 검색·포즈 조합·카메라 보정은 실행하지 않았다. '
        '기존 실제 러프 기록에는 새 VLM 슬롯이 없어 의미 품질을 평가할 수 없다. '
        'synthetic_policy_fixture는 정책 검증용이며 실제 작가 정확도 증거가 아니다.\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
