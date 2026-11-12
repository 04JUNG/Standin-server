import copy
from pathlib import Path
import pytest
from standin_eval.rubric_v32 import RUBRIC, score_case, summarize
from standin_eval.util import write_json, write_jsonl, read_jsonl, read_json, sha256_file


def record(scope='full_body'):
    criteria = {k: {'status': 'not_applicable' if k in RUBRIC['scopes'][scope]['inactive_keys'] else 'match',
                    'reason': 'synthetic contract fixture'} for k in RUBRIC['weights']}
    return {'case_id': 'C1', 'rubric_id': RUBRIC['rubric_id'], 'rubric_version': '3.2', 'run_id': 'synthetic-v32',
            'evaluation_scope': scope, 'review_status': 'ai_assisted_unreviewed',
            'result_status': 'returned_all', 'expected_people': 1, 'returned_people': 1, 'execution_error': None,
            'people': [{'reference_person_id': 'P1', 'chosen_rank': 1, 'candidate_id': 'candidate',
                        'final_image': 'image.png', 'final_image_sha256': '0'*64, 'refine_status': 'timeout',
                        'reviewer': 'fixture', 'criteria': criteria}]}


@pytest.mark.parametrize('scope,part,state,low,high,perfect', [
    ('full_body','head','match',100,100,True),
    ('full_body','head','uncertain',95,100,False),
    ('full_body','head','mismatch',95,95,False),
    ('full_body','action','uncertain',65,100,False),
    ('full_body','head','unobservable',None,None,False),
    ('full_body','head','not_run',None,None,False),
    ('upper_body','head','match',45,45,True),
    ('upper_body','head','uncertain',40,45,False),
    ('upper_body','torso','uncertain',25,45,False),
    ('upper_body','torso','unobservable',None,None,False),
])
def test_scope_intervals(scope, part, state, low, high, perfect):
    r = record(scope);r['people'][0]['criteria'][part]['status'] = state
    c = score_case(r)
    assert (c['score_low'],c['score_high'],c['confirmed_perfect']) == (low,high,perfect)


@pytest.mark.parametrize('second,expected,eligible', [('mismatch',90,True),('uncertain',90,True),('unobservable',None,False),('not_run',None,False)])
def test_multiple_people_same_part(second,expected,eligible):
    r=record();r.update(expected_people=2,returned_people=2)
    r['people'].append(copy.deepcopy(r['people'][0]));r['people'][1]['reference_person_id']='P2'
    r['people'][0]['criteria']['right_arm']['status']='mismatch'
    r['people'][1]['criteria']['right_arm']['status']=second
    c=score_case(r)
    assert c['score_low']==c['score_high']==expected
    assert c['eligible_for_scope_metrics']==eligible
    assert c['uncertain_loss']==0
    if second=='uncertain':
        assert not c['all_active_parts_definitive']


@pytest.mark.parametrize('scope,expected', [('full_body',50),('upper_body',45)])
def test_missing_person_cap(scope,expected):
    r=record(scope);r.update(expected_people=2,result_status='returned_partial')
    assert score_case(r)['exact_score']==expected
    r['people'][0]['criteria']['head']['status']='uncertain'
    assert score_case(r)['exact_score'] is None


@pytest.mark.parametrize('mutation',[
    lambda r:r.update(rubric_version='3.1'),lambda r:r.pop('evaluation_scope'),
    lambda r:r.update(total_score=100),lambda r:r.update(returned_people=True),
    lambda r:r['people'][0]['criteria']['head'].update(reason='  '),
    lambda r:r['people'][0]['criteria']['head'].update(status='not_applicable'),
    lambda r:r['people'][0].update(chosen_rank=6),
    lambda r:r.update(result_status='execution_error'),
])
def test_invalid_records(mutation):
    r=record();mutation(r)
    with pytest.raises(ValueError):score_case(r)


def test_inactive_cannot_match():
    r=record('upper_body');r['people'][0]['criteria']['left_leg']['status']='match'
    with pytest.raises(ValueError):score_case(r)


def test_missing_artifact_overrides_normal_zero():
    r=record();r.update(people=[],returned_people=0,result_status='returned_none')
    assert score_case(r)['exact_score']==0
    assert score_case(r,blocked_case_ids=['C1'])['exact_score'] is None
    r.update(result_status='missing_artifacts')
    assert score_case(r)['exclusion_reason']=='missing_core_artifacts'


def test_scope_denominators_and_empty():
    a=record();a['people'][0]['criteria']['head']['status']='uncertain'
    b=record('upper_body');b['case_id']='C2'
    _,s=summarize([a,b])
    assert s['scopes']['full_body']['P_confirmed']==1
    assert s['scopes']['full_body']['confirmed_perfect_count']==0
    assert s['scopes']['upper_body']['confirmed_perfect_count']==1
    assert s['scopes']['upper_body']['threshold_value'] is None
    assert s['scopes']['full_body']['P_confirmed_over_N']==.5
    _,s=summarize([])
    assert s['result_return']['rate'] is None
    assert s['scopes']['full_body']['mean_low'] is None


def bundle(tmp_path):
    root=tmp_path/'assets';root.mkdir();(root/'image.png').write_bytes(b'synthetic-image')
    r=record('upper_body');r['people'][0]['final_image_sha256']=sha256_file(root/'image.png')
    p=r['people'][0]
    m={'cases':[{'case_id':'C1','input_image':'image.png','input_sha256':p['final_image_sha256'],
        'reference_person_ids':['P1'],'framing':'01_상반신','evaluation_scope':'upper_body',
        'result_status':'returned_all','response_completed':True,
        'candidates':[{'reference_person_id':'P1','candidate_id':'candidate','rank':1,'final_image':'image.png',
                       'final_image_sha256':p['final_image_sha256'],'refine_status':'timeout'}],
        'selections':[{'reference_person_id':'P1','candidate_id':'candidate','rank':1}]}]}
    c={'run_id':r['run_id'],'rubric_id':RUBRIC['rubric_id'],'rubric_version':'3.2','mode':'replay_review',
       'review_status':'ai_assisted_unreviewed'}
    write_json(tmp_path/'manifest.json',m);write_json(tmp_path/'config.json',c);write_jsonl(tmp_path/'reviews.jsonl',[r])
    return root,r,m,c


def test_v32_automatic_to_final_review(tmp_path):
    from standin_eval.rubric_judge import judge_bundle
    from standin_eval.rubric_package import finalize_review,audit_judge
    root,r,m,c=bundle(tmp_path)
    def fake(prompt,original,final):
        if final is None:
            return {'criteria':{k:{'description':'원본만 해석','visibility':'uncertain' if k=='head' else 'observable'} for k in RUBRIC['weights']}}
        return {'criteria':{k:{'status':'match','reason':'비교 초안'} for k in RUBRIC['weights']}}
    result=judge_bundle(tmp_path/'manifest.json',tmp_path/'config.json',tmp_path/'ai',root,judge=fake,model='fake',max_calls=2)
    assert result['scopes']['upper_body']['mean_low']==40
    assert result['scopes']['upper_body']['mean_high']==45
    assert result['scopes']['full_body']['E']==0
    decisions=read_jsonl(tmp_path/'ai/final-review.jsonl')
    decisions[0].update(decision='approve',reviewer='human',reviewed_at='2026-10-28T18:00:00+09:00')
    write_jsonl(tmp_path/'approve.jsonl',decisions)
    final=finalize_review(tmp_path/'ai',tmp_path/'approve.jsonl',root,tmp_path/'human')
    assert final['human_reviewed']
    assert final['review_status']=='human_reviewed'
    audit=audit_judge(tmp_path/'human',tmp_path/'ai',tmp_path/'audit')
    assert audit['judgements']==4
    assert audit['agreement']==1


def test_scope_cannot_be_changed_to_improve_score(tmp_path):
    from standin_eval.rubric_score import score_bundle
    root,r,m,c=bundle(tmp_path)
    m['cases'][0]['framing']='full-body'
    write_json(tmp_path/'manifest.json',m)
    with pytest.raises(ValueError,match='scope'):
        score_bundle(tmp_path/'manifest.json',tmp_path/'reviews.jsonl',tmp_path/'config.json',tmp_path/'result',root)
