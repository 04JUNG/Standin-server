import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from standin_eval.blind_review import summarize,ARMS


def fixture():
    groups=[{'slot':s,'candidates':[{'id':s+str(i)} for i in range(5)]} for s in 'ABC']
    ids=[c['id'] for g in groups for c in g['candidates']]
    public={'study_id':'unit-test-only','public_data_sha256':'fixture','candidate_ids':ids,
        'queries':[{'id':'Q1','failed':False,'groups':groups},{'id':'Q2','failed':True,'groups':[]}]}
    key={'study_id':'unit-test-only','public_data_sha256':'fixture',
         'queries':{q:{'failed':q=='Q2','slots':dict(zip('ABC',ARMS))} for q in ('Q1','Q2')}}
    sub={'study_id':'unit-test-only','public_data_sha256':'fixture','rater':'unit-test-not-a-person',
         'rater_origin':'human-self-report','ratings':{c:'no' for c in ids}}
    return public,key,sub


def test_unrated_never_becomes_a_negative_score():
    p,k,s=fixture();del s['ratings']['A0'];s['complete']=True
    got=summarize(p,k,s);assert got['status']=='incomplete' and got['metrics'] is None


def test_uncertainty_and_failures_are_not_dropped():
    p,k,s=fixture();s['ratings']['A0']='uncertain';s['ratings']['B1']='yes'
    got=summarize(p,k,s)
    assert got['extraction_failures']==1
    a=got['metrics']['anchor_4']['top5'];assert a['n']==2 and a['uncertain']==1 and a['upper_rate']==.5
    b=got['metrics']['anchor_117'];assert b['top1']['confirmed_useful']==0 and b['top5']['confirmed_useful']==1


def test_invalid_or_wrong_study_rejected():
    for change in ({'study_id':'other'},{'public_data_sha256':'other'},{'rater':''},{'rater_origin':'assistant'}):
        p,k,s=fixture();s.update(change)
        try:summarize(p,k,s)
        except ValueError:pass
        else:raise AssertionError('invalid submission accepted')


def test_extra_candidate_cannot_change_metric():
    p,k,s=fixture();s['ratings']['injected']='yes'
    try:summarize(p,k,s)
    except ValueError:pass
    else:raise AssertionError('unknown candidate accepted')


if __name__=='__main__':
    for n,f in sorted(list(globals().items())):
        if n.startswith('test_'):f();print('PASS',n)
