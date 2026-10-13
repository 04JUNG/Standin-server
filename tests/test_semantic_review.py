"""사람 검수는 모델 판정과 분리하고, 임시 파일로만 쓰기 테스트한다."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.semantic_review import install_review_routes


class Index:
    docs=[SimpleNamespace(pose_id='allowed')]
    def pose_json(self,pid):
        return {'thumbnail_urls':{'front':'/test-front.jpg','side':'/test-side.jpg'},'tags':{'pose_action':[]}}


def fixture(tmp):
    report=Path(tmp)/'report.json';feedback=Path(tmp)/'feedback.jsonl'
    report.write_text(json.dumps({'rows':[
        {'id':'Q001','category':'test','text':'팔을 드는 자세','top':[['allowed',.8,'팔을 드는 자세'],['excluded',.9,'제외']], 'baseline_top':[['allowed',.6,'팔을 드는 자세']],'final_verdict':'partial','visual_note':'AI만의 메모'},
        {'id':'Q002','category':'test','text':'없는 포즈','top':[], 'baseline_top':[], 'final_verdict':'no_result','visual_note':'없음'}]}))
    app=FastAPI();install_review_routes(app,report,feedback,Index())
    client=TestClient(app);data=client.get('/review-data').json()
    return client,data,report,feedback


def test_data_keeps_ai_separate_and_filters_excluded_poses():
    with tempfile.TemporaryDirectory() as tmp:
        c,d,_,p=fixture(tmp)
        assert len(d['rows'])==2 and d['reviews']=={} and not p.exists()
        assert [p['pose_id'] for p in d['rows'][0]['current']]==['allowed']
        assert d['rows'][0]['ai_verdict']=='partial'


def test_save_reload_and_revision_are_append_only():
    with tempfile.TemporaryDirectory() as tmp:
        c,d,report,p=fixture(tmp);original=report.read_bytes()
        body={'review_id':d['review_id'],'query_id':'Q001','verdict':'match','selected_pose_id':'allowed','note':'사람이 확인한 메모'}
        assert c.post('/review-feedback',json=body).status_code==200
        saved=c.get('/review-data').json()['reviews']['Q001']
        assert saved['note']==body['note'] and saved['selected_rank']==1
        assert saved['source']=='human-review-ui' and report.read_bytes()==original
        body['verdict']='wrong';c.post('/review-feedback',json=body)
        assert len(p.read_text().splitlines())==2
        assert c.get('/review-data').json()['reviews']['Q001']['verdict']=='wrong'


def test_invalid_query_candidate_verdict_and_version_do_not_write():
    with tempfile.TemporaryDirectory() as tmp:
        c,d,_,p=fixture(tmp);valid={'review_id':d['review_id'],'query_id':'Q001','verdict':'match','selected_pose_id':'allowed'}
        for field,value,status in [('query_id','Q999',422),('selected_pose_id','excluded',422),('verdict','unexpected',422),('review_id','old-version',409)]:
            assert c.post('/review-feedback',json={**valid,field:value}).status_code==status
        assert not p.exists()


def test_no_result_can_be_deferred_but_not_marked_match():
    with tempfile.TemporaryDirectory() as tmp:
        c,d,_,p=fixture(tmp);body={'review_id':d['review_id'],'query_id':'Q002','verdict':'match','note':'결과 없음'}
        assert c.post('/review-feedback',json=body).status_code==422
        body['verdict']='deferred';assert c.post('/review-feedback',json=body).status_code==200
        saved=c.get('/review-data').json()['reviews']['Q002'];assert saved['selected_pose_id'] is None


def test_partial_bad_log_does_not_hide_other_reviews():
    with tempfile.TemporaryDirectory() as tmp:
        c,d,_,p=fixture(tmp)
        body={'review_id':d['review_id'],'query_id':'Q001','verdict':'partial','selected_pose_id':'allowed'}
        c.post('/review-feedback',json=body)
        with p.open('a') as stream:
            stream.write('not json\n[]\nnull\n')
        assert c.get('/review-data').json()['reviews']['Q001']['verdict']=='partial'


if __name__=='__main__':
    tests=[v for k,v in list(globals().items()) if k.startswith('test_')]
    for fn in tests:fn();print('PASS',fn.__name__)
    print(f'{len(tests)}/{len(tests)} passed')
