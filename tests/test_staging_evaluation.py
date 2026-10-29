import copy
import importlib.util
from pathlib import Path
import pytest
from standin_eval.staging_compare import compare, report
from standin_eval.staging_live import assignments, candidate, refine_payload
from standin_eval.rubric_v32 import score_case
from tests.test_rubric_v32 import record


def result(state='match', scope='full_body'):
    row = record(scope); row['people'][0]['criteria']['head']['status'] = state
    return {'revision':'abc', 'contract':{'fixed':'v1'}, 'arms':{'search':[score_case(row)],'refine':[score_case(row)]}}


@pytest.mark.parametrize('old,new,label', [('mismatch','match','improved'),('match','mismatch','regressed'),
    ('uncertain','match','overlapping_uncertainty'),('match','match','unchanged'),
    ('unobservable','match','became_evaluable'),('match','not_run','became_excluded')])
def test_comparison_keeps_uncertainty_and_exclusion(old,new,label):
    value = compare(result(new),result(old))
    assert value['arms']['search']['changes'][0]['change'] == label
    expected = 0 if old=='unobservable' or new=='not_run' else 1
    assert value['arms']['search']['scopes']['full_body']['paired'] == expected


def test_contract_drift_is_not_improvement():
    before = result(); after = result(); after['contract']['fixed']='v2'
    with pytest.raises(ValueError,match='contract changed'): compare(after,before)


def test_first_run_baseline_report(tmp_path):
    value = report(tmp_path,result())
    assert value['status']=='baseline_created'
    assert (tmp_path/'report.html').is_file()


def test_dynamic_no_result_cannot_fall_back_to_legacy():
    assert candidate({'dynamic_search':{'selected':[]},'candidates':[{'pose_id':'legacy'}]}) is None


def test_dynamic_refine_requires_same_pose_binding():
    p={'dynamic_search':{'selected':[{'pose_id':'chosen'}]},'refine_candidates':[{'pose_id':'other'}]}
    assert candidate(p)['_refine_bound'] is False
    p['refine_candidates']=[{'pose_id':'chosen','view':'side','distance':.1}]
    assert candidate(p)['_refine_bound'] is True


def test_person_mapping_uses_boxes_not_response_order():
    c={'case_id':'C','expected_person_ids':['a','b'],'source_person_boxes':[
        {'reference_person_id':'a','box':[0,0,10,10]},{'reference_person_id':'b','box':[20,0,30,10]}]}
    people=[{'index':0,'box':[20,0,30,10]},{'index':1,'box':[0,0,10,10]}]
    assert assignments(c,people)['a']['index']==1
    c['source_person_boxes']=[]
    assert assignments(c,people)=={}


def test_refine_preserves_server_permissions():
    value=refine_payload({'refine_allowed':False,'lower_body_observed':False,'refine_v3':{'signed':'value'}},
                         {'pose_id':'p','view':'front','camera':{'rotation':[]},'distance':.5})
    assert value['refine_allowed'] is False
    assert value['lower_body_observed'] is False
    assert value['refine_v3']=={'signed':'value'}


def test_task_definition_is_isolated_and_disables_notifications():
    path=Path(__file__).resolve().parents[1]/'scripts/run_staging_evaluation_ci.py'
    spec=importlib.util.spec_from_file_location('staging_ci',path); module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source={'family':'production','taskDefinitionArn':'secret','cpu':'1024','containerDefinitions':[
        {'name':'inference','image':'immutable','healthCheck':{},'secrets':[{'name':'DISCORD_WEBHOOK_ALERT'},{'name':'GEMINI_API_KEY'}],
         'environment':[{'name':'DISCORD_ALERT_MENTION'},{'name':'DEPLOYMENT_VERSION','value':'abc'}]}]}
    original=copy.deepcopy(source); derived=module.evaluation_definition(source)
    assert source==original
    assert derived['family']=='standin-staging-fixed50-evaluation'
    assert 'taskDefinitionArn' not in derived
    container=derived['containerDefinitions'][0]
    assert container['secrets']==[{'name':'GEMINI_API_KEY'}]
    assert 'healthCheck' not in container
    assert container['image']=='immutable'


def test_workflow_is_develop_only_after_deploy():
    text=(Path(__file__).resolve().parents[1]/'.github/workflows/deploy.yml').read_text()
    step=text.split('- name: Evaluate fixed 50 cuts')[1].split('- name: Publish')[0]
    assert "github.ref == 'refs/heads/develop'" in step
    assert 'STAGING_EVAL_ENABLED' in step
    assert text.index('wait-for-service-stability: true') < text.index('- name: Evaluate fixed 50 cuts')


def test_live_fifty_cut_pipeline_with_injected_services(tmp_path, monkeypatch):
    import types
    from standin_eval import staging_live as live
    from standin_eval.util import write_json, read_json, sha256_file
    import src.refine_thumbnail as thumbnails
    package=tmp_path/'package'; package.mkdir(); (package/'input.png').write_bytes(b'image')
    cases=[{'case_id':f'C{i:03}', 'input':{'path':'input.png','sha256':sha256_file(package/'input.png')},
            'expected_person_ids':[f'P{i}'], 'framing':'full_body'} for i in range(50)]
    write_json(package/'dataset-manifest.json',{'cases':cases})
    calls={'source':0,'compare':0,'refine':0}
    def request(url,**kwargs):
        if url.endswith('/refine'):
            calls['refine']+=1
            return 200,b'',{'refined':True,'bvh':'refined','reason':'ok'}
        return 200,b'',{'inference_metadata':{'deployment_version':'abc'},'people':[
            {'index':0,'refine_allowed':True,'keypoints':[[0,0]]*17,'scores':[1]*17,
             'candidates':[{'pose_id':'p','view':'front','distance':.1}]}]}
    def judge(prompt,source,final):
        keys=('right_arm','left_arm','right_leg','left_leg','torso','head','action')
        if final is None:
            calls['source']+=1
            return {'criteria':{k:{'description':'target','visibility':'observable'} for k in keys}}
        calls['compare']+=1
        return {'criteria':{k:{'reason':'comparison','status':'match'} for k in keys}}
    monkeypatch.setattr(live,'_request_json',request)
    monkeypatch.setattr(live,'_fetch_binary',lambda *a,**k:(200,b'base'))
    monkeypatch.setattr(live,'GeminiRubricJudge',lambda *a:judge)
    monkeypatch.setattr(thumbnails,'ConverterThumbnailRenderer',lambda *a,**k:types.SimpleNamespace(render=lambda s:types.SimpleNamespace(data=b'png')))
    monkeypatch.setenv('EVAL_DATA_SHA256','dataset')
    monkeypatch.setenv('EVAL_CONTRACT_VERSION','v1')
    output=tmp_path/'report'; output.mkdir()
    assert live.execute(package,output,'abc','fake')
    assert calls=={'source':50,'compare':100,'refine':50}
    current=read_json(output/'result.json')
    assert len(current['arms']['search'])==50
    assert current['summaries']['refine']['scopes']['full_body']['mean_low']==100
    assert read_json(output/'completion.json')['complete']
