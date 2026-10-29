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


@pytest.mark.parametrize('shards', [1, 3])
def test_live_fifty_cut_pipeline_with_injected_services(tmp_path, monkeypatch, shards):
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
    output=tmp_path/'report'
    if shards == 1:
        output.mkdir()
        assert live.execute(package,output,'abc','fake')
    else:
        from standin_eval.staging_merge import merge_fragments
        fragments=[]
        for index in range(shards):
            fragment=tmp_path/f'shard-{index}'; fragment.mkdir()
            assert live.execute(package,fragment,'abc','fake',shard_index=index,shard_count=shards)
            fragments.append(fragment)
        merge_fragments(fragments,output)
    assert calls=={'source':50,'compare':100,'refine':50}
    current=read_json(output/'result.json')
    assert len(current['arms']['search'])==50
    assert current['summaries']['refine']['scopes']['full_body']['mean_low']==100
    assert read_json(output/'completion.json')['complete']


def test_inline_asset_initializer_uses_available_staging_image():
    from scripts.run_staging_evaluation_ci import evaluation_definition
    value=evaluation_definition({'containerDefinitions':[
        {'name':'inference','image':'current:sha'},
        {'name':'body-assets-init','image':'deleted@digest','command':['python','-c','inline initializer'],
         'environment':[{'name':'BODY_ASSETS_MANIFEST_SHA256','value':'pinned'}]}]})
    init=value['containerDefinitions'][1]
    assert init['image']=='current:sha'
    assert init['command']==['python','-c','inline initializer']
    assert init['environment'][0]['value']=='pinned'


def test_bootstrap_prefers_uploaded_harness_over_image_package(tmp_path):
    import hashlib, os, subprocess, sys, zipfile
    from scripts.run_staging_evaluation_ci import bootstrap
    archive=tmp_path/'source.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('standin_eval/__init__.py','')
        z.writestr('standin_eval/staging_live.py','print("uploaded-harness-selected")')
    command=bootstrap().replace('/tmp/eval-code',str(tmp_path/'unpacked'))
    env={**os.environ,'EVAL_CODE_URL':archive.as_uri(),'EVAL_CODE_SHA256':hashlib.sha256(archive.read_bytes()).hexdigest()}
    completed=subprocess.run([sys.executable,'-c',command],env=env,capture_output=True,text=True,timeout=10)
    assert completed.returncode==0,completed.stderr
    assert completed.stdout.strip()=='uploaded-harness-selected'


def test_product_changes_do_not_reset_evaluation_series(tmp_path):
    from scripts.run_staging_evaluation_ci import evaluator_fingerprint, EVALUATOR_FILES
    root=tmp_path; (root/'standin_eval').mkdir(); (root/'evaluation/rubrics').mkdir(parents=True)
    for name in EVALUATOR_FILES:
        (root/'standin_eval'/name).write_text('fixed evaluator')
    (root/'standin_eval/perspective_search.py').write_text('search-v1')
    before=evaluator_fingerprint(root)
    (root/'standin_eval/perspective_search.py').write_text('search-v2')
    assert evaluator_fingerprint(root)==before
    (root/'standin_eval/rubric_judge.py').write_text('changed evaluator')
    assert evaluator_fingerprint(root)!=before


def test_merge_rejects_missing_or_duplicate_cuts(tmp_path):
    from standin_eval.staging_merge import merge_fragments
    from standin_eval.util import write_json
    folder=tmp_path/'fragment'; folder.mkdir()
    value=result(); write_json(folder/'result.json',value)
    with pytest.raises(ValueError,match='50 disjoint'):
        merge_fragments([folder],tmp_path/'output')
    value['arms']['search']*=50
    write_json(folder/'result.json',value)
    with pytest.raises(ValueError,match='50 disjoint'):
        merge_fragments([folder],tmp_path/'output')


def test_deploy_initializer_preserves_inline_logic_and_pins_sha():
    from scripts.prepare_staging_asset_initializer import prepare
    source={'containerDefinitions':[{'name':'body-assets-init','image':'old',
        'command':['python','-c','pinned code'],'environment':[{'name':'MANIFEST','value':'fixed'}]}]}
    with pytest.raises(ValueError,match='immutable'):
        prepare(copy.deepcopy(source),'mutable:latest')
    image='136889124221.dkr.ecr.ap-northeast-2.amazonaws.com/standin/inference:'+('a'*40)
    updated=prepare(copy.deepcopy(source),image)['containerDefinitions'][0]
    assert updated['image']==image
    assert updated['command']==source['containerDefinitions'][0]['command']
    assert updated['environment']==source['containerDefinitions'][0]['environment']
    workflow=(Path(__file__).resolve().parents[1]/'.github/workflows/deploy.yml').read_text()
    step=workflow.split('- name: Keep staging inline')[1].split('- name: Prepare explicit')[0]
    assert "if: github.ref == 'refs/heads/develop'" in step


def test_numeric_search_camera_keeps_orientation_without_refine_binding():
    import numpy as np
    from standin_eval.staging_live import preview_rotation
    from src.refine_projection import search_rotation
    chosen=candidate({'dynamic_search':{'selected':[{'pose_id':'p',
        'camera':{'yaw':330,'pitch':-30,'distance':4.0}}]},'refine_candidates':[]})
    assert not chosen['_refine_bound']
    assert np.allclose(preview_rotation(chosen),search_rotation(330,-30))
    assert not np.allclose(preview_rotation(chosen),np.eye(3))


def test_failed_run_report_preserves_baseline_and_identifies_cut(tmp_path):
    from scripts.run_staging_evaluation_ci import failure_report
    from standin_eval.util import write_json, write_jsonl
    baseline=tmp_path/'baseline.json'; baseline.write_bytes(b'previous-complete-run')
    folder=tmp_path/'shards/1/report'
    write_json(folder/'errors/CB019-P1.json',{'error_type':'ThumbnailRenderFailed'})
    write_jsonl(folder/'search/judge-traces.jsonl',[{'case_id':'CB036','status':'judge_error','error_type':'ValueError'}])
    failure_report(tmp_path,[{'shard':1,'error_type':'RuntimeError'}])
    assert baseline.read_bytes()==b'previous-complete-run'
    text=(tmp_path/'report/report.md').read_text()
    assert 'CB019-P1' in text and 'CB036' in text and '기준선은 갱신하지 않았습니다' in text
    assert not (tmp_path/'report/result.json').exists()


def test_real_judge_requests_pinned_schema_for_both_phases(monkeypatch,tmp_path):
    import json,types
    from standin_eval.rubric_judge import GeminiRubricJudge,TARGET_SCHEMA,REVIEW_SCHEMA
    from src.vlm.client import GeminiVLMClient
    configs=[]
    def generate(**kwargs):
        configs.append(kwargs['config'])
        return types.SimpleNamespace(text=json.dumps({}),model_version='fake',response_id='id')
    judge=object.__new__(GeminiRubricJudge); judge.model='fake'
    judge.client=types.SimpleNamespace(models=types.SimpleNamespace(generate_content=generate))
    monkeypatch.setattr(GeminiVLMClient,'_to_part',lambda x:'image')
    judge('target',tmp_path/'original.png',None)
    judge('compare',tmp_path/'original.png',tmp_path/'final.png')
    for config, field, allowed in [(configs[0],'visibility',['observable','unobservable','uncertain']),
                                   (configs[1],'status',['match','mismatch','unobservable','uncertain','not_run'])]:
        schema=config.response_json_schema
        assert schema['required']==list(TARGET_SCHEMA['properties'])
        assert schema['additionalProperties'] is False
        for criterion in schema['properties'].values():
            assert criterion['properties'][field]=={'type':'string','enum':allowed}
