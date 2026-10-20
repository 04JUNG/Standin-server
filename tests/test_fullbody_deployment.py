"""Staging rollout contract and derived-cache preparation, no cloud/VLM calls."""
import copy
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.prepare_fullbody_deployment import prepare
from scripts.fullbody_preflight_environment import export
from src.fullbody_assets import prepare_assets, digest, validate_generation_assets
from src.pose_contract import apply_bundle_calibration, validate_calibration_against_config, PoseContractError
from src.config import CFG


def task():
    return dict(taskRoleArn='runtime-role', executionRoleArn='execution-role',
        containerDefinitions=[dict(name='inference', image='unchanged-image',
            environment=[dict(name=k, value=v) for k,v in {
                'POSE_MODEL_URI': 's3://assets/models/manifest.json',
                'POSE_LIBRARY_URI': 's3://assets/library.tar.gz',
                'POSE_MODEL_VARIANT': 'cascade', 'POSE_CANARY_STAGE': 'canary-100',
                'VLM_PROVIDER': 'gemini', 'REFINE_THUMBNAIL_CONVERTER_URL': 'http://converter:8001',
                'OTHER_SETTING': 'preserve-me',
            }.items()], secrets=[dict(name='GEMINI_API_KEY', valueFrom='secret-reference')])])


def environment(value):
    return {r['name']:r['value'] for r in value['containerDefinitions'][0]['environment']}


def test_default_cascade_task_is_rendered_as_explicit_fullbody_without_losing_features():
    original=task(); before=copy.deepcopy(original)
    result=prepare(original, enabled='1')
    env=environment(result)
    assert original == before
    assert result['taskRoleArn']==before['taskRoleArn']
    assert result['containerDefinitions'][0]['secrets']==before['containerDefinitions'][0]['secrets']
    assert result['containerDefinitions'][0]['image']=='unchanged-image'
    assert env['POSE_MODEL_VARIANT']=='humanart-m' and env['POSE_BACKEND']=='rtmlib'
    assert env['ROUTING_FULLBODY_ENABLED']==env['ROUTING_PREPARE_ASSETS']==env['POSE_CALIBRATION_FROM_MANIFEST']=='1'
    assert env['OTHER_SETTING']=='preserve-me'
    assert env['REFINE_THUMBNAIL_CONVERTER_URL']=='http://converter:8001'


def test_cold_start_grace_keeps_health_probe_and_retry_policy():
    original=task()
    health=dict(command=['CMD-SHELL','probe'],startPeriod=60,interval=30,timeout=5,retries=3)
    original['containerDefinitions'][0]['healthCheck']=health.copy()
    result=prepare(original,enabled='1')['containerDefinitions'][0]['healthCheck']
    assert result==dict(health,startPeriod=300)


def test_explicit_legacy_switch_preserves_legacy_selection():
    env=environment(prepare(task(), enabled='0', variant='current-x', canary='off'))
    assert env['ROUTING_FULLBODY_ENABLED']=='0' and env['POSE_MODEL_VARIANT']=='current-x'
    assert env['POSE_CANARY_STAGE']=='off' and 'ROUTING_PREPARE_ASSETS' not in env


@pytest.mark.parametrize('variable', ['POSE_MODEL_URI','POSE_LIBRARY_URI','VLM_PROVIDER'])
def test_missing_required_deployment_reference_stops_before_ecs(variable):
    value=task(); rows=value['containerDefinitions'][0]['environment']
    rows[:]=[r for r in rows if r['name']!=variable]
    with pytest.raises(ValueError):prepare(value, enabled='1')


def test_missing_vlm_secret_and_conflicting_rollout_are_not_hidden():
    value=task(); value['containerDefinitions'][0]['secrets']=[]
    with pytest.raises(ValueError, match='real VLM'):prepare(value, enabled='1')
    with pytest.raises(ValueError, match='conflicts'):prepare(task(), enabled='1', variant='cascade')
    with pytest.raises(ValueError, match='canary-100'):prepare(task(), enabled='1', canary='off')
    with pytest.raises(ValueError):prepare(task(), enabled='yes')


def test_preflight_file_never_exports_secret_values_or_references():
    value=prepare(task(), enabled='1')
    value['containerDefinitions'][0]['environment'].append(dict(name='GEMINI_API_KEY',value='do-not-export'))
    text=export(value)
    assert 'POSE_MODEL_VARIANT=humanart-m' in text
    assert 'GEMINI_API_KEY' not in text and 'secret-reference' not in text and 'do-not-export' not in text
    value['containerDefinitions'][0]['environment'].append(dict(name='DATA_DIR',value='/app/data\nINJECT=1'))
    with pytest.raises(ValueError):export(value)


def test_workflow_preflights_rendered_image_before_changing_running_service():
    workflow=(Path(__file__).resolve().parents[1]/'.github/workflows/deploy.yml').read_text()
    assert workflow.index('prepare_fullbody_deployment.py') < workflow.index('id: render')
    assert workflow.index('id: render') < workflow.index('check_fullbody_startup.py')
    assert workflow.index('check_fullbody_startup.py') < workflow.index('aws ecs update-service')
    assert workflow.index('check_fullbody_startup.py') < workflow.index('amazon-ecs-deploy-task-definition@')
    assert "github.ref == 'refs/heads/develop' && '1' || '0'" in workflow
    assert 'POSE_MODEL_VARIANT=${{' not in workflow  # cannot overwrite prepared contract with cascade


@pytest.mark.parametrize('enabled',[False,True])
def test_health_reports_actual_initialized_engine_not_only_an_env_flag(monkeypatch,enabled):
    import api.app as api
    engine=SimpleNamespace(camera=SimpleNamespace(cameras=[None]*117,pose_ids=['a']),
                           semantic=SimpleNamespace(backend='bound-semantic')) if enabled else None
    monkeypatch.setitem(api.STATE,'pipeline',SimpleNamespace(_fullbody_search=engine))
    monkeypatch.setitem(api.STATE,'pose_count',1)
    monkeypatch.setattr(api,'_refine_capability',lambda:{'config_valid':True})
    info=api.healthz()['dynamic_routing']
    assert info['enabled'] is enabled
    assert info['camera_count']==(117 if enabled else 0)
    assert info['camera_pose_count']==(1 if enabled else 0)


def calibration():
    return {k:getattr(CFG,k) for k in ('skeleton_kpt_threshold','min_skeleton_score','distance_metric',
        'fallback_pos_full','fallback_pos_reduced','fallback_angle_full','fallback_angle_reduced',
        'fallback_hybrid_full','fallback_hybrid_reduced')}


def test_manifest_owned_profile_is_whole_atomic_and_still_passes_exact_validation():
    values=calibration(); values['skeleton_kpt_threshold']=.35
    cfg=SimpleNamespace(**calibration())
    bundle=SimpleNamespace(calibration=values)
    apply_bundle_calibration(bundle,cfg)
    validate_calibration_against_config(bundle,cfg)
    assert vars(cfg)==values
    bad=copy.deepcopy(bundle); bad.calibration['min_skeleton_score']=float('nan')
    before=copy.deepcopy(vars(cfg))
    with pytest.raises(PoseContractError):apply_bundle_calibration(bad,cfg)
    assert vars(cfg)==before


@pytest.fixture
def library(tmp_path,monkeypatch):
    from src.library import _synthetic_pose
    from scripts import build_semantic_body_index as body, build_semantic_shape_index as shapes
    root=tmp_path/'library'; root.mkdir(); (root/'bvh').mkdir()
    db=root/'poses.db'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE poses (pose_id TEXT,bvh_path TEXT)')
        conn.executemany('INSERT INTO poses VALUES (?,?)',[('a','data/bvh/a.bvh'),('bad','data/bvh/bad.bvh')])
    (root/'bvh/a.bvh').write_bytes(b'unchanged-test-bvh')
    (root/'search-v2.json').write_text(json.dumps({'poses':[{'pose_id':'a'},{'pose_id':'bad'}]}))
    (root/'search-taxonomy-v2.json').write_text('{}')
    def load(path,frame):
        if not Path(path).is_file():raise FileNotFoundError(path)
        return _synthetic_pose('standing'),np.ones(17)
    monkeypatch.setattr(body,'load_bvh_pose',load)
    monkeypatch.setattr(shapes,'load_bvh_pose',load)
    return root,db


def test_missing_implicit_caches_build_only_with_opt_in_and_never_modify_sources(library):
    root,db=library
    source=[db,root/'bvh/a.bvh',root/'search-v2.json']
    before=[digest(p) for p in source]
    with pytest.raises(ValueError,match='caches missing'):
        prepare_assets(db_path=db,root=root,allowed={'a'},environ={})
    first=prepare_assets(db_path=db,root=root,allowed={'a'},environ={'ROUTING_PREPARE_ASSETS':'1'})
    assert first['receipt']['built'] and first['receipt']['pose_count']==1
    second=prepare_assets(db_path=db,root=root,allowed={'a'},environ={})
    assert not second['receipt']['built'] and second['body']==first['body']
    assert before==[digest(p) for p in source]
    (root/'bvh/a.bvh').write_bytes(b'updated-test-bvh')
    third=prepare_assets(db_path=db,root=root,allowed={'a'},environ={'ROUTING_PREPARE_ASSETS':'1'})
    assert third['receipt']['cache_key']!=first['receipt']['cache_key']


def test_missing_explicit_corrupt_or_incomplete_inputs_fail_closed(library):
    root,db=library
    with pytest.raises(FileNotFoundError):
        prepare_assets(db_path=db,root=root,allowed={'a'},environ={
            'ROUTING_PREPARE_ASSETS':'1','SEMANTIC_285_BODY_INDEX_PATH':str(root/'missing.json')})
    with pytest.raises(FileNotFoundError):
        prepare_assets(db_path=db,root=root,allowed={'a','bad'},environ={'ROUTING_PREPARE_ASSETS':'1'})
    result=prepare_assets(db_path=db,root=root,allowed={'a'},environ={'ROUTING_PREPARE_ASSETS':'1'})
    data=json.loads(result['body'].read_text()); data['poses']['a']['source_sha256']='0'*64
    result['body'].write_text(json.dumps(data))
    with pytest.raises(ValueError,match='hash mismatch'):
        prepare_assets(db_path=db,root=root,allowed={'a'},environ={'ROUTING_PREPARE_ASSETS':'1'})


def test_generation_files_are_checked_without_calling_gemini(tmp_path,monkeypatch):
    import src.natural_pose as natural
    validate_generation_assets(tmp_path,{'POSE_GENERATION_ENABLED':'0'})
    rig=tmp_path/'rig.bvh'; rig.write_bytes(b'test-rig')
    taxonomy=tmp_path/'taxonomy.json'; taxonomy.write_text('{}')
    env={'POSE_GENERATION_RIG':str(rig),'POSE_GENERATION_TAXONOMY':str(taxonomy)}
    with pytest.raises(ValueError,match='rig hash'):validate_generation_assets(tmp_path,env)
    monkeypatch.setattr(natural,'RIG_SHA256',digest(rig))
    validate_generation_assets(tmp_path,env)
