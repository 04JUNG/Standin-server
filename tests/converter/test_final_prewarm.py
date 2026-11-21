"""Automatic default changes, restart/resume, late work and lifecycle contracts."""
from dataclasses import replace
import json
import time
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from converter_api.final_cache import canonical, digest, FinalStore
from converter_api.final_inputs import load_inputs, approved_default
from converter_api.final_prewarm import DefaultFbxPrewarmer, InputsChanged
from converter_api.work_queue import WorkStopped, current_work
from tests.converter.test_final_cache import Runner, FBX_MAGIC


@pytest.fixture
def setup(tmp_path, monkeypatch):
    library=tmp_path/'library';(library/'bvh').mkdir(parents=True)
    (library/'bvh/pose.bvh').write_bytes(b'registered source')
    with sqlite3.connect(library/'poses.db') as db:
        db.execute('create table poses(pose_id text primary key, bvh_path text)')
        db.execute('insert into poses values (?,?)',('pose','data/bvh/pose.bvh'))
    assets=[];characters={}
    for cid,gender,role in [('male-base','male','default'),('female-base','female','default'),('male-next','male','preset')]:
        path=tmp_path/(cid+'.fbx');path.write_bytes(FBX_MAGIC+cid.encode())
        sha=digest(path.read_bytes())
        assets.append(dict(characterId=cid,assetSha256=sha,gender=gender,role=role,
            shape=dict(controls=[dict(key='height',default=168.684899)]),shapeSource={}))
        characters[cid]=dict(display_name=cid,artifact_uri_env=cid.upper().replace('-','_')+'_URI',
                             sha256=sha,rig_profile='mixamo',revision='v1')
        monkeypatch.setenv(characters[cid]['artifact_uri_env'],path.as_uri())
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps(dict(defaultCharacterId='male-base',assets=assets)))
    registry=tmp_path/'characters.json';registry.write_text(json.dumps(dict(schema_version=1,characters=characters)))
    shapes=tmp_path/'shapes.json';shapes.write_text('{}')
    root=tmp_path/'cache'
    for key,value in dict(CONVERTER_FINAL_CACHE_ROOT=root,CONVERTER_FINAL_LIBRARY_ROOT=library,
        CONVERTER_FINAL_CATALOG_PATH=catalog,CONVERTER_CHARACTER_REGISTRY=registry,
        CONVERTER_DEFAULT_CHARACTER_ID='male-base',CONVERTER_SHAPE_REGISTRY=shapes,
        CONVERTER_FINAL_PREWARM_INTERVAL_SECONDS='.02',CONVERTER_FINAL_PREWARM_RETRY_SECONDS='.02').items():
        monkeypatch.setenv(key,str(value))
    blend=tmp_path/'source.blend';blend.write_bytes(b'approved blend')
    controller=tmp_path/'controller.py';controller.write_bytes(b'approved controller')
    def resolve(body, reg):
        character=reg.resolve(body['characterId'])
        recipe=dict(rules='test-approved-v1',values={'height_cm':body['shape']['controls'][0]['default']},
                    defaults={'height_cm':body['shape']['controls'][0]['default']},shape_sha256=None,
                    blend=str(blend),blend_sha256=digest(blend.read_bytes()),
                    controller=str(controller),controller_sha256=digest(controller.read_bytes()))
        return character,recipe
    monkeypatch.setattr('converter_api.final_prewarm.approved_default',resolve)
    Runner.calls=0
    runner=Runner()
    worker=DefaultFbxPrewarmer.from_env(runner)
    def change(cid='male-next'):
        raw=json.loads(catalog.read_text())
        for a in raw['assets']:
            if a['gender']=='male':a['role']='default' if a['characterId']==cid else 'preset'
        raw['defaultCharacterId']=cid;catalog.write_text(json.dumps(raw))
    return SimpleNamespace(worker=worker,runner=runner,catalog=catalog,library=library,root=root,change=change)


def test_changed_default_only_bakes_new_model_and_reverting_reuses_original(setup):
    s=setup;s.worker.run_once();assert Runner.calls==4
    assert s.worker.snapshot()['status']=='ready'
    s.change();s.worker.run_once()
    assert Runner.calls==6  # Both camera variants of the new male; female is reused.
    assert s.worker.snapshot()['defaults']==['female-base','male-next']
    assert s.worker.snapshot()['ready']==4
    s.change('male-base');s.worker.run_once();assert Runner.calls==6


def test_restart_resumes_verified_artifacts_without_conversion(setup):
    s=setup;s.worker.run_once()
    resumed=DefaultFbxPrewarmer.from_env(s.runner);resumed.run_once()
    assert Runner.calls==4 and resumed.snapshot()['ready']==4
    assert all(r['origin']=='hit' for r in resumed.snapshot()['results'])


def test_numeric_default_change_is_a_new_exact_combination(setup):
    s=setup;s.worker.run_once()
    raw=json.loads(s.catalog.read_text());raw['assets'][0]['shape']['controls'][0]['default']+=1
    s.catalog.write_text(json.dumps(raw));s.worker.run_once()
    assert Runner.calls==6


def test_changed_policy_reaudits_even_when_catalog_is_unchanged(setup, monkeypatch):
    s=setup;s.worker.run_once()
    monkeypatch.setattr('converter_api.final_prewarm.policy_sha256',lambda:'f'*64)
    monkeypatch.setattr('converter_api.final_cache.policy_sha256',lambda:'f'*64)
    s.worker.run_once();assert Runner.calls==8


def test_late_old_generation_cannot_mark_new_defaults_ready(setup, monkeypatch):
    s=setup;original=s.runner.convert;changed=False
    def late(**options):
        nonlocal changed
        result=original(**options)
        if not changed:s.change();changed=True
        return result
    monkeypatch.setattr(s.runner,'convert',late)
    with pytest.raises(InputsChanged):s.worker.run_once()
    assert s.worker.snapshot()['status']!='ready'
    s.worker.run_once();assert s.worker.snapshot()['status']=='ready'
    assert 'male-next' in s.worker.snapshot()['defaults']


def test_failed_asset_is_retried_and_successful_assets_are_reused(setup, monkeypatch):
    s=setup;original=s.runner.convert;failed=False
    def flaky(**options):
        nonlocal failed
        if not failed:failed=True;raise ValueError('failed final reimport')
        return original(**options)
    monkeypatch.setattr(s.runner,'convert',flaky)
    s.worker.run_once();assert s.worker.snapshot()['status']=='retry_pending'
    assert s.worker.snapshot()['failed']==1 and Runner.calls==3
    s.worker.run_once();assert s.worker.snapshot()['status']=='ready' and Runner.calls==4


def test_foreground_preemption_leaves_pending_work_for_retry(setup, monkeypatch):
    s=setup;original=s.runner.convert
    def preempted(**options):
        assert current_work.get().kind=='preview'
        raise WorkStopped('PREVIEW_PREEMPTED',503,'execution')
    monkeypatch.setattr(s.runner,'convert',preempted)
    s.worker.run_once();assert s.worker.snapshot()['status']=='paused_for_foreground'
    assert s.worker.snapshot()['pending']==4
    monkeypatch.setattr(s.runner,'convert',original);s.worker.run_once()
    assert s.worker.snapshot()['status']=='ready'


def test_completed_generation_audit_repairs_corrupt_final_bytes(setup, monkeypatch):
    import converter_api.final_prewarm as prewarm
    # A fresh Linux VM can have monotonic() < audit interval; zero is not
    # necessarily in the past. Exercise that epoch without changing other clocks.
    monkeypatch.setattr(prewarm, 'time', SimpleNamespace(monotonic=lambda: 5.0))
    s=setup;s.worker.run_once();next(s.root.glob('*.fbx')).write_bytes(b'corrupt')
    s.worker._last_audit=prewarm.time.monotonic()-s.worker.audit-1;s.worker.run_once()
    assert s.worker.snapshot()['status']=='ready' and Runner.calls==5


def test_api_lifecycle_starts_watching_without_a_trigger_endpoint(setup):
    from converter_api.app import create_app
    s=setup
    with TestClient(create_app(runner=s.runner)) as client:
        def ready(cid):
            until=time.monotonic()+5
            while time.monotonic()<until:
                value=client.get('/final-cache/prewarm-status').json()
                if value['status']=='ready' and cid in value['defaults']:return value
                time.sleep(.02)
            raise AssertionError(value)
        assert ready('male-base')['ready']==4
        s.change();assert ready('male-next')['ready']==4
        assert Runner.calls==6
    assert not client.app.state.final_prewarm.thread.is_alive()


def test_registry_versions_remain_immutable_during_catalog_change(setup, monkeypatch):
    s=setup;s.worker.run_once()
    import os
    old=os.environ['CONVERTER_FINAL_LIBRARY_SHA256'];before=(s.root/('library-'+old+'.json')).read_bytes()
    s.change();s.worker.run_once()
    new=os.environ['CONVERTER_FINAL_LIBRARY_SHA256']
    assert old!=new and (s.root/('library-'+old+'.json')).read_bytes()==before
    assert (s.root/('library-'+new+'.json')).is_file()


def test_manual_and_automatic_sources_use_identical_trusted_input_manifest(setup):
    defaults,sources,registered=load_inputs(setup.library,setup.catalog)
    assert len(defaults)==2 and sources[0]['source_bvh_sha256']==digest(b'registered source')
    assert json.loads(registered)['characters']['male-next']


def test_catalog_defaults_must_match_approved_authoring_recipe(setup, monkeypatch):
    from converter_api.registry import CharacterRegistry
    body=json.loads(setup.catalog.read_text())['assets'][0]
    monkeypatch.setattr('converter.shape_spec.resolve_shape',lambda *args:dict(values={'height_cm':0}))
    with pytest.raises(ValueError,match='default controls mismatch'):
        approved_default(body,CharacterRegistry.from_env())


def test_full_generation_plan_is_written_before_first_conversion(setup, monkeypatch):
    s=setup;original=s.runner.convert
    def planned(**options):
        plan=json.loads(next(s.root.glob('prewarm-plan-*.json')).read_text())
        assert plan['total']==4 and len(plan['assets'])==4
        assert plan['estimate']['upper_seconds_at_61_9']==4*61.9
        assert all(row['status']=='not_checked' for row in plan['assets'])
        return original(**options)
    monkeypatch.setattr(s.runner,'convert',planned);s.worker.run_once()
    assert s.worker.snapshot()['ready']==4


def test_optional_prewarm_storage_failure_keeps_api_available(setup, monkeypatch, tmp_path):
    from converter_api.app import create_app
    path=tmp_path/'not-a-directory';path.write_text('occupied')
    monkeypatch.setenv('CONVERTER_FINAL_CACHE_ROOT',str(path))
    with TestClient(create_app(runner=setup.runner)) as client:
        until=time.monotonic()+2
        while time.monotonic()<until:
            status=client.get('/final-cache/prewarm-status').json()
            if status['status']=='retry_pending':break
            time.sleep(.01)
        assert status['status']=='retry_pending'
        assert Runner.calls==0


def test_prewarm_keeps_the_export_hand_screen_gate(setup, monkeypatch):
    monkeypatch.setattr('converter.hand_contact.selected_hand_screen_enabled',lambda scope:scope=='full')
    captured=[]
    def screen_unavailable(**options):
        captured.append(options['selected_hand_screen'])
        raise ValueError('selected hand screen unavailable')
    monkeypatch.setattr(setup.runner,'convert',screen_unavailable)
    setup.worker.run_once()
    assert captured==[True]*4
    assert setup.worker.snapshot()['ready']==0
    assert setup.worker.snapshot()['status']=='retry_pending'
