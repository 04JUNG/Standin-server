"""Failure boundaries and zero compatibility; no mock implies CSP acceptance."""
import hashlib
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from converter.outline_spec import (normalize, appearance_hash, thickness_m,
    CHARACTERS, CSP_CERTIFIED, VERSION, cache_key, expression_for, NATIVE_SURFACE,
    NATIVE_EXPRESSION, NATIVE_MAPPING, certified)
from converter.outline_spec import physical_thickness, LEGACY_SURFACE, supported_targets
from converter_api.outline import register_outline


@pytest.mark.parametrize('value',[True,0,'1',[],{}, {'version':VERSION,'width':True},
    {'version':VERSION,'width':float('nan')},{'version':VERSION,'width':float('inf')},
    {'version':VERSION,'width':-1},{'version':VERSION,'width':2.01},
    {'version':VERSION,'width':1,'color':'black'},{'version':'future','width':1}])
def test_invalid_appearance_is_never_silently_zero(value):
    with pytest.raises(ValueError): normalize(value)


def test_zero_and_numeric_encoding_have_stable_identity():
    assert normalize()==normalize({'version':VERSION,'width':-0.0})
    assert appearance_hash({'version':VERSION,'width':2.0})=='4396270491be9d88f58ed0531d9a72fcfd97b065f5be97e987e5a32076cf15ab'
    assert appearance_hash({'version':VERSION,'width':1})!=appearance_hash({'version':VERSION,'width':2})


def test_native_transparent_acceptance_excludes_old_models_and_rejected_surface():
    assert expression_for(NATIVE_SURFACE)==(NATIVE_EXPRESSION,NATIVE_MAPPING)
    assert expression_for(NATIVE_SURFACE)!=expression_for('inverted-opaque')
    assert not certified(next(iter(CHARACTERS.values())),'full',2,surface=NATIVE_SURFACE)
    for cid in ('male-default83-v0-7-6-p1','female-default80-v0-3-9-p1'):
        assert certified(CHARACTERS[cid],'full',.65)
        assert certified(CHARACTERS[cid],'head',2)
        assert not certified(CHARACTERS[cid],'head',1)
        assert not certified(CHARACTERS[cid],'full',0)
        assert not certified(CHARACTERS[cid],'full',2,surface=LEGACY_SURFACE)
    with pytest.raises(ValueError): expression_for('unreviewed')


def test_capability_intervals_match_actual_acceptance_without_cartesian_widening():
    for row in supported_targets():
        bounds=row['width']
        for width in (0,.25,.5,.65,1,1.25,1.75,2):
            lower=width>bounds['min'] if bounds['min_exclusive'] else width>=bounds['min']
            advertised=lower and width<=bounds['max']
            assert certified(row['character_sha256'],row['output_scope'],width)==advertised


def test_physical_width_is_stature_relative_and_scope_independent():
    assert thickness_m({'version':VERSION,'width':2},1.75)==.002625
    assert thickness_m({'version':VERSION,'width':.5},1.75)==.00065625
    assert thickness_m({'version':VERSION,'width':1},2)==.0015
    for height in (False,0,2.51,float('nan')):
        with pytest.raises(ValueError): thickness_m(None,height)


def test_native_scope_mapping_keeps_full_stature_but_reduces_close_crop_width():
    appearance={'version':VERSION,'width':2}
    assert physical_thickness(appearance,1.75,'full')==.002625
    assert physical_thickness(appearance,1.75,'head')==.002625*.15
    assert physical_thickness(appearance,1.75,'half')==.002625*.60
    assert physical_thickness(appearance,1.75,'head',surface=LEGACY_SURFACE)==.002625
    for scope in ('full','half','bust','shoulder_head','head'):
        assert physical_thickness(None,1.75,scope)==0


def test_cache_identity_covers_appearance_stature_scope_source_and_runtime(monkeypatch):
    import converter.outline_spec as spec
    args=['a'*64,{'version':VERSION,'width':1},1.75,'full','approved','b'*64]
    key=cache_key(*args)
    for index,value in [(0,'c'*64),(1,{'version':VERSION,'width':2}),
            (2,1.8),(3,'head'),(4,'other'),(5,'d'*64)]:
        changed=args.copy();changed[index]=value
        assert cache_key(*changed)!=key
    monkeypatch.setattr(spec,'runtime_revision',lambda:'e'*64)
    assert cache_key(*args)!=key


@pytest.fixture
def http(monkeypatch):
    from converter_api.app import ApiProblem, _error_response
    monkeypatch.setenv('CONVERTER_INTERACTIVE_TOKEN','t'*32)
    app=FastAPI()
    @app.exception_handler(ApiProblem)
    async def handle(request,error): return _error_response(error)
    cid=next(iter(CHARACTERS))
    registry=SimpleNamespace(metadata=lambda value:SimpleNamespace(sha256=CHARACTERS[value]))
    class NeverRunner:
        def inspect_blender(self): raise AssertionError('uncertified request started Blender')
    register_outline(app,registry,NeverRunner())
    return TestClient(app)


def post(http,*,token='t'*32,**changes):
    cid=next(iter(CHARACTERS))
    fields=dict(source_artifact_sha256='a'*64,appearance=json.dumps({'version':VERSION,'width':1}),
        output_scope='full',character_id=cid,character_sha256=CHARACTERS[cid],
        binding_sha256='b'*64,stature_m='1.75')
    fields.update(changes)
    return http.post('/outline-fbx',headers={'Authorization':'Bearer '+token},
        files={'fbx':('source.fbx',b'never-consumed')},data=fields)


def test_capability_never_advertises_rejected_expression_even_if_env_enabled(http,monkeypatch):
    monkeypatch.setenv('CONVERTER_OUTLINE_ENABLED','true')
    capability=http.get('/outline-contract')
    assert capability.headers['cache-control']=='no-store'
    data=capability.json()
    assert data['enabled'] is True and data['csp_certified'] is True
    assert data['certified_expressions']==[NATIVE_EXPRESSION]
    assert data['acceptance_scope']=='posed-fbx-import'
    assert len(data['supported_targets'])==10
    for row in data['supported_targets']:
        assert row['character_id'] in data['supported_characters']
        assert row['character_sha256']==CHARACTERS[row['character_id']]
        bounds=(dict(min=0,max=2,min_exclusive=True) if row['output_scope']=='full'
            else dict(min=2,max=2,min_exclusive=False))
        assert row['width']==bounds
    assert CSP_CERTIFIED
    assert post(http).status_code==503


@pytest.mark.parametrize('width',[.5,.65,1,1.5])
@pytest.mark.parametrize('scope',['half','bust','shoulder_head','head'])
def test_uncertified_crop_width_never_starts_worker(http,monkeypatch,width,scope):
    monkeypatch.setenv('CONVERTER_OUTLINE_ENABLED','true')
    cid='male-default83-v0-7-6-p1'
    r=post(http,character_id=cid,character_sha256=CHARACTERS[cid],output_scope=scope,
        appearance=json.dumps(dict(version=VERSION,width=width)))
    assert r.status_code==503 and r.json()['error']['code']=='OUTLINE_NOT_CSP_CERTIFIED'


def test_accepted_full_midpoint_still_checks_actual_source_hash(http,monkeypatch):
    from converter.scene_contract import FBX_MAGIC
    monkeypatch.setenv('CONVERTER_OUTLINE_ENABLED','true')
    cid='male-default83-v0-7-6-p1'
    r=http.post('/outline-fbx',headers={'Authorization':'Bearer '+'t'*32},
        files={'fbx':('source.fbx',FBX_MAGIC+b'not-the-bound-source')},
        data=dict(source_artifact_sha256='a'*64,appearance=json.dumps(dict(version=VERSION,width=.65)),
            output_scope='full',character_id=cid,character_sha256=CHARACTERS[cid],
            binding_sha256='b'*64,stature_m='1.78'))
    assert r.status_code==409 and r.json()['error']['code']=='SOURCE_FBX_SHA256_MISMATCH'


def test_default_off_and_zero_keeps_legacy_path_without_worker(http,monkeypatch):
    monkeypatch.delenv('CONVERTER_OUTLINE_ENABLED',raising=False)
    assert http.get('/outline-contract').json()['enabled'] is False
    assert post(http).json()['error']['code']=='OUTLINE_DISABLED'
    r=post(http,appearance=json.dumps({'version':VERSION,'width':0}))
    assert r.status_code==400 and r.json()['error']['code']=='OUTLINE_ZERO_USE_LEGACY'


def test_authentication_and_unsupported_bindings_fail_without_worker(http,monkeypatch):
    assert http.post('/outline-fbx',files={'fbx':('x',b'x')},data={}).status_code==422
    monkeypatch.setenv('CONVERTER_OUTLINE_ENABLED','true')
    assert post(http,token='x'*32).status_code==401
    assert post(http,character_id='personal').json()['error']['code']=='OUTLINE_UNSUPPORTED'
    assert post(http,binding_sha256='not-sha').json()['error']['code']=='INVALID_OUTLINE'
    assert post(http,appearance='{"version":"fbx-outline-v1","width":true}').status_code==400
    assert post(http,output_scope='scene').status_code==400
