"""Strict separate skin choice, legacy no-op, exact asset/target boundaries."""
import json
from types import SimpleNamespace
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from converter.skin_spec import VERSION,normalize,skin_hash,manifest,manifest_hash,EXPRESSION
from converter_api.skin import register_skin


@pytest.mark.parametrize('value',[True,1,'flesh',[],{},dict(version=VERSION,id=True),dict(version=VERSION,id='other'),dict(version='future',id='flesh'),dict(version=VERSION,id='flesh',url='https://example.com/x')])
def test_invalid_skin_is_not_silently_default(value):
    with pytest.raises(ValueError):normalize(value)


def test_default_choice_and_asset_pins_are_independent_of_outline_json():
    assert normalize()==normalize(dict(version=VERSION,id='default'))
    assert skin_hash()!=skin_hash(dict(version=VERSION,id='flesh'))
    from converter.outline_spec import normalize as outline
    with pytest.raises(ValueError):outline(dict(version='fbx-outline-v1',width=1,skin='flesh'))
    assert len(manifest_hash())==64 and len(manifest()['models'])==22
    for row in manifest()['models'].values():
        assert row['bones'] in (52,54) and 'flesh' in row['textures']
        assert set(row['textures']) == {'gray-guide','flesh','flesh-guide','muscle'}
        assert len(row['uv']['sha256'])==64


@pytest.fixture
def http(monkeypatch):
    from converter_api.app import ApiProblem,_error_response
    monkeypatch.setenv('CONVERTER_INTERACTIVE_TOKEN','t'*32)
    monkeypatch.delenv('CONVERTER_SKIN_ENABLED',raising=False)
    monkeypatch.delenv('CONVERTER_SKIN_ASSET_ROOT',raising=False)
    app=FastAPI()
    @app.exception_handler(ApiProblem)
    async def error(request,problem):return _error_response(problem)
    class NeverRunner:
        def inspect_blender(self):raise AssertionError('invalid skin request started Blender')
    registry=SimpleNamespace(metadata=lambda cid:SimpleNamespace(sha256=manifest()['models'][cid]['character_sha256']))
    register_skin(app,registry,NeverRunner());return TestClient(app)


def post(http,**changes):
    cid=next(iter(manifest()['models']));fields=dict(source_artifact_sha256='a'*64,skin=json.dumps(dict(version=VERSION,id='flesh')),binding_sha256='b'*64,character_id=cid,character_sha256=manifest()['models'][cid]['character_sha256'],output_scope='full');fields.update(changes)
    return http.post('/skin-fbx',headers={'Authorization':'Bearer '+'t'*32},files={'fbx':('source.fbx',b'never-consumed')},data=fields)


def test_default_off_and_default_skin_use_legacy_without_blender(http):
    capability=http.get('/skin-contract').json()
    assert not capability['enabled'] and not capability['assets_available']
    assert capability['expression']==EXPRESSION and capability['order']==['skin','outline']
    assert post(http).json()['error']['code']=='SKIN_DISABLED'
    assert post(http,skin=json.dumps(dict(version=VERSION,id='default'))).json()['error']['code']=='SKIN_DEFAULT_USE_LEGACY'


@pytest.mark.parametrize('scope',['half','bust','shoulder_head','head'])
def test_unproved_crop_never_accepts_a_uv_guess(http,monkeypatch,scope):
    monkeypatch.setenv('CONVERTER_SKIN_ENABLED','true')
    response=post(http,output_scope=scope)
    assert response.status_code==422 and response.json()['error']['code']=='SKIN_UNSUPPORTED'


def test_arbitrary_skin_url_model_or_asset_is_refused(http,monkeypatch):
    monkeypatch.setenv('CONVERTER_SKIN_ENABLED','true')
    assert post(http,character_id='personal').json()['error']['code']=='SKIN_UNSUPPORTED'
    assert post(http,skin=json.dumps(dict(version=VERSION,id='flesh',url='/tmp/image.png'))).status_code==400
    assert post(http,binding_sha256='not-a-digest').status_code==400
    assert http.get('/skin-assets/'+'a'*64).status_code==401
    assert http.get('/skin-assets/'+'a'*64,headers={'Authorization':'Bearer '+'t'*32}).status_code==404


def test_capability_exposes_only_relative_approved_asset_ids(http):
    capability=http.get('/skin-contract').json()
    for asset in capability['assets']:
        assert asset['url']=='/skin-assets/'+asset['sha256']
        assert asset['kind'] in ('uv','texture','preview-texture') and len(asset['sha256'])==64
    for target in capability['supported_targets']:
        assert target['output_scope']=='full'
        assert target['character_sha256']==manifest()['models'][target['character_id']]['character_sha256']


def test_replaced_asset_manifest_cannot_inherit_csp_acceptance(monkeypatch):
    from converter import skin_spec as spec
    cid=next(iter(spec.manifest()['models']));sha=spec.manifest()['models'][cid]['character_sha256'];skin=dict(version=VERSION,id='flesh')
    assert spec.certified(sha,'full',skin)
    monkeypatch.setattr(spec,'manifest_hash',lambda:'f'*64)
    assert not spec.certified(sha,'full',skin)


def test_changed_material_writer_cannot_inherit_csp_acceptance(monkeypatch):
    from converter import skin_spec as spec
    sha=next(iter(spec.manifest()['models'].values()))['character_sha256']
    skin=dict(version=VERSION,id='flesh')
    assert spec.certified(sha,'full',skin)
    monkeypatch.setattr(spec,'material_revision',lambda:'f'*64)
    assert not spec.certified(sha,'full',skin)
