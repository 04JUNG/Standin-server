import json,hashlib
import pytest
from converter.shape_spec import resolve_shape,digest_shape,ShapeError,PINS

def setup_registry(tmp_path,monkeypatch):
 r={'rules':'standin-body-0.6.7','blend':str(tmp_path/'source.blend'),'controller':str(tmp_path/'source.py'),'blend_sha256':PINS['standin-body-0.6.7'][0],'controller_sha256':PINS['standin-body-0.6.7'][1],'defaults':{'muscle':0},'controls':[{'key':'muscle','min':0,'max':2,'step':.01,'default':0}],'asset_sha256':'a'*64}
 p=tmp_path/'registry.json';p.write_text(json.dumps({'schema':'shape-bake.v1','characters':{'body':r}}));monkeypatch.setenv('CONVERTER_SHAPE_REGISTRY',str(p))
 # Source file trust is covered separately; boundary tests don't need binary models.
 import converter.shape_spec as spec
 original=spec.validate_recipe;monkeypatch.setattr(spec,'validate_recipe',lambda r:original(r,False))
 return r

def envelope(c):return {'schemaVersion':'body-shape.v1','controls':c,'shapeSha256':digest_shape('body','a'*64,c)}

def test_digest_matches_javascript():
 assert digest_shape('body','a'*64,{'muscle':1.0})==hashlib.sha256(('['+'"body-shape.v1","body","'+'a'*64+'",[["muscle",1]]]').encode()).hexdigest()

def test_valid_shape_and_default(tmp_path,monkeypatch):
 setup_registry(tmp_path,monkeypatch)
 assert resolve_shape('body','a'*64)['values']=={'muscle':0}
 assert resolve_shape('body','a'*64,envelope({'muscle':1.5}))['values']=={'muscle':1.5}

@pytest.mark.parametrize('value',[True,'1',float('nan'),float('inf'),-.1,2.1,.005])
def test_invalid_values_fail_closed(tmp_path,monkeypatch,value):
 setup_registry(tmp_path,monkeypatch)
 with pytest.raises(ShapeError):resolve_shape('body','a'*64,{'schemaVersion':'body-shape.v1','controls':{'muscle':value},'shapeSha256':'b'*64})

@pytest.mark.parametrize('change',[{'controls':{'unknown':0}},{'shapeSha256':'b'*64},{'schemaVersion':'body-shape.v0'}])
def test_stale_or_unknown_shape_rejected(tmp_path,monkeypatch,change):
 setup_registry(tmp_path,monkeypatch)
 with pytest.raises(ShapeError):resolve_shape('body','a'*64,{**envelope({'muscle':1}),**change})

def test_asset_or_capability_mismatch_rejected(tmp_path,monkeypatch):
 setup_registry(tmp_path,monkeypatch)
 with pytest.raises(ShapeError):resolve_shape('body','b'*64)
 with pytest.raises(ShapeError):resolve_shape('unregistered','a'*64,envelope({'muscle':1}))
 assert resolve_shape('unregistered','a'*64) is None


def test_missing_shape_source_closes_health(tmp_path,monkeypatch):
 from tests.converter.test_api_contract import _client
 monkeypatch.setenv('CONVERTER_SHAPE_REGISTRY',str(tmp_path/'missing.json'))
 response=_client(tmp_path).get('/healthz')
 assert response.status_code==503
 assert response.json()['checks']['shape_sources']['ok'] is False

def test_shape_does_not_claim_exact_parent_mode():
 from dataclasses import replace
 from converter_api.runner import BlenderRunner,RunnerSettings,ConversionRejectedError
 runner=BlenderRunner(replace(RunnerSettings.from_env(),force_exact_v324=True))
 with pytest.raises(ConversionRejectedError,match='exact V3.2.4'):
  runner.convert(bvh_bytes=b'x',character_path=__import__('pathlib').Path('/unused'),character_id='x',character_sha256='a'*64,conversion_id='test',shape_recipe={})
