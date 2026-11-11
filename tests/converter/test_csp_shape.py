import hashlib,json
from pathlib import Path
import pytest
from converter_api.registry import CharacterRegistry
from converter_api.csp_shape import validate_request
from converter.shape_spec import ShapeError,digest_shape

@pytest.fixture
def boundary(tmp_path,monkeypatch):
    blend=tmp_path/'body.blend';blend.write_bytes(b'approved-test-source')
    controller=tmp_path/'body.py';controller.write_text("VERSION='standin-body-test'\n")
    source={'blend':str(blend),'controller':str(controller),'blend_sha256':hashlib.sha256(blend.read_bytes()).hexdigest(),'controller_sha256':hashlib.sha256(controller.read_bytes()).hexdigest()}
    cid='male-test';asset_sha='a'*64
    shape={'schemaVersion':'body-shape.v1','controls':{'height':180},'shapeSha256':digest_shape(cid,asset_sha,{'height':180})}
    data={'schema':'shape-bake.v2','sources':{'standin-body-test':source},'characters':{cid:{'rules':'standin-body-test','preset':'BASE','asset':'test','asset_sha256':asset_sha,'defaults':{'height_cm':178},'controls':[{'key':'height','min':150,'max':200,'step':1,'default':178}]}}}
    path=tmp_path/'shape.json';path.write_text(json.dumps(data));monkeypatch.setenv('CONVERTER_SHAPE_REGISTRY',str(path))
    registry_file=tmp_path/'characters.json';registry_file.write_text(json.dumps({'schema_version':1,'characters':{cid:{'display_name':'test','artifact_uri_env':'TEST_URI','sha256':asset_sha,'rig_profile':'mixamo','revision':'test'}}}))
    registry=CharacterRegistry(registry_file,default_character_id=cid)
    payload={'character_id':cid,'character_sha256':asset_sha,'shape':shape,'expected_shape_source':{'rules':'standin-body-test','blendSha256':source['blend_sha256'],'controllerSha256':source['controller_sha256']}}
    return payload,registry,blend

def test_shape_values_are_resolved_from_pinned_registry(boundary):
    p,r,_=boundary;assert validate_request(p,r)['values']['height_cm']==180

def test_wrong_source_and_character_sha_rejected(boundary):
    p,r,_=boundary
    with pytest.raises(ShapeError):validate_request({**p,'character_sha256':'0'*64},r)
    with pytest.raises(ShapeError):validate_request({**p,'expected_shape_source':{**p['expected_shape_source'],'blendSha256':'0'*64}},r)

def test_missing_or_dropped_control_digest_rejected(boundary):
    p,r,_=boundary
    with pytest.raises(ShapeError):validate_request({**p,'shape':{**p['shape'],'controls':{}}},r)
    with pytest.raises(ShapeError):validate_request({**p,'expected_shape_source':None},r)

def test_modified_source_file_is_rejected_before_blender(boundary):
    p,r,blend=boundary;blend.write_bytes(b'changed')
    with pytest.raises(ShapeError):validate_request(p,r)
