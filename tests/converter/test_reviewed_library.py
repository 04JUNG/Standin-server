import hashlib,json
import pytest
from converter.reviewed_library import restore,_catalog,VERSION

def setup(tmp_path,monkeypatch):
    source=tmp_path/'pose.bvh'; source.write_text('derived reviewed skeleton')
    character=tmp_path/'character.fbx';character.write_bytes(b'character')
    h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    digest=h(source);asset=tmp_path/(digest+'.fbx');asset.write_bytes(b'reviewed final fbx')
    row=dict(pose_id='approved',path=asset.name,fbx_sha256=h(asset),bones=52)
    data=dict(version=VERSION,character_sha256=h(character),poses={digest:row})
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(data))
    monkeypatch.setenv('REVIEWED_LIBRARY_ROOT',str(tmp_path));monkeypatch.setenv('REVIEWED_LIBRARY_SHA256',h(manifest));_catalog.cache_clear()
    job=dict(bvh_path=str(source),character_fbx=str(character),character_sha256=h(character),output_path=str(tmp_path/'out.fbx'),mirror=False,force_exact_v324=False)
    return job,asset,manifest

def test_exact_reviewed_bytes(tmp_path,monkeypatch):
    job,asset,_=setup(tmp_path,monkeypatch);r=restore(job)
    assert r['ok'] and r['reviewed_library']['reviewed_fbx_sha256']==hashlib.sha256(asset.read_bytes()).hexdigest()
    assert (tmp_path/'out.fbx').read_bytes()==asset.read_bytes()

@pytest.mark.parametrize('change',[{'mirror':True},{'force_exact_v324':True},{'shape_recipe':{'shape':'other'}},{'character_sha256':'f'*64}])
def test_other_variants_do_not_reuse_default(tmp_path,monkeypatch,change):
    job,_,_=setup(tmp_path,monkeypatch);job.update(change);assert restore(job) is None
    assert not (tmp_path/'out.fbx').exists()

def test_changed_bvh_uses_normal_path(tmp_path,monkeypatch):
    job,_,_=setup(tmp_path,monkeypatch);(tmp_path/'pose.bvh').write_text('refined pose')
    assert restore(job) is None

def test_tampered_fbx_rejected(tmp_path,monkeypatch):
    job,asset,_=setup(tmp_path,monkeypatch);asset.write_bytes(b'changed')
    with pytest.raises(ValueError,match='integrity'):restore(job)

def test_tampered_catalog_rejected(tmp_path,monkeypatch):
    job,_,manifest=setup(tmp_path,monkeypatch);manifest.write_text('{}')
    with pytest.raises(ValueError,match='manifest changed'):restore(job)
