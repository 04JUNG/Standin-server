"""Poster lineage, anatomical reference fallback, and packaging regressions."""
import io
import json
from pathlib import Path
import sqlite3
import tarfile

import numpy as np
from PIL import Image
import pytest

from src import pose_posters as posters
from scripts.deploy_pose_library import _check_posters, make_archive, DeployError


def image_bytes(color='gray'):
    b = io.BytesIO()
    Image.new('RGB', (256, 256), color).save(b, format='PNG')
    return b.getvalue()


def write_poster(root, source, scope='full'):
    data = image_bytes()
    meta = {**posters.identity(source, scope), 'image_sha256': posters.digest(data), 'bytes': len(data)}
    (root / (source+'__'+scope+'.png')).write_bytes(data)
    (root / (source+'__'+scope+'.json')).write_text(json.dumps(meta))
    return meta


def test_poster_requires_exact_source_model_scope_renderer_and_png_hash(tmp_path):
    source = 'a'*64
    meta = write_poster(tmp_path, source)
    assert posters.read_poster(tmp_path, source)[1]['kind'] == 'default-body'
    path = tmp_path/(source+'__full.json')
    for key in ['version','source_bvh_sha256','character_id','character_sha256','renderer_revision','scope','view','projection','image_sha256']:
        path.write_text(json.dumps({**meta,key:'changed'}))
        with pytest.raises(ValueError): posters.read_poster(tmp_path, source)
    path.write_text(json.dumps(meta))
    (tmp_path/(source+'__full.png')).write_bytes(image_bytes('red'))
    with pytest.raises(ValueError): posters.read_poster(tmp_path, source)
    with pytest.raises((OSError,ValueError)): posters.read_poster(tmp_path,source,'half')
    with pytest.raises(ValueError): posters.read_poster(tmp_path,'../'+source)


def test_pose_only_uses_anatomical_scope_not_raised_leg_or_screen_height(tmp_path,monkeypatch):
    import src.bvh
    points=np.zeros((17,3));points[0]=[0,3,0]
    points[5]=[-.5,2,0];points[6]=[.5,2,0]
    points[11]=[-.35,1,0];points[12]=[.35,1,0]
    points[7]=[-.8,1.5,0];points[8]=[.8,1.5,0]
    points[9]=[-1.1,.9,0];points[10]=[1.1,.9,0]
    points[13]=[-.35,.5,0];points[14]=[.35,.5,0]
    points[15]=[-.35,0,0];points[16]=[.35,0,0]
    def source(path,frame):
        p=points.copy();kind=Path(path).read_bytes()
        if kind==b'raised':p[13:17,1]+=10
        if kind==b'lowered-hands':p[9:11,1]-=3
        return p,np.ones(17)
    monkeypatch.setattr(src.bvh,'load_coco17',source)
    posters.pose_only.cache_clear()
    results={s:posters.poster(tmp_path,b'base',s) for s in posters.OUTPUT_SCOPES}
    assert len({posters.digest(data) for data,_ in results.values()})==5
    assert all(meta['kind']=='pose-only' and meta['character_id']=='none' for _,meta in results.values())
    for scope in ('half','bust','shoulder_head','head'):
        assert posters.pose_only(b'base',scope)==posters.pose_only(b'raised',scope)
    assert posters.pose_only(b'base','half')!=posters.pose_only(b'lowered-hands','half')
    for scope in ('bust','shoulder_head','head'):
        assert posters.pose_only(b'base',scope)==posters.pose_only(b'lowered-hands',scope)
    posters.pose_only.cache_clear()


def test_scope_poster_bundle_is_complete_and_packaged(tmp_path):
    directory=tmp_path/'pose-posters';directory.mkdir()
    bvh=tmp_path/'bvh';bvh.mkdir();(bvh/'pose.bvh').write_bytes(b'library source')
    with sqlite3.connect(tmp_path/'poses.db') as db:
        db.execute('CREATE TABLE poses(pose_id TEXT,bvh_path TEXT)')
        db.execute("INSERT INTO poses VALUES('pose','data/bvh/pose.bvh')")
    source=posters.digest(b'library source')
    for scope in posters.OUTPUT_SCOPES:write_poster(directory,source,scope)
    (directory/'library.json').write_text(json.dumps({'complete':True,'failures':0,'expected':5}))
    assert _check_posters(tmp_path)=={'status':'verified','images':5}
    (directory/'private-rough.png').write_bytes(b'should never be packaged')
    (directory/'progress.jsonl').write_text('offline diagnostics')
    archive=tmp_path/'bundle.tgz';make_archive(tmp_path,archive)
    with tarfile.open(archive) as tar:
        names=tar.getnames()
    assert all('pose-posters/'+source+'__'+scope+'.png' in names for scope in posters.OUTPUT_SCOPES)
    assert not any('private-rough' in name or 'progress.jsonl' in name for name in names)
    (directory/(source+'__bust.png')).unlink()
    with pytest.raises(DeployError):_check_posters(tmp_path)


def test_old_bundle_is_explicit_pose_only_not_fake_complete(tmp_path):
    assert _check_posters(tmp_path)=={'status':'absent','fallback':'pose-only'}

def test_api_preserves_quarantine_and_hash_checks_without_converter(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    import api.app as mod
    from fastapi.responses import FileResponse
    client=TestClient(mod.app)
    monkeypatch.setattr(mod, 'quarantine_record', lambda *a: {'reason':'known_bad'})
    response=client.get('/pose/known_bad/poster?source_sha256='+'a'*64)
    assert response.status_code==409
    monkeypatch.setattr(mod, 'get_pose_bvh', lambda pose: FileResponse(tmp_path/'source.bvh'))
    (tmp_path/'source.bvh').write_bytes(b'changed source')
    assert client.get('/pose/pose/poster?source_sha256='+'a'*64).status_code==409
