import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
import threading

from converter_api.skin_asset_cache import SkinAssetVerificationCache


def asset(root, name='approved.bin', data=b'approved'):
    (root/name).write_bytes(data)
    return dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def test_warm_contract_checks_identity_without_reading_bytes(tmp_path):
    entries=[asset(tmp_path),asset(tmp_path,'other.bin')]
    cache=SkinAssetVerificationCache()
    ok,cold=cache.check(entries,str(tmp_path),'manifest-a')
    assert ok and cold['hashed_files']==2 and cold['hashed_bytes']==16
    ok,warm=cache.check(entries,str(tmp_path),'manifest-a')
    assert ok and warm['cache_hits']==2 and warm['hashed_bytes']==0


def test_same_size_tampering_with_restored_mtime_is_rejected_and_repair_recovers(tmp_path):
    entry=asset(tmp_path);cache=SkinAssetVerificationCache()
    assert cache.check([entry],str(tmp_path),'a')[0]
    path=tmp_path/entry['path'];old=path.stat()
    path.write_bytes(b'tampered');os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns))
    assert not cache.check([entry],str(tmp_path),'a')[0]
    path.write_bytes(b'approved')
    assert cache.check([entry],str(tmp_path),'a')[0]


def test_atomic_replace_delete_manifest_and_root_changes_invalidate(tmp_path):
    entry=asset(tmp_path);cache=SkinAssetVerificationCache()
    assert cache.check([entry],str(tmp_path),'a')[0]
    replacement=tmp_path/'replacement';replacement.write_bytes(b'tampered')
    replacement.replace(tmp_path/entry['path'])
    assert not cache.check([entry],str(tmp_path),'a')[0]
    asset(tmp_path)
    assert cache.check([entry],str(tmp_path),'a')[0]
    assert cache.check([entry],str(tmp_path),'b')[1]['hashed_files']==1
    other=tmp_path/'other';other.mkdir();asset(other)
    assert cache.check([entry],str(other),'b')[1]['hashed_files']==1
    (other/entry['path']).unlink()
    assert not cache.check([entry],str(other),'b')[0]
    assert not cache.check([entry],'','b')[0]


def test_symlink_and_parent_escape_cannot_inherit_verified_identity(tmp_path):
    root=tmp_path/'root';root.mkdir();entry=asset(root)
    cache=SkinAssetVerificationCache();assert cache.check([entry],str(root),'a')[0]
    outside=tmp_path/'outside';outside.write_bytes(b'approved')
    (root/entry['path']).unlink();(root/entry['path']).symlink_to(outside)
    assert not cache.check([entry],str(root),'a')[0]
    nested=root/'nested';nested.symlink_to(tmp_path,target_is_directory=True)
    escaped={**entry,'path':'nested/outside'}
    assert not cache.check([escaped],str(root),'a')[0]


def test_concurrent_cold_requests_hash_each_file_once(tmp_path):
    entry=asset(tmp_path);cache=SkinAssetVerificationCache()
    barrier=threading.Barrier(8)
    def check(_):
        barrier.wait()
        return cache.check([entry],str(tmp_path),'a')
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows=list(pool.map(check,range(8)))
    assert all(ok for ok,_ in rows)
    assert sum(row['hashed_files'] for _,row in rows)==1
    assert sum(row['cache_hits'] for _,row in rows)==7


def test_mutation_during_verification_is_never_published(tmp_path,monkeypatch):
    entry=asset(tmp_path);cache=SkinAssetVerificationCache()
    original=cache._identity;calls=0
    def changing(path,root):
        nonlocal calls
        calls+=1
        if calls==2:path.write_bytes(b'tampered')
        return original(path,root)
    monkeypatch.setattr(cache,'_identity',changing)
    assert not cache.check([entry],str(tmp_path),'a')[0]
    assert not cache._verified


def test_changed_expected_descriptor_cannot_reuse_evidence(tmp_path):
    entry=asset(tmp_path);cache=SkinAssetVerificationCache()
    assert cache.check([entry],str(tmp_path),'a')[0]
    assert not cache.check([{**entry,'sha256':'0'*64}],str(tmp_path),'a')[0]


def test_http_contract_reuses_evidence_but_asset_delivery_rechecks_bytes(tmp_path,monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from converter_api.app import ApiProblem, _error_response
    from converter_api import skin
    entry=asset(tmp_path)
    digest='c'*64
    data=dict(revision='test',models={'body':dict(character_sha256=digest,uv=entry,
        textures={'flesh':entry},preview_textures={'flesh':entry})})
    monkeypatch.setattr(skin,'manifest',lambda:data)
    monkeypatch.setattr(skin,'manifest_hash',lambda:'a'*64)
    monkeypatch.setattr(skin,'runtime_revision',lambda:'b'*64)
    monkeypatch.setattr(skin,'certified',lambda *args:True)
    monkeypatch.setenv('CONVERTER_SKIN_ASSET_ROOT',str(tmp_path))
    monkeypatch.setenv('CONVERTER_INTERACTIVE_TOKEN','t'*32)
    app=FastAPI()
    @app.exception_handler(ApiProblem)
    async def error(request,problem):return _error_response(problem)
    calls=[]
    original=SkinAssetVerificationCache._verify
    def verify(self,*args):
        calls.append(1)
        return original(self,*args)
    monkeypatch.setattr(SkinAssetVerificationCache,'_verify',verify)
    skin.register_skin(app,None,None)
    client=TestClient(app)
    assert client.get('/skin-contract').json()['assets_available']
    assert client.get('/skin-contract').json()['assets_available']
    assert len(calls)==1
    (tmp_path/entry['path']).write_bytes(b'tampered')
    response=client.get('/skin-assets/'+entry['sha256'],headers={'Authorization':'Bearer '+'t'*32})
    assert response.status_code==503
    assert not client.get('/skin-contract').json()['assets_available']
    asset(tmp_path)
    assert client.get('/skin-contract').json()['assets_available']
