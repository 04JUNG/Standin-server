import hashlib

import pytest

from scripts.prevalidate_hand_screen_library import prepare
from tests.test_hand_screen_catalog import Client


def manifest(tmp_path):
    fixtures=[]
    for pid in ('a','b'):
        raw=('BVH '+pid).encode();(tmp_path/(pid+'.bvh')).write_bytes(raw)
        fixtures.append(dict(id=pid,bvh=pid+'.bvh',sha256=hashlib.sha256(raw).hexdigest()))
    return dict(fixtures=fixtures,hand_settings=[dict(id='original',hands=None)])


def test_offline_build_contains_both_positive_and_negative_certificates(tmp_path):
    client=Client({'b':'intersection'})
    result=prepare(client,manifest(tmp_path),tmp_path)
    assert len(result['entries'])==2 and len(client.calls)==2
    assert {r['receipt']['status'] for r in result['entries'].values()}=={'clear_screen','intersection'}
    assert client.state.profile_calls==2


@pytest.mark.parametrize('field,value',[('character_id','different'),('character_sha256','d'*64),
    ('hand_library_sha256','d'*64),('shape_sha256','d'*64)])
def test_cannot_build_certificates_for_a_different_runtime_body(tmp_path,field,value):
    client=Client();client.profile[field]=value
    with pytest.raises(ValueError):prepare(client,manifest(tmp_path),tmp_path)
    assert not client.calls


def test_source_bytes_must_match_operator_manifest(tmp_path):
    data=manifest(tmp_path);(tmp_path/'a.bvh').write_text('BVH changed')
    client=Client()
    with pytest.raises(ValueError):prepare(client,data,tmp_path)
    assert not client.calls


def test_profile_change_during_build_aborts_publication(tmp_path):
    client=Client();original=client.screening_profile
    def changed(timeout):
        result=original(timeout)
        if client.state.profile_calls>1:result['profile_sha256']='d'*64
        return result
    client.screening_profile=changed
    with pytest.raises(ValueError):prepare(client,manifest(tmp_path),tmp_path)


def test_transient_failure_is_never_published_as_a_certificate(tmp_path):
    with pytest.raises(ValueError):prepare(Client({'a':'unavailable'}),manifest(tmp_path),tmp_path)
