"""Real HTTP contracts on both apps; fake Blender output is explicitly a fixture."""
import copy
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import api.app as inference
from api.body_handoff import BodyRenderRequest, apply_refine_result, build_auto_render_request
from api.body_render import ConverterBodyRenderer, _RENDER_SLOT
from api.models import CutResultOut, RefineResponse
from converter.protocol import SOLVER_VERSION, THUMBNAIL_RENDERER_VERSION
from converter_api.app import create_app
from converter_api.registry import CharacterRegistry
from src.config import CFG


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from tests.test_body_api_contract import BodyAPIContractTests
    from tests.converter.test_api_contract import FakeRunner, VALID_BVH
    case = BodyAPIContractTests()
    case.setUp()
    for a in case.raw['assets']:
        a['metadata'] = {'converter_character_id': a['body_id']}
    case.write()
    response = CutResultOut.model_validate(case.response())
    batch = build_auto_render_request(response, 0, request_id='body-test',
                                     retarget_version=SOLVER_VERSION, renderer_version=THUMBNAIL_RENDERER_VERSION)
    for pose in batch.poses:
        path = case.root / f'{pose.pose_id}.bvh'
        path.write_bytes(VALID_BVH)
        pose.pose_sha256 = hashlib.sha256(VALID_BVH).hexdigest()
    # Registry matches the same FBX bytes as the selected catalog entry.
    rows = {}
    env = {}
    for a in case.raw['assets']:
        key = a['body_id'].upper() + '_URI'
        env[key] = str(case.root / a['fbx_path'])
        rows[a['body_id']] = dict(display_name=a['body_id'], artifact_uri_env=key,
                                  sha256=a['asset_sha256'], rig_profile='mixamo', revision='v1')
    registry_path = tmp_path / 'characters.json'
    registry_path.write_text(json.dumps(dict(schema_version=1, characters=rows)))
    registry = CharacterRegistry(registry_path, default_character_id='regular', environ=env)
    runner = FakeRunner()
    converter = TestClient(create_app(registry=registry, runner=runner))
    def opener(req, timeout):
        res = converter.post('/render-thumbnail', content=req.data, headers=dict(req.header_items()))
        if res.status_code != 200:
            from urllib.error import HTTPError
            raise HTTPError(req.full_url, res.status_code, 'fixture error', {}, io.BytesIO(res.content))
        obj = io.BytesIO(res.content)
        obj.status = res.status_code
        obj.headers = res.headers
        return obj
    renderer = ConverterBodyRenderer('http://converter.internal', opener=opener)
    monkeypatch.setenv('BODY_RENDER_CONVERTER_URL', 'http://converter.internal')
    monkeypatch.setattr(CFG, 'body_catalog_path', str(case.path))
    monkeypatch.setattr(inference, 'get_bvh_path', lambda db, pose: str(case.root / (pose + '.bvh')))
    with patch('api.body_render.ConverterBodyRenderer', return_value=renderer):
        yield TestClient(inference.app), batch, runner, case, converter, renderer
    case.doCleanups()


def test_auto_five_poses_reach_selected_character_in_order(setup):
    client, batch, runner, _, _, _ = setup
    response = client.post('/body/render', json=batch.model_dump(mode='json'))
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['body']['body_id'] == 'muscular'
    assert data['character_id'] == 'muscular'
    assert [p['pose_id'] for p in data['previews']] == [p.pose_id for p in batch.poses]
    assert len(runner.calls) == 5
    assert all(c['character_id'] == 'muscular' for c in runner.calls)
    assert all(c['character_sha256'] == batch.body.asset_sha256 for c in runner.calls)
    assert all(p['data'] for p in data['previews'])


def test_refine_body_is_sent_instead_of_original_url(setup):
    from tests.converter.test_api_contract import VALID_BVH
    client, batch, runner, _, _, _ = setup
    text = VALID_BVH.decode() + '\n'
    result = RefineResponse(pose_id=batch.poses[2].pose_id, view=batch.poses[2].view,
                            refined=True, reason='ok', bvh_url=batch.poses[2].bvh_url,
                            bvh=text, backend='numpy')
    updated = apply_refine_result(batch, 2, result, request_id='refined-test', resolved_pose_sha256='0'*64)
    res = client.post('/body/render', json=updated.model_dump(mode='json'))
    assert res.status_code == 200, res.text
    assert runner.calls[2]['bvh_bytes'] == text.encode()
    assert runner.calls[0]['bvh_bytes'] == VALID_BVH
    assert updated.poses[2].bvh_url is None
    assert res.json()['selection_revision'] == 1
    assert res.json()['body'] == batch.body.model_dump(mode='json')


def test_manual_body_change_uses_new_fbx_same_pose_sources(setup):
    client, batch, runner, case, _, _ = setup
    asset = next(a for a in case.raw['assets'] if a['body_id'] == 'slim')
    data = batch.model_dump(mode='json')
    data.update(request_id='manual-test', selection_revision=1, selection_source='user_override')
    data['body'] = {k: asset[k] for k in data['body']}
    res = client.post('/body/render', json=data)
    assert res.status_code == 200, res.text
    assert all(c['character_id'] == 'slim' for c in runner.calls)
    assert [p['source_pose_sha256'] for p in res.json()['previews']] == [p.pose_sha256 for p in batch.poses]


@pytest.mark.parametrize('mutation,status', [
    ('catalog',409), ('asset',409), ('pose_hash',409), ('url',422), ('order',422),
    ('missing_pose',404), ('unregistered',409), ('unapproved',409), ('quarantined',409),
])
def test_invalid_batch_rejected_before_converter(setup, mutation, status):
    client, batch, runner, case, _, _ = setup
    data = batch.model_dump(mode='json')
    if mutation == 'catalog': data['catalog_sha256'] = '0'*64
    if mutation == 'asset': data['body']['asset_sha256'] = '0'*64
    if mutation == 'pose_hash': data['poses'][-1]['pose_sha256'] = '0'*64
    if mutation == 'url': data['poses'][0]['bvh_url'] = 'http://external.invalid/secret'
    if mutation == 'order': data['poses'].reverse()
    if mutation == 'missing_pose': (case.root / 'pose-4.bvh').unlink()
    if mutation in {'unregistered','unapproved'}:
        a = next(a for a in case.raw['assets'] if a['body_id']=='muscular')
        if mutation == 'unregistered': a['metadata'] = {}
        else: a['availability'] = 'qa_pending'
        case.write()
        data['catalog_sha256'] = hashlib.sha256(case.path.read_bytes()).hexdigest()
    with patch('api.body_render.quarantine_record', return_value={} if mutation=='quarantined' else None):
        res = client.post('/body/render', json=data)
    assert res.status_code == status, res.text
    assert not runner.calls


def test_converter_rejects_wrong_body_hash_before_blender(setup):
    from tests.converter.test_api_contract import VALID_BVH
    _, _, runner, _, converter, _ = setup
    res = converter.post('/render-thumbnail', data={'character_id':'muscular', 'expected_character_sha256':'0'*64},
                         files={'bvh':('pose.bvh',VALID_BVH)})
    assert res.status_code == 409
    assert not runner.calls


def test_missing_refined_body_cannot_fall_back_silently(setup):
    _, batch, _, _, _, _ = setup
    res = RefineResponse(pose_id=batch.poses[0].pose_id,view=batch.poses[0].view,
                         refined=True,reason='ok',bvh_url=batch.poses[0].bvh_url,backend='numpy')
    with pytest.raises(ValidationError):
        apply_refine_result(batch,0,res,request_id='bad-refine',resolved_pose_sha256='0'*64)


@pytest.mark.parametrize('header', ['X-Standin-Character-SHA256','X-Standin-Source-BVH-SHA256',
                                    'X-Standin-Solver-Version','X-Standin-Thumbnail-View'])
def test_wrong_converter_identity_returns_no_cards(setup, header):
    client,batch,_,_,_,renderer=setup
    original=renderer.open
    def corrupt(req,timeout):
        response=original(req,timeout)
        response.headers[header]='wrong'
        return response
    renderer.open=corrupt
    res=client.post('/body/render',json=batch.model_dump(mode='json'))
    assert res.status_code==502
    assert 'previews' not in res.json()


def test_mid_batch_failure_returns_no_partial_success(setup):
    client,batch,runner,_,_,_=setup
    from converter_api.runner import ConversionTimeoutError
    original=runner.convert
    def fail_third(**kwargs):
        if len(runner.calls)==2:raise ConversionTimeoutError('test timeout')
        return original(**kwargs)
    runner.convert=fail_third
    res=client.post('/body/render',json=batch.model_dump(mode='json'))
    assert res.status_code==504, res.text
    assert 'previews' not in res.json()


def test_disabled_and_busy_do_not_call_converter(setup,monkeypatch):
    client,batch,runner,_,_,_=setup
    _RENDER_SLOT.acquire()
    try:
        res=client.post('/body/render',json=batch.model_dump(mode='json'))
        assert res.status_code==503 and res.headers['retry-after']=='2'
    finally:_RENDER_SLOT.release()
    monkeypatch.delenv('BODY_RENDER_CONVERTER_URL')
    assert client.post('/body/render',json=batch.model_dump(mode='json')).status_code==503
    assert not runner.calls


def test_preview_result_rejects_stale_or_partial_cards(setup):
    from api.body_render import BodyPreviewResult, preview_result_matches
    client,batch,_,_,_,_=setup
    res=client.post('/body/render',json=batch.model_dump(mode='json'))
    result=BodyPreviewResult.model_validate(res.json())
    assert preview_result_matches(batch,result)
    for field,value in [('selection_revision',99),('request_id','old'),('catalog_sha256','0'*64),
                        ('renderer_version','old'),('previews',result.previews[:-1])]:
        assert not preview_result_matches(batch,result.model_copy(update={field:value}))


def test_refined_hash_and_dual_source_rejected(setup):
    client,batch,runner,_,_,_=setup
    data=batch.model_dump(mode='json')
    data['poses'][0].update(kind='refined',bvh='fixture text',bvh_url=None)
    assert client.post('/body/render',json=data).status_code==422
    data['poses'][0]['pose_sha256']=hashlib.sha256(b'fixture text').hexdigest()
    data['poses'][0]['bvh_url']='/pose/pose-0/bvh'
    assert client.post('/body/render',json=data).status_code==422
    assert not runner.calls
