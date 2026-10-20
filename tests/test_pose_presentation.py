"""Offline search -> /analyze -> retrievable viewer bytes; no Gemini/AWS calls."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image
import pytest
from pydantic import ValidationError

import api.app as api
from api.models import CutResultOut
from api.pose_presentation import PosePresentationOut, PreviewAssetOut, PosePreviewOut, build_pose_presentation
from src.body_scope import BodyScope, ScopeDetection
from src.config import CFG
from tests.test_body_matching import cut
from tests.test_fullbody_routing import engine, decision, evidence


BVH = 'HIERARCHY\nROOT Hips\n{\n OFFSET 0 0 0\n CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n End Site\n {\n OFFSET 0 1 0\n }\n}\nMOTION\nFrames: 1\nFrame Time: 0.033333\n0 0 0 0 0 0\n'
DIGEST = hashlib.sha256(BVH.encode()).hexdigest()


def hit(pid='a', **extra):
    return dict(pose_id=pid, family=pid, rank_score=.01, evidence={},
        refine_allowed=False, release_allowed=False, actual_mesh_validated=False, **extra)


def report(status='ok', **extra):
    return dict(version='fullbody-dynamic-v1', route='hybrid', status=status,
        selected=[], ranking='weighted_rrf_k60_geometry_0.65_semantic_0.35', elapsed_ms=1., **extra)


def resolver(pid):
    from urllib.parse import quote
    return PreviewAssetOut(transport='library_bvh', bvh_url=f'/pose/{quote(pid,safe="")}/bvh',
        bvh_sha256=DIGEST, library_lookup_supported=True)


def generated(status='preview_ready', **extra):
    return dict(status=status, refine_allowed=False, release_allowed=False, actual_mesh_validated=False,
        candidates=[dict(pose_id='generated_a', bvh=BVH, bvh_sha256=DIGEST,
            refine_allowed=False, release_allowed=False, actual_mesh_validated=False)], **extra)


def present(data):
    original = deepcopy(data)
    result = build_pose_presentation(data, output_scope='half', resolve_library=resolver)
    assert data == original  # Projection never edits ranking, gates or generation.
    return result


def test_real_engine_report_has_renderable_group_without_legacy_promotion():
    data = engine().search(decision(), evidence())
    out = present(data)
    assert out.status == 'ready'
    assert [c.pose_id for c in out.library_candidates] == [h['pose_id'] for h in data['selected']]
    assert out.library_candidates[0].camera_kind == 'fixed_117'
    assert out.ranking.endswith('semantic_0.35')
    assert not out.auto_apply_allowed and not out.export_allowed


@pytest.mark.parametrize('status', ['error', 'blocked', 'undetermined', 'not_run'])
def test_search_failure_never_exposes_injected_generation_or_selected(status):
    data = report(status, generation=generated())
    data['selected'] = [hit()]
    out = present(data)
    assert out.status == status
    assert not out.library_candidates and not out.generated_candidates
    assert 'generation_requires_successful_empty_search' in out.issues


def test_partial_search_keeps_available_results_and_error_banner():
    data = report('partial_error'); data['selected'] = [hit()]
    out = present(data)
    assert out.status == 'partial_error' and len(out.library_candidates) == 1


def test_generated_bytes_are_inline_bound_and_never_library_lookup():
    out = present(report('no_result', generation=generated()))
    assert out.status == 'review_ready' and out.search_status == 'no_result'
    assert not out.library_candidates
    item = out.generated_candidates[0]
    assert item.pose_id is None and item.source == 'generation'
    assert item.asset.encoding == 'utf-8' and item.lookup_allowed
    assert item.kind == 'generated_review' and item.review_required
    assert item.asset.bvh == BVH and item.asset.bvh_sha256 == DIGEST
    assert item.asset.bvh_url is None and not item.asset.library_lookup_supported
    assert not item.export_allowed and not item.release_allowed


@pytest.mark.parametrize('status', ['rejected', 'error', 'timeout', 'unsupported'])
def test_failed_generation_diagnostics_never_become_viewer_candidates(status):
    out = present(report('no_result', generation=generated(status,
        diagnostics={'near_candidates':[{'bvh':BVH}], 'private_path':'/private/review'})))
    assert out.status == 'generation_failed' and not out.generated_candidates
    assert '/private/' not in out.model_dump_json()


def test_near_and_geometry_fallback_never_masquerade_as_qualified():
    data = report(near_candidates=[hit('b', match_level='near_match')])
    data['selected'] = [hit(match_level='geometry_fallback', review_required=True,
        not_exact_match=True, constraints_satisfied=False)]
    out = present(data)
    assert out.status == 'review_ready' and not out.library_candidates
    assert [c.kind for c in out.review_candidates] == ['geometry_review', 'near_match']


@pytest.mark.parametrize('selected_count', range(6))
@pytest.mark.parametrize('geometry_review', [False, True])
def test_rough_candidates_share_five_slots_without_mirror(selected_count, geometry_review):
    data = report(
        near_candidates=[hit(f'near_{i}') for i in range(2)],
        framing_references=[hit(f'framing_{i}') for i in range(5)],
    )
    data['selected'] = [hit(f'selected_{i}', review_required=geometry_review)
                        for i in range(selected_count)]
    out = present(data)
    candidates = out.library_candidates + out.review_candidates + out.generated_candidates
    expected = ([f'selected_{i}' for i in range(selected_count)]
                + [f'near_{i}' for i in range(2)]
                + [f'framing_{i}' for i in range(5)])[:5]
    assert [c.pose_id for c in candidates] == expected
    assert not out.issues


def test_five_search_candidates_do_not_append_five_near_candidates():
    data = report(near_candidates=[hit(f'near_{i}') for i in range(5)])
    data['selected'] = [hit(f'selected_{i}') for i in range(5)]
    out = present(data)
    assert [c.pose_id for c in out.library_candidates] == [f'selected_{i}' for i in range(5)]
    assert not out.review_candidates


def test_unavailable_search_preview_leaves_room_for_reference():
    data = report(near_candidates=[hit('near_0'), hit('near_1')])
    data['selected'] = [hit(f'selected_{i}') for i in range(5)]

    def resolve(pid):
        if pid == 'selected_1':
            raise OSError('unavailable')
        return resolver(pid)

    out = build_pose_presentation(data, output_scope='half', resolve_library=resolve)
    assert [c.rank for c in out.library_candidates] == [1, 3, 4, 5]
    assert [c.pose_id for c in out.review_candidates] == ['near_0']
    assert out.status == 'partial_error'


@pytest.mark.parametrize('selected_count', range(6))
def test_generated_previews_only_fill_remaining_five_slots(selected_count):
    generation = generated()
    generation['candidates'] = [
        dict(generation['candidates'][0], bvh=BVH + '\n' * i,
             bvh_sha256=hashlib.sha256((BVH + '\n' * i).encode()).hexdigest())
        for i in range(3)
    ]
    data = report('ok' if selected_count else 'no_result',
                  reason='geometry_review_fallback_not_qualified', generation=generation)
    data['selected'] = [hit(f'geometry_{i}', match_level='geometry_fallback',
                           review_required=True, constraints_satisfied=False)
                        for i in range(selected_count)]
    out = present(data)
    assert len(out.review_candidates) == selected_count
    assert len(out.generated_candidates) == min(3, 5 - selected_count)
    assert out.status == 'review_ready'
    assert not out.issues


def test_presentation_contract_rejects_more_than_five_across_groups():
    data = report(); data['selected'] = [hit(f'selected_{i}') for i in range(5)]
    raw = present(data).model_dump()
    raw['review_candidates'] = present(report(near_candidates=[hit('near')])).model_dump()['review_candidates']
    with pytest.raises(ValidationError, match='presentation_candidate_budget_exceeded'):
        PosePresentationOut.model_validate(raw)


def test_valid_generation_survives_later_geometry_review_fallback():
    data = report(reason='geometry_review_fallback_not_qualified', generation=generated())
    data['selected'] = [hit(match_level='geometry_fallback', review_required=True,
        not_exact_match=True, constraints_satisfied=False)]
    out = present(data)
    assert len(out.generated_candidates) == len(out.review_candidates) == 1
    assert not out.issues and out.status == 'review_ready'


def test_basic_reference_is_plan_not_fake_3d_artifact():
    data = report(basic_model={'kind':'basic_shoulder_head','artifact_status':'not_rendered'})
    data['route'] = 'basic_model'
    out = present(data)
    assert out.status == 'basic_reference' and out.basic_model['artifact_status'] == 'not_rendered'
    assert not out.library_candidates and not out.generated_candidates


@pytest.mark.parametrize('pid', ['../file', 'a/b', 'a\\b', '.', '\x00x', ''])
def test_paths_never_become_library_urls(pid):
    data = report(); data['selected'] = [hit(pid)]
    out = present(data)
    assert out.status == 'error' and out.issues == ['library_match_preview_unavailable']


def test_unicode_spaces_query_and_fragment_are_encoded_as_one_path_component():
    data = report(); data['selected'] = [hit('손 허리?#')]
    url = present(data).library_candidates[0].asset.bvh_url
    assert url.endswith('%EC%86%90%20%ED%97%88%EB%A6%AC%3F%23/bvh')


@pytest.mark.parametrize('bad', ['hash', 'permission', 'missing_permission', 'size'])
def test_bad_generated_payload_fails_closed(bad):
    gen = generated(); row = gen['candidates'][0]
    if bad == 'hash': row['bvh_sha256'] = '0'*64
    if bad == 'permission': row['release_allowed'] = True
    if bad == 'missing_permission': del row['refine_allowed']
    if bad == 'size': row['bvh'] = 'x'*2_000_001
    out = present(report('no_result', generation=gen))
    assert out.status == 'error' and not out.generated_candidates


def test_schema_cannot_enable_export_and_rejects_mixed_asset_transport():
    item = present(report('no_result', generation=generated())).generated_candidates[0]
    raw = item.model_dump(); raw['export_allowed'] = True
    with pytest.raises(ValidationError): PosePreviewOut.model_validate(raw)
    raw = item.asset.model_dump(); raw['bvh_url'] = '/pose/generated_a/bvh'
    with pytest.raises(ValidationError): PreviewAssetOut.model_validate(raw)


def http_result(monkeypatch, tmp_path, data, *, legacy=False, count=1, quarantine=False, changed=False):
    result = cut(count)
    if not legacy:
        result.person_candidates = [[] for _ in range(count)]
        result.person_routing = [{'fullbody_search':deepcopy(data)} for _ in range(count)]
    for d in result.descriptors: d.output_scope = ScopeDetection(BodyScope.HALF, 'vlm_person')
    pose_ids = {'a'} | {row['pose_id'] for key in ('selected', 'near_candidates', 'framing_references')
                        for row in data.get(key, [])}
    class Pipe:
        _fullbody_search = SimpleNamespace(camera=SimpleNamespace(hashes={pid:DIGEST for pid in pose_ids}))
        def process_cut(self, *args, **kwargs): return result
    bvh_path = tmp_path/'pose.bvh'; bvh_path.write_text(BVH+'\n' if changed else BVH)
    monkeypatch.setattr(api, 'get_bvh_path', lambda db,pid: str(bvh_path) if pid in pose_ids else None)
    monkeypatch.setattr(api, 'quarantine_record', lambda *a: {'reason':'test'} if quarantine else None)
    monkeypatch.setattr(CFG, 'data_dir', str(tmp_path))
    monkeypatch.setitem(api.STATE, 'pipeline', Pipe())
    monkeypatch.setitem(api.STATE, 'db_path', str(tmp_path/'fixture.db'))
    image=io.BytesIO(); Image.new('RGB',(100,200),'white').save(image, format='PNG')
    response=TestClient(api.app).post('/analyze', files={'file':('rough.png',image.getvalue(),'image/png')})
    assert response.status_code == 200, response.text
    return response.json()


def test_http_regression_empty_legacy_but_one_dynamic_candidate_can_download(monkeypatch, tmp_path):
    data=report(); data['selected']=[hit()]
    raw=http_result(monkeypatch,tmp_path,data)
    p=CutResultOut.model_validate(raw).people[0]
    assert p.candidates == [] and len(p.dynamic_search.selected) == 1
    assert len(p.pose_presentation.library_candidates) == 1
    assert p.pose_presentation.output_scope == 'half'
    asset=p.pose_presentation.library_candidates[0].asset
    download=TestClient(api.app).get(asset.bvh_url)
    assert download.status_code == 200 and download.content == BVH.encode()
    assert hashlib.sha256(download.content).hexdigest() == asset.bvh_sha256
    assert asset.thumbnail_url is None  # Never invent a thumbnail or rerender.


def test_http_five_search_plus_five_near_returns_only_five_display_candidates(monkeypatch, tmp_path):
    data = report(near_candidates=[hit(f'near_{i}') for i in range(5)])
    data['selected'] = [hit(f'selected_{i}') for i in range(5)]
    person = http_result(monkeypatch, tmp_path, data)['people'][0]
    presentation = person['pose_presentation']
    assert len(person['dynamic_search']['selected']) == 5
    assert len(person['dynamic_search']['near_candidates']) == 5
    assert [c['pose_id'] for c in presentation['library_candidates']] == [f'selected_{i}' for i in range(5)]
    assert not presentation['review_candidates'] and not presentation['generated_candidates']
    asset = presentation['library_candidates'][0]['asset']
    download = TestClient(api.app).get(asset['bvh_url'])
    assert download.status_code == 200 and download.content == BVH.encode()
    assert hashlib.sha256(download.content).hexdigest() == asset['bvh_sha256']


def test_http_generated_preview_is_typed_and_keeps_no_legacy_candidate(monkeypatch,tmp_path):
    p=http_result(monkeypatch,tmp_path,report('no_result',generation=generated()))['people'][0]
    assert p['candidates'] == [] and p['pose_presentation']['status'] == 'review_ready'
    assert p['pose_presentation']['generated_candidates'][0]['asset']['bvh'] == BVH


@pytest.mark.parametrize('cause', ['quarantined','changed'])
def test_http_unusable_assets_report_error_not_no_result(monkeypatch,tmp_path,cause):
    data=report(); data['selected']=[hit()]
    p=http_result(monkeypatch,tmp_path,data,quarantine=cause=='quarantined',changed=cause=='changed')['people'][0]
    assert len(p['dynamic_search']['selected']) == 1
    assert p['pose_presentation']['status'] == 'error'
    assert not p['pose_presentation']['library_candidates']


def test_http_legacy_and_per_person_identity_are_preserved(monkeypatch,tmp_path):
    legacy=http_result(monkeypatch,tmp_path,report(),legacy=True)['people'][0]
    assert len(legacy['candidates']) == 5 and legacy['pose_presentation'] is None
    data=report(); data['selected']=[hit()]
    result=http_result(monkeypatch,tmp_path,data,count=2)['people']
    assert [p['index'] for p in result] == [0,1]
    assert all(len(p['pose_presentation']['library_candidates']) == 1 for p in result)


def test_openapi_exposes_typed_preview_contract():
    schemas=api.app.openapi()['components']['schemas']
    assert 'PosePresentationOut' in schemas and 'PreviewAssetOut' in schemas
    assert 'pose_presentation' in schemas['PersonOut']['properties']


def test_candidate_id_binds_person_bytes_camera_scope_not_rank():
    data = report(); data['selected'] = [hit()]
    a = build_pose_presentation(data,person_index=0,output_scope='full',resolve_library=resolver)
    original = a.library_candidates[0].candidate_id
    data['selected'][0]['rank_score'] = .5
    assert build_pose_presentation(data,person_index=0,output_scope='full',resolve_library=resolver).library_candidates[0].candidate_id == original
    for index,scope in [(1,'full'),(0,'half')]:
        other=build_pose_presentation(data,person_index=index,output_scope=scope,resolve_library=resolver)
        assert other.library_candidates[0].candidate_id != original
    data['selected'][0]['camera']={'yaw':45.,'pitch':0.,'distance':4.}
    assert build_pose_presentation(data,person_index=0,output_scope='full',resolve_library=resolver).library_candidates[0].candidate_id != original


@pytest.mark.parametrize('distance', [None,4.,8.])
def test_normalized_camera_reproduces_real_search_projection(distance):
    import numpy as np
    from api.pose_presentation import render_camera
    from src.camera_search import Camera,project
    from src.library import _synthetic_pose
    points=_synthetic_pose('standing')
    normalized=render_camera(dict(yaw=45.,pitch=-30.,distance=distance),'fixed_117')
    hip=(points[11]+points[12])/2; shoulder=(points[5]+points[6])/2
    p=(points-hip)/np.linalg.norm(shoulder-hip)
    p=p@np.asarray(normalized.rotation).T
    xy=p[:,:2]*[1,-1]
    if distance is not None:xy=xy/(p[:,2]+distance)[:,None]
    expected,_=project(points,Camera(45.,-30.,distance))
    np.testing.assert_allclose(xy,expected,atol=1e-6)


def test_semantic_camera_is_not_claimed_as_pixel_fitted_search_view():
    import numpy as np
    from api.pose_presentation import render_camera
    from src.semantic_shapes import rotate_for_camera
    raw=dict(yaw_deg=45.,pitch_deg=30.,roll_deg=0.)
    cam=render_camera(raw,'semantic_preview')
    points=np.eye(3)
    np.testing.assert_allclose(points@np.asarray(cam.rotation).T,rotate_for_camera(points,raw),atol=1e-9)
    assert not cam.projection_matches_search and not cam.bvh_camera_rotation_applied


def test_generated_source_and_pose_id_cannot_be_confused():
    item=present(report('no_result',generation=generated())).generated_candidates[0].model_dump()
    item['pose_id']='generated_a'
    with pytest.raises(ValidationError):PosePreviewOut.model_validate(item)
