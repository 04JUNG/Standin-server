"""Observable geometry, output agreement and review gates for local matching."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pose_curation.orientation import Orientation, write_oriented_bvh
from pose_curation.scoped.geometry import features
from pose_curation.scoped.index import representatives, snapshot, ScopedIndex
from pose_curation.scoped.matching import observed, projection, rank_angles
from pose_curation.review.scoped_routes import scoped_router
from pose_curation.storage import write_json, sha256
from tests.test_pose_curation import make_motion, library
from pose_curation.motion import Motion


@pytest.fixture
def pose(tmp_path):
    source, target = tmp_path / 'motion.bvh', tmp_path / 'pose.bvh'
    make_motion(source)
    Motion.load(source).export(90, target)
    return target


def test_grouping_ignores_root_orientation_but_export_projection_agrees(pose, tmp_path):
    before = features(pose)
    target = tmp_path / 'rotated.bvh'
    spec = Orientation(37,25,-18)
    write_oriented_bvh(pose, target, spec)
    after = features(target)
    assert np.allclose(before['half'], after['half'], atol=1e-6)
    assert np.allclose(before['bust'], after['bust'], atol=1e-6)
    assert np.allclose(projection(before['body'], spec), projection(after['body'], Orientation()), atol=2e-5)


@pytest.mark.parametrize('angles', [(37,25,-18),(-65,-33,21),(120,50,-125)])
def test_continuous_angle_fit_recovers_known_projection(angles):
    body = np.zeros((17,3))
    body[5:11] = [[1,2,0],[-1,2,0],[1.4,1,1.1],[-1.8,2.7,.6],[1.6,2.3,1.8],[-.9,3.4,1.1]]
    query = projection(body, Orientation(*angles)) * 100 + [320,280]
    scores = np.ones(17)
    mask = observed(query, scores)
    result = rank_angles(np.stack([body]), query, scores, mask, [0])[0]
    assert result['error'] < 5e-4
    spec = Orientation(**{k:result['orientation'][k] for k in ['yaw','pitch','roll']})
    predicted = projection(body,spec)*result['scale']+result['translation']
    assert np.max(np.abs(predicted[mask]-query[mask])) < .12
    # Matching may return a depth-ambiguous orientation; only projected error
    # and the output convention can be asserted from a single 2D observation.


def test_face_hidden_legs_and_low_score_arm_are_not_observations():
    body = np.zeros((17,2)); body[5:11] = [[100,100],[200,100],[80,170],[240,140],[50,230],[240,210]]
    scores = np.ones(17);scores[9]=.1
    mask=observed(body,scores)
    assert np.flatnonzero(mask).tolist()==[5,6,7,8,10]
    changed=body.copy();changed[:5]=np.nan;changed[11:]=np.nan;changed[9]=1e8
    assert np.array_equal(observed(changed,scores), mask)
    scores[7:11]=.1
    with pytest.raises(ValueError):observed(body,scores)


def test_cropped_bust_elbows_without_wrists_do_not_enable_angle_matching():
    points=np.zeros((17,2));points[5:11]=[[600,900],[50,1000],[620,1250],[30,1250],[620,1200],[-80,1200]]
    scores=np.zeros(17);scores[5:11]=[.54,.55,.35,.38,.25,.17]
    with pytest.raises(ValueError):observed(points,scores,[690,1336])
    scores[5:11]=1
    points[9]=[1000,1600]  # both wrists outside the image
    with pytest.raises(ValueError):observed(points,scores,[690,1336])


def test_representatives_are_real_and_cover_variants_deterministically():
    values=np.array([[0.,0.],[0.,0.],[1.,1.],[1.01,1.],[5.,5.]])
    chosen,assignments,distances=representatives(values,10,.02)
    again=representatives(values,10,.02)
    assert len(chosen)==3 and chosen==again[0]
    assert np.max(distances)<=.02
    assert np.array_equal(assignments, again[1])
    assert assignments[0]==assignments[1]


def test_review_and_source_revision_invalidate_index(pose,tmp_path):
    item=SimpleNamespace(key='pose',content_hash=sha256(pose),bvh=pose,metadata={})
    catalog=SimpleNamespace(all=lambda:[item])
    review={('pose',item.content_hash):{'status':'accepted'}}
    before=snapshot(catalog,review)
    review[('pose',item.content_hash)]['status']='rejected'
    assert snapshot(catalog,review)!=before


def test_routes_require_same_origin_and_only_observable_scope(tmp_path):
    index=SimpleNamespace(read=lambda: (_ for _ in ()).throw(ValueError('stale')))
    app=FastAPI();app.include_router(scoped_router(tmp_path,tmp_path,None,None,index))
    client=TestClient(app)
    assert client.get('/api/scoped').status_code==409
    assert client.get('/api/scoped/queries').json()=={'items':[]}
    assert client.post('/api/scoped/match',json={'query':'rough_001:0'}).status_code==403
    assert client.post('/api/scoped/match',headers={'X-Pose-Review':'1','Origin':'http://evil.test'},json={'query':'rough_001:0'}).status_code==403
    assert client.post('/api/scoped/match',headers={'X-Pose-Review':'1'},json={'query':'rough_001:0','scope':'head'}).status_code==422
    assert client.post('/api/scoped/match',headers={'X-Pose-Review':'1'},json={'query':'../foo:0'}).status_code==422


def test_library_filter_rejects_a_stale_review_instead_of_returning_excluded_pose(library):
    from pose_curation.scoped.index import build
    from pose_curation.review.app import create_app
    from pose_curation.review.store import ReviewStore
    data, _ = library
    curation=data/'curation'
    manifest=build(data,curation)
    assert manifest['source_count']==1
    with TestClient(create_app(data,curation),base_url='http://127.0.0.1') as client:
        assert client.get('/api/poses?library_scope=half').json()['total']==1
        source=manifest['sources'][0]
        ReviewStore(curation/'reviews.sqlite').save(source['key'],source['content_hash'],'rejected','exclude')
        assert client.get('/api/poses?library_scope=half').status_code==409
        assert client.get('/api/poses').json()['total']==0
