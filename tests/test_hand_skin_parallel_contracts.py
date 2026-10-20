"""Offline contract comparisons; frozen providers never call a model/network.

Real Blender/ASGI checks live in replay_search.py. These tests exercise policy
branches deterministically, including cases unavailable in the six BVH replay.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from api.pose_presentation import build_pose_presentation
from src.config import CFG
from src.experimental.text_pose_generation import GenerationResult
from src.fullbody_routing import FullBodySearch
from src.mirror_reference import mirror_reference
from tests.test_fullbody_routing import Camera, Semantic, decision, evidence
from tests.test_mirror_reference import fixture as mirror_fixture
from tests.test_pose_presentation import hit
from tests.test_hand_skin_parallel import Checker, gate
from converter.hand_contact import validate_screen

def normalize(v):
    if isinstance(v,dict):
        return {k:normalize(x) for k,x in v.items() if k not in {
            'elapsed_ms','availability','screened_artifact_sha256','checks_diagnostics',
            'wall_elapsed_ms','discarded_speculative','policy'}}
    if isinstance(v,list):return [normalize(x) for x in v]
    return v


@pytest.fixture(autouse=True)
def bind_unchanged_policy(monkeypatch):
    monkeypatch.setattr(CFG,'refine_pose_quarantine_path',
        str(Path(__file__).resolve().parents[1]/'config/refine_pose_quarantine.v1.json'))


def asset(pid):
    return dict(transport='library_bvh',bvh_url=f'/pose/{pid}/bvh',
        bvh_sha256=hashlib.sha256(('BVH '+pid).encode()).hexdigest(),library_lookup_supported=True)


def present(raw,scope):
    before=deepcopy(raw)
    p=build_pose_presentation(raw,output_scope=scope,resolve_library=asset)
    assert raw==before
    assert not p.issues
    assert not any((p.auto_apply_allowed,p.refine_allowed,p.export_allowed))
    for c in p.library_candidates+p.generated_candidates+p.review_candidates:
        assert not any((c.auto_apply_allowed,c.refine_allowed,c.export_allowed,c.release_allowed,c.actual_mesh_validated))
    return p


@pytest.mark.parametrize('scope',['full','half','bust','head','shoulder_head'])
@pytest.mark.parametrize('route',['hybrid','semantic','geometric'])
@pytest.mark.parametrize('fault',['clear','intersection','unavailable','search_error'])
def test_search_to_presentation_contracts_same(tmp_path,scope,route,fault):
    results=[]
    for parallel in (False,True):
        statuses={'b':fault} if fault in ('intersection','unavailable') else {}
        checker=Checker(statuses)
        semantic=Semantic(status='raise' if fault=='search_error' else None)
        e=FullBodySearch([SimpleNamespace(pose_id=p,meta={}) for p in ('a','b','c')],
            semantic,Camera(),semantic_weight=.6,hand_skin_gate=gate(tmp_path,checker,parallel))
        ev=evidence();ev.update(output_scope=scope,hand_presets={'left':'relaxed','right':'fist'})
        d=decision(sufficient=route!='semantic',text=route!='geometric')
        raw=e.search(d,ev);p=present(raw,scope)
        assert p.output_scope==scope and raw['ranking'].endswith('semantic_0.6')
        assert [c.pose_id for c in p.library_candidates]==[r['pose_id'] for r in raw['selected']]
        for c in p.library_candidates:
            assert c.asset.model_dump()==asset(c.pose_id) | dict(bvh=None,encoding=None,thumbnail_url=None,
                thumbnail_view=None,thumbnail_is_search_view=False)
            assert c.render_camera.projection=='orthographic'
            assert c.render_camera.yaw_degrees==45 if c.camera_kind=='fixed_117' else c.render_camera.yaw_degrees==0
        results.append((raw,p))
    a,b=results
    assert normalize(a[1].model_dump())==normalize(b[1].model_dump())
    for key in ('status','reason','ranking','branches','selected','generation'):
        assert normalize(a[0].get(key))==normalize(b[0].get(key))


class FrozenGenerator:
    def __init__(self,status):self.status=status;self.calls=0
    def generate(self,description):
        self.calls+=1
        if self.status=='raise':raise RuntimeError('frozen local exception')
        return GenerationResult(self.status,reason='frozen_fixture',candidates=(
            [dict(bvh='BVH generated',refine_allowed=False,release_allowed=False,actual_mesh_validated=False)]
            if self.status=='preview_ready' else []))


@pytest.mark.parametrize('status',['preview_ready','error','timeout','unsupported','rejected','raise'])
@pytest.mark.parametrize('screen',['clear_screen','intersection','unavailable'])
def test_frozen_generation_errors_and_bytes_contract(tmp_path,status,screen):
    results=[]
    for parallel in (False,True):
        generator=FrozenGenerator(status)
        checker=Checker({'generated':screen})
        e=FullBodySearch([SimpleNamespace(pose_id=p,meta={}) for p in ('a','b','c')],
            Semantic(()),Camera(()),generator=generator,semantic_weight=.6,
            hand_skin_gate=gate(tmp_path,checker,parallel))
        raw=e.search(decision(),evidence());p=present(raw,'half')
        assert generator.calls==1
        if p.generated_candidates:
            [c]=p.generated_candidates
            assert status=='preview_ready' and screen=='clear_screen'
            assert c.asset.bvh=='BVH generated' and c.asset.transport=='inline_bvh'
            assert c.asset.bvh_sha256==hashlib.sha256(c.asset.bvh.encode()).hexdigest()
            assert c.pose_id is None and c.asset.bvh_url is None and not c.asset.library_lookup_supported
        else:assert not (raw.get('generation') or {}).get('candidates')
        results.append((raw,p))
    assert normalize(results[0][1].model_dump())==normalize(results[1][1].model_dump())
    for key in ('status','reason','generation'):
        assert normalize(results[0][0].get(key))==normalize(results[1][0].get(key))


@pytest.mark.parametrize('status',['clear_screen','intersection','unavailable'])
def test_real_mirror_policy_with_session_preserves_main_and_optional_priority(tmp_path,status):
    results=[]
    for parallel in (False,True):
        e,state,d,required,_=mirror_fixture()
        checker=Checker({'right':status})
        session=gate(tmp_path,checker,parallel,ids=('left','right','other')).session()
        for pid in ('left','right','other'):
            sha=asset(pid)['bvh_sha256']
            e.camera.hashes[pid]=sha
            e.semantic.core_records[pid]['meta']['bvh_sha256']=sha
            e.semantic.body[pid]['source_sha256']=sha
        selected=session.filter([hit('left')]);main=deepcopy(session.report())
        with session.optional_phase():row,comparison=mirror_reference(e,selected,d,required,session)
        assert selected[0]['hand_skin_screen']['status']=='clear_screen' and not main['unverified']
        assert [c[3] for c in checker.calls]==[False,True]
        raw=dict(status='ok',route='hybrid',ranking='weighted_rrf_k60_geometry_0.4_semantic_0.6',
            selected=selected,mirror_comparison=comparison,mirror_references=[row] if row else [])
        p=present(raw,'bust')
        assert p.status=='ready'
        if status=='clear_screen':
            [ref]=p.review_candidates
            assert ref.kind=='mirror_reference' and ref.pose_id=='right' and ref.rank_score is None
            assert ref.mirror_comparison.original_candidate_id==p.library_candidates[0].candidate_id
            assert not ref.render_camera.projection_matches_search
        else:assert not p.review_candidates and comparison['status']=='unavailable'
        results.append(p)
    assert normalize(results[0].model_dump())==normalize(results[1].model_dump())
