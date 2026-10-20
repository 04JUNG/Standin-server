"""Opt-in local real-library/Blender check; never downloads or rewrites assets.

MIRROR_ASSET_ROOT points to a mounted release (poses.db, bvh, tags, caches).
MIRROR_CHARACTER_FBX supplies the existing registered standin-master-v2 FBX.
No VLM, private rough, deployed service or receipt stub participates in this test.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace
from urllib.parse import urlparse

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not (os.getenv('MIRROR_ASSET_ROOT') and os.getenv('MIRROR_CHARACTER_FBX')),
    reason='requires explicitly mounted real library and registered FBX + Blender')


def test_actual_registered_pair_search_skin_and_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import api.app as api
    from api.pose_presentation import build_pose_presentation
    from converter_api.app import create_app
    from converter_api.registry import CharacterRegistry
    from src.camera_search import CameraIndex
    from src.fullbody_routing import FullBodySearch
    from src.hand_skin_screen import HandSkinGate, HandSkinClient
    from src.search_routing import (PoseDescription, JointEvidence, RoutingConfig,
        assess_semantic_evidence, decide_route)
    from src.semantic_adapter_285 import build_semantic285_searcher, excluded_pose_ids

    root = Path(os.environ['MIRROR_ASSET_ROOT']).resolve()
    with sqlite3.connect((root/'poses.db').as_uri()+'?mode=ro', uri=True) as conn:
        entries = [SimpleNamespace(pose_id=p, meta=json.loads(m))
                   for p, m in conn.execute('SELECT pose_id, meta_json FROM poses')]
    allowed = {e.pose_id for e in entries} - excluded_pose_ids()
    semantic = None
    for cache in sorted((root/'fullbody-cache').iterdir()):
        semantic = build_semantic285_searcher(tags_path=root/'search-v2.json',
            taxonomy_path=root/'search-taxonomy-v2.json', db_path=root/'poses.db',
            library_pose_ids=allowed, body_index_path=cache/'body.json', shape_index_path=cache/'shapes.json')
        if hasattr(semantic, 'service'):
            break
    assert hasattr(semantic, 'service'), getattr(semantic, 'reason', 'missing validated cache')
    camera = CameraIndex.from_library(allowed, root/'bvh')
    assert not camera.errors
    registry = CharacterRegistry(Path(__file__).resolve().parents[1]/'config/characters.example.json',
        environ={'STANDIN_MASTER_V2_URI': os.environ['MIRROR_CHARACTER_FBX']})
    converter = TestClient(create_app(registry=registry))
    receipts = []

    class Response:
        def __init__(self, response):
            self.status, self.data = response.status_code, response.content
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit): return self.data[:limit]

    def opener(request, timeout):
        response = converter.post(urlparse(request.full_url).path,
            content=request.data, headers=dict(request.headers))
        assert response.status_code == 200, response.text
        receipts.append(response.json())
        return Response(response)

    skin = HandSkinClient('http://local-converter', 'standin-master-v2',
        registry.metadata('standin-master-v2').sha256, opener=opener)
    engine = FullBodySearch(entries, semantic, camera,
        hand_skin_gate=HandSkinGate(skin, root/'bvh', camera.hashes))
    desc = PoseDescription(certainty='high', actions=['raise_arm'],
        arm_states={'left': 'raised', 'right': 'unknown'}, visible_scope='half',
        search_text='왼팔을 머리 위로 들어 올린 자세', body_details='왼팔을 머리 위로 들어 올린다')
    sem = assess_semantic_evidence(desc, RoutingConfig())
    assert sem.valid
    decision = decide_route(JointEvidence(True, None, [], {}, {}), sem)
    # Deterministic library projection, explicitly not a real observed rough.
    i = camera.pose_ids.index('b9_raise_hand_stand_mirror')
    mask = np.zeros(17, bool)
    mask[[5, 6, 7, 9, 11, 12, 13, 15]] = True
    report = engine.search(decision, dict(keypoints=camera.projections[0, i].copy(),
        joint_mask=mask, coverage_class='full', output_scope='half'))
    assert report['status'] == 'ok', report.get('reason')
    assert report['mirror_comparison']['status'] == 'ready'
    assert not report['hand_skin_screen']['unverified']
    assert not report['mirror_hand_skin_screen']['unverified']

    # Relocate only a COPY for the actual API registry/download handler.
    db = tmp_path/'poses.db'
    shutil.copyfile(root/'poses.db', db)
    with sqlite3.connect(db) as conn:
        conn.executemany('UPDATE poses SET bvh_path=? WHERE pose_id=?',
            [(str(root/'bvh'/(pid+'.bvh')), pid) for pid in allowed])
    monkeypatch.setitem(api.STATE, 'db_path', str(db))
    monkeypatch.setitem(api.STATE, 'pipeline', SimpleNamespace(_fullbody_search=engine))
    presentation = build_pose_presentation(report, output_scope='half', resolve_library=api._library_preview_asset)
    [reference] = [r for r in presentation.review_candidates if r.kind == 'mirror_reference']
    source = next(c for c in presentation.library_candidates
                  if c.candidate_id == reference.mirror_comparison.original_candidate_id)
    assert source.pose_id == 'b9_raise_hand_stand_mirror'
    assert reference.pose_id == 'b9_raise_hand_stand'
    assert source.asset.bvh_sha256 == reference.mirror_comparison.original_bvh_sha256
    assert len(presentation.library_candidates) == 4 and len(presentation.review_candidates) == 1
    assert presentation.status == 'ready' and not presentation.issues
    assert reference.rank_score is None and reference.score_kind is None
    assert source.render_camera.rotation == reference.render_camera.rotation
    assert not reference.render_camera.projection_matches_search
    for item in (source, reference):
        assert not item.export_allowed and not item.refine_allowed and not item.release_allowed
        response = api.get_pose_bvh(item.pose_id)
        assert hashlib.sha256(Path(response.path).read_bytes()).hexdigest() == item.asset.bvh_sha256
        assert any(r['status'] == 'clear_screen' and r['source_bvh_sha256'] == item.asset.bvh_sha256
                   for r in receipts)
    # Optional local handoff evidence; never committed to GitHub.
    if os.getenv('MIRROR_EVIDENCE_DIR'):
        evidence = Path(os.environ['MIRROR_EVIDENCE_DIR'])
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence/'presentation.json').write_text(presentation.model_dump_json(indent=2))
        (evidence/'skin-receipts.json').write_text(json.dumps(receipts, indent=2))
