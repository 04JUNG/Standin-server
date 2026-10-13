"""Text-only PR must not introduce dynamic routing or change /analyze schema."""
import sys
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from api import app as api
from api.models import CutResultOut
from src.semantic_search import SemanticSearchService, SemanticSettings, ErrorCode
from tests.test_semantic_search import _write_library


def test_text_scope_has_no_rough_routes_or_analyze_extension():
    paths={r.path for r in api.app.routes}
    assert '/semantic-search' in paths
    assert '/rough-semantic-search' not in paths
    assert 'semantic_route' not in CutResultOut.model_fields
    assert not hasattr(api.CFG,'semantic_route_mode')


def test_index_load_failure_is_not_no_result():
    old=api.STATE.get('semantic')
    try:
        api.STATE['semantic']=SemanticSearchService(None,SemanticSettings())
        response=TestClient(api.app).post('/semantic-search',json={'query':'야구공을 던지는 자세'})
        assert response.status_code==503
        assert response.json()['status']=='error'
        assert response.json()['error']['code']==ErrorCode.INDEX_NOT_LOADED
    finally:
        if old is None: api.STATE.pop('semantic',None)
        else: api.STATE['semantic']=old


def test_review_preferences_are_opt_in_during_service_initialization():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _write_library(tmp)
        cfg = copy.copy(api.CFG)
        cfg.data_dir = tmp
        cfg.semantic_enabled = True
        cfg.semantic_dense_enabled = False
        cfg.semantic_verify_db_sha256 = True
        cfg.semantic_tags_path = paths['tags_path']
        cfg.semantic_taxonomy_path = paths['taxonomy_path']
        cfg.semantic_lexicon_path = paths['lexicon_path']
        cfg.semantic_review_preferences_path = ''
        service = SemanticSearchService.from_config(cfg, db_path=paths['db_path'])
        assert service.ready
        assert not service.index.review_preferences
        preference_path = Path(tmp) / 'preferences.json'
        preference_path.write_text(json.dumps({'schema_version': 1, 'entries': [
            {'query': '공 던지기', 'preferred_pose_id': 'pitch_01'}]}), encoding='utf-8')
        cfg.semantic_review_preferences_path = str(preference_path)
        service = SemanticSearchService.from_config(cfg, db_path=paths['db_path'])
        assert service.ready
        assert service.index.review_preferences['공 던지기']['preferred_pose_id'] == 'pitch_01'
        assert service.index.identity['review_preferences_sha256']


def test_optional_semantic_readiness_does_not_break_existing_health():
    previous=dict(api.STATE); required=api.CFG.semantic_required
    try:
        api.STATE.clear();api.STATE.update(pipeline=SimpleNamespace(),pose_count=1,
                                          semantic=SemanticSearchService(None,SemanticSettings()))
        api.CFG.semantic_required=False
        client=TestClient(api.app)
        assert client.get('/healthz').status_code==200
        api.CFG.semantic_required=True
        assert client.get('/healthz').status_code==503
    finally:
        api.STATE.clear();api.STATE.update(previous);api.CFG.semantic_required=required


if __name__=='__main__':
    funcs=[v for k,v in sorted(globals().items()) if k.startswith('test_')]
    for f in funcs:f();print('PASS',f.__name__)
    print(f'{len(funcs)}/{len(funcs)} passed')
