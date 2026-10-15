import hashlib
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from api.natural_search_app import app
from api.natural_search import get_runtime,Runtime,ParsedParts,fail
from api.natural_search_models import NaturalSearchResponse
from tests.test_natural_parts import minimal_bvh,SameSourceIndex


class Parser:
    def parse(self,text):return ParsedParts(upper={'description':text}),dict(provider='test',calls=0)


class Index(SameSourceIndex):
    identity=dict(library_version='fixture-not-production',source_db_sha256='d'*64,tags_sha256='e'*64,total=1,eligible=1)
    records=[]
    def search(self,*args,**kwargs):
        result=super().search(*args,**kwargs)
        for h in result['hits']:h.update(family='same',lateral=None)
        return result


@pytest.fixture
def client(tmp_path):
    bvh=tmp_path/'source.bvh';minimal_bvh(bvh)
    index=Index(bvh);index.records=[dict(pose_id='same',bvh_sha256=hashlib.sha256(bvh.read_bytes()).hexdigest())]
    runtime=Runtime(index,Parser());app.dependency_overrides[get_runtime]=lambda:runtime
    with TestClient(app) as c:yield c,runtime
    app.dependency_overrides.clear()


def test_parts_actual_composer_serializes_bytes_and_defaults(client):
    c,r=client;response=c.post('/natural-search/parts',json={'upper':{'description':'오른팔을 앞으로 굽힘'},'output_scope':'upper_only'})
    assert response.status_code==200,response.text
    result=NaturalSearchResponse.model_validate(response.json());assert result.parser['calls']==0 and not result.generation_executed
    assert len(result.candidates)==1 # No duplicate padding.
    h=result.candidates[0];assert hashlib.sha256(h.bvh.encode()).hexdigest()==h.bvh_sha256
    assert h.inferred_parts==['lower'] and not h.release_allowed
    assert result.lower.inferred_default and result.export_intent['preserve_arms_hands']
    assert not result.export_intent['actual_mesh_cropped']
    assert result.upper.hits[0].bvh_url=='/natural-search/library/same/bvh'


def test_text_shares_search_without_rough(client):
    response=client[0].post('/natural-search/text',json={'text':'오른팔을 앞으로 굽힘','top_k':1})
    assert response.status_code==200,response.text
    data=response.json();assert data['input_source']=='text' and data['parsed_parts']['upper']
    assert not data['skeleton_used_for_ranking'] and data['export_intent']['requested_scope']=='full'


@pytest.mark.parametrize('path,payload',[
    ('parts',{}),('parts',{'upper':{'description':' '}}),
    ('parts',{'upper':{'description':'서기','group':'standing'}}),
    ('parts',{'upper':{'description':'팔 내림'},'top_k':0}),('parts',{'upper':{'description':'팔 내림'},'top_k':6}),
    ('parts',{'upper':{'description':'팔 내림'},'top_k':True}),('parts',{'upper':{'description':'팔 내림','angle':90}}),
    ('text',{'text':' '}),('text',{'text':'a'*2001}),('text',{'text':'팔 내림','generate':True}),
    ('text',{'text':'팔 내림','output_scope':'lower_only'}),
])
def test_invalid_inputs(client,path,payload):
    assert client[0].post('/natural-search/'+path,json=payload).status_code==422


def test_unrelated_text_abstains(client):
    c,r=client;r.parser=SimpleNamespace(parse=lambda _: (ParsedParts(),dict(provider='test',calls=0)))
    data=c.post('/natural-search/text',json={'text':'안녕하세요'}).json()
    assert data['status']=='not_run' and data['candidates']==[]


def test_busy_and_parser_errors_are_not_fake_candidates(client):
    c,r=client;r.lock.acquire()
    try:
        response=c.post('/natural-search/parts',json={'upper':{'description':'팔 내림'}})
        assert response.status_code==429 and response.headers['retry-after']=='2'
    finally:r.lock.release()
    def bad(_):fail(502,'invalid_text_parser_response',True)
    r.parser=SimpleNamespace(parse=bad);response=c.post('/natural-search/text',json={'text':'팔 내림'})
    assert response.status_code==502 and 'candidates' not in response.json() and not r.lock.locked()


def test_disabled_and_assets_503_not_mock(monkeypatch):
    app.dependency_overrides.clear();monkeypatch.setenv('NATURAL_SEARCH_ENABLED','0')
    with TestClient(app) as c:
        response=c.post('/natural-search/parts',json={'upper':{'description':'팔 내림'}})
        assert response.status_code==503 and response.json()['detail']['code']=='natural_search_disabled'
        monkeypatch.setenv('NATURAL_SEARCH_ENABLED','1');monkeypatch.delenv('NATURAL_SEARCH_BUNDLE_DIR',raising=False)
        assert c.get('/natural-search/health').json()['detail']['code']=='natural_search_assets_unavailable'


def test_source_identity_and_unknown_id(client):
    c,r=client;response=c.get('/natural-search/library/same/bvh');assert response.status_code==200
    assert hashlib.sha256(response.content).hexdigest()==response.headers['x-source-sha256']
    assert c.get('/natural-search/library/generated_fake/bvh').status_code==404
    r.index.path.write_text('tampered');assert c.get('/natural-search/library/same/bvh').status_code==503


def test_openapi_is_typed(client):
    spec=client[0].get('/openapi.json').json();assert '/natural-search/text' in spec['paths']
    assert 'arms_front_bent' in spec['components']['schemas']['UpperPartQuery']['properties']['group']['enum']
    assert spec['components']['schemas']['PreviewCandidate']['properties']['release_allowed']['const'] is False
    assert spec['components']['schemas']['TextSearchRequest']['additionalProperties'] is False


def test_real_parser_configuration_has_one_attempt_and_strict_json(monkeypatch):
    from google import genai
    from api.natural_search import GeminiTextParser
    seen={}
    class Client:
        def __init__(self,**kwargs):seen.update(kwargs);self.models=self
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def generate_content(self,**kwargs):
            seen.update(kwargs)
            return SimpleNamespace(text='{"upper":{"description":"오른팔을 앞으로 굽힘"},"lower":null}')
    monkeypatch.setenv('GEMINI_API_KEY','test-only-not-a-real-key');monkeypatch.setattr(genai,'Client',Client)
    parts,trace=GeminiTextParser().parse('오른팔을 앞으로 굽힘')
    assert parts.upper.description=='오른팔을 앞으로 굽힘' and parts.lower is None and trace['calls']==1
    assert seen['http_options'].timeout==15000 and seen['http_options'].retry_options.attempts==1
    assert seen['config'].temperature==0 and seen['config'].response_mime_type=='application/json'


def test_missing_parser_key_and_invalid_provider_json_fail_closed(monkeypatch):
    from google import genai
    from fastapi import HTTPException
    from api.natural_search import GeminiTextParser
    monkeypatch.delenv('GEMINI_API_KEY',raising=False)
    with pytest.raises(HTTPException) as e:GeminiTextParser().parse('팔 내림')
    assert e.value.status_code==503
    class Client:
        def __init__(self,**kwargs):self.models=self
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def generate_content(self,**kwargs):return SimpleNamespace(text='{"upper":{"description":"팔","angle":90}}')
    monkeypatch.setenv('GEMINI_API_KEY','test-only-not-a-real-key');monkeypatch.setattr(genai,'Client',Client)
    with pytest.raises(HTTPException) as e:GeminiTextParser().parse('팔 내림')
    assert e.value.status_code==502 and e.value.detail['code']=='invalid_text_parser_response'


def test_committed_schema_matches_current_models():
    import json
    from pathlib import Path
    saved=json.loads((Path(__file__).resolve().parents[1]/'docs/openapi/natural-search-v1.json').read_text())
    assert saved==app.openapi()
