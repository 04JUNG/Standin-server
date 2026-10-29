"""Deadline recovery keeps the real description/schema and hybrid contract."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from google import genai
from google.genai import types
from PIL import Image

from src.config import CFG
from src.vlm.client import GeminiVLMClient


PAYLOAD = {'people': [{'visible': True, 'certainty': 'high',
    'postures': ['sitting'], 'actions': ['fold_arms'],
    'body_details': '의자에 앉아 양팔을 가슴 앞에 교차해 팔짱을 낀다',
    'search_text': '의자에 앉아 팔짱을 낀 자세'}]}
IMAGE = Image.new('RGB', (16, 16))
BOXES = [(0, 0, 1, 1)]


class Clock:
    def __init__(self): self.now = 0.
    def __call__(self): return self.now
    def sleep(self, seconds): self.now += seconds


def setup(monkeypatch, outcomes):
    monkeypatch.setattr(CFG, 'routing_describe_timeout_ms', 15000)
    monkeypatch.setattr(CFG, 'routing_describe_budget_ms', 45000)
    clock, calls = Clock(), []
    outcomes = iter(outcomes)
    def generate(**kwargs):
        calls.append(kwargs)
        seconds, status = next(outcomes)
        clock.now += seconds
        if status != 200:
            raise genai.errors.ServerError(status, {'error': {
                'code': status, 'status': 'DEADLINE_EXCEEDED', 'message': 'private detail'}})
        return SimpleNamespace(text=json.dumps(PAYLOAD))
    client = GeminiVLMClient(client=SimpleNamespace(models=SimpleNamespace(generate_content=generate)),
        clock=clock, sleep=clock.sleep, jitter=lambda *_: .75)
    return client, clock, calls


def test_504_retry_extends_processing_time_without_changing_quality_inputs(monkeypatch):
    client, clock, calls = setup(monkeypatch, [(13, 504), (20, 200)])
    result = client.describe_people(IMAGE, BOXES)
    assert result.status == 'ok' and result.attempts == 2
    assert result.descriptions[0].actions == ['fold_arms']
    assert result.elapsed_ms == pytest.approx(33750)
    configs = [c['config'] for c in calls]
    assert [c.http_options.timeout for c in configs] == [15000, 30000]
    for c in configs:
        if 'retry_options' in types.HttpOptions.model_fields:
            assert c.http_options.retry_options.attempts == 1
    assert configs[0].system_instruction == configs[1].system_instruction
    assert configs[0].response_json_schema == configs[1].response_json_schema
    assert calls[0]['contents'][0] == calls[1]['contents'][0]
    assert calls[0]['contents'][1] == calls[1]['contents'][1]


def test_retry_timeout_is_capped_by_remaining_budget(monkeypatch):
    client, clock, calls = setup(monkeypatch, [(15, 504), (29, 504)])
    result = client.describe_people(IMAGE, BOXES)
    assert [c['config'].http_options.timeout for c in calls] == [15000, 29250]
    assert result.status == 'error' and not result.descriptions and result.attempts == 2
    assert result.error_reason == 'upstream_error'


def test_exhausted_budget_does_not_start_or_sleep_for_retry(monkeypatch):
    client, clock, calls = setup(monkeypatch, [(45, 504)])
    result = client.describe_people(IMAGE, BOXES)
    assert len(calls) == result.attempts == 1
    assert clock.now == 45 and result.status == 'error'


def test_short_budget_caps_first_attempt(monkeypatch):
    client, clock, calls = setup(monkeypatch, [(1, 200)])
    monkeypatch.setattr(CFG, 'routing_describe_budget_ms', 2000)
    assert client.describe_people(IMAGE, BOXES).status == 'ok'
    assert calls[0]['config'].http_options.timeout <= 2000


@pytest.mark.parametrize('hint,retry', [(1.5, True), (3., False)])
def test_provider_cooldown_is_not_shortened(monkeypatch, hint, retry):
    client, clock, calls = setup(monkeypatch, [])
    from src.search_routing import DescribeOutcome
    seen = []
    def once(*args):
        seen.append(args[-1])
        return DescribeOutcome('error', retryable=True, retry_after_seconds=hint)
    monkeypatch.setattr(client, '_describe_people_once', once)
    client.describe_people(IMAGE, BOXES)
    assert len(seen) == (2 if retry else 1)
    assert clock.now == (hint if retry else 0)


def test_recovered_description_reaches_semantic_search_and_fusion(monkeypatch):
    from src.search_routing import RoutingConfig, assess_semantic_evidence, decide_route
    from tests.test_fullbody_routing import engine, decision, evidence
    client, _, _ = setup(monkeypatch, [(13, 504), (20, 200)])
    recovered = client.describe_people(IMAGE, BOXES)
    sem = assess_semantic_evidence(recovered.descriptions[0], RoutingConfig())
    e = engine()
    queries = []
    search = e.semantic.search
    def semantic(query, top_k):
        queries.append(query)
        return search(query, top_k)
    e.semantic.search = semantic
    out = e.search(decide_route(decision().joint, sem), evidence())
    assert out['status'] == 'ok' and out['route'] == 'hybrid'
    assert e.semantic.calls == e.camera.calls == 1
    assert queries[0].actions == ['fold_arms']
    assert '팔짱' in queries[0].text
    assert set(out['selected'][0]['evidence']) == {'geometry', 'semantic'}
    assert e.generator.calls == 0


def test_final_failure_never_runs_semantic_or_generator(monkeypatch):
    from src.search_routing import RoutingConfig, assess_semantic_evidence, decide_route
    from tests.test_fullbody_routing import engine, decision, evidence
    client, _, _ = setup(monkeypatch, [(13, 504), (29, 504)])
    failed = client.describe_people(IMAGE, BOXES)
    sem = assess_semantic_evidence(None, RoutingConfig(), vlm_error=failed.error_reason)
    e = engine()
    out = e.search(decide_route(decision().joint, sem), evidence())
    assert out['status'] == 'partial_error' and out['route'] == 'geometric'
    assert e.semantic.calls == e.generator.calls == 0
    assert out['branches']['semantic']['status'] == 'error'


def test_real_sdk_504_then_success_uses_two_requests_only(monkeypatch):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((dict(self.headers), json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            status = 504 if len(requests) == 1 else 200
            payload = {'error': {'code': 504, 'status': 'DEADLINE_EXCEEDED', 'message': 'private detail'}} if status == 504 else {
                'candidates': [{'content': {'role': 'model', 'parts': [{'text': json.dumps(PAYLOAD)}]}, 'finishReason': 'STOP'}]}
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_args): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(CFG, 'routing_describe_timeout_ms', 15000)
    monkeypatch.setattr(CFG, 'routing_describe_budget_ms', 45000)
    sdk = genai.Client(api_key='local-fault-test', http_options=types.HttpOptions(
        base_url=f'http://127.0.0.1:{server.server_port}', timeout=45000))
    try:
        client = GeminiVLMClient(client=sdk, sleep=lambda _: None, jitter=lambda *_: .5)
        result = client.describe_people(IMAGE, BOXES)
    finally:
        sdk.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
    assert result.status == 'ok' and result.attempts == len(requests) == 2
    assert requests[0][1]['generationConfig'] == requests[1][1]['generationConfig']
    assert [next(v for k,v in headers.items() if k.lower() == 'x-server-timeout') for headers,_ in requests] == ['15','30']


def test_shared_analysis_elapsed_time_shortens_retry(monkeypatch):
    from src.analysis_budget import analysis_budget
    client, clock, calls = setup(monkeypatch, [(5, 504), (2, 200)])
    with analysis_budget(120, clock=clock):
        clock.now = 90  # Initial analysis/extraction already spent 90 seconds.
        result = client.describe_people(IMAGE, BOXES)
    assert result.status == 'ok'
    assert [c['config'].http_options.timeout for c in calls] == [15000, 9250]
    assert clock.now < 105


def test_reserve_exhaustion_makes_no_provider_call(monkeypatch):
    from src.analysis_budget import analysis_budget
    client, clock, calls = setup(monkeypatch, [])
    with analysis_budget(120, clock=clock):
        clock.now = 106
        result = client.describe_people(IMAGE, BOXES)
    assert result.status == 'error' and result.attempts == 0 and not calls
    assert not result.descriptions


def test_cancelled_request_does_not_retry(monkeypatch):
    from src.hand_skin_parallel import request_cancelled
    client, clock, calls = setup(monkeypatch, [(5, 504)])
    event = threading.Event()
    def cancel(_):
        event.set()
    client._sleep = cancel
    token = request_cancelled.set(event)
    try:
        result = client.describe_people(IMAGE, BOXES)
        assert result.attempts == 1 and len(calls) == 1
    finally:
        request_cancelled.reset(token)


def test_side_review_shares_description_budget_and_abstains(monkeypatch):
    from tests.test_action_relations import relation
    client, clock, calls = setup(monkeypatch, [])
    payload = {'people': [{**PAYLOAD['people'][0], 'pose_relations': [relation()]}]}
    def generate(**kw):
        calls.append(kw)
        clock.now += 45
        return SimpleNamespace(text=json.dumps(payload))
    client._client.models.generate_content = generate
    result = client.describe_people(IMAGE, BOXES)
    assert result.status == 'ok' and len(calls) == 1
    assert result.descriptions[0].pose_relations[0]['side'] == 'unknown'
    assert result.descriptions[0].laterality_review['status'] == 'unavailable_abstain'
