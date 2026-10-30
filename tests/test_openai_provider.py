"""Serving migration contracts using the real SDK over an offline HTTP transport."""
import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image
from google.genai import types
from openai import OpenAI

from src.config import CFG
from src.vlm.client import OpenAIVLMClient, VLMUnavailable
from src.vlm.openai_transport import OpenAIContentClient


def response(payload, status="completed"):
    return {"id": "resp_offline", "object": "response", "created_at": 1,
            "model": "offline-model", "status": status,
            "output": [{"id": "msg_offline", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": json.dumps(payload), "annotations": []}]}],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}


def transport(handler):
    sdk = OpenAI(api_key="offline-placeholder", max_retries=0,
                 http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return OpenAIContentClient(client=sdk)


def test_responses_images_schema_timeout_and_actual_usage(monkeypatch):
    captured = []
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "low")

    def handler(request):
        assert request.url.path == "/v1/responses"
        captured.append((json.loads(request.content), request.extensions["timeout"]))
        return httpx.Response(200, json=response({"people": []}))

    client = transport(handler)
    schema = {"type": "object", "properties": {"people": {"type": "array", "items": {"type": "string"}}},
              "required": ["people"], "additionalProperties": False}
    original = copy.deepcopy(schema)
    result = client.generate_content(model="offline-model",
        contents=["Return JSON", types.Part.from_bytes(data=b"png-data", mime_type="image/png")],
        config=types.GenerateContentConfig(response_json_schema=schema, temperature=0,
            http_options=types.HttpOptions(timeout=1234), max_output_tokens=1000))
    payload, timeout = captured[0]
    assert payload["store"] is False and "temperature" not in payload
    assert payload["text"]["format"]["strict"] is True
    assert payload["reasoning"] == {"effort": "low"}
    assert payload["max_output_tokens"] == 1000
    assert payload["input"][0]["content"][1]["image_url"] == "data:image/png;base64,cG5nLWRhdGE="
    assert timeout["read"] == 1.234 and schema == original
    assert result.response_id == "resp_offline"
    assert result.usage_metadata.total_token_count == 15


def test_dynamic_schema_keeps_optional_fields_and_local_validation():
    captured = []
    client = transport(lambda req: (captured.append(json.loads(req.content)) or httpx.Response(200, json=response({}))))
    schema = {"type": "object", "properties": {"hands": {"type": "object", "additionalProperties": {"type": "string"}}}}
    client.generate_content(model="offline-model", contents="Return JSON", config={"response_json_schema": schema})
    assert captured[0]["text"]["format"] == {"type": "json_object"}
    assert json.dumps(schema) in captured[0]["instructions"]
    assert "reasoning" not in captured[0]


def test_sdk_has_no_automatic_retry_or_gemini_model_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_REASONING_EFFORT", raising=False)
    calls = []
    client = transport(lambda req: (calls.append(req) or httpx.Response(429, json={"error": {
        "message": "private billing details", "type": "insufficient_quota", "code": "insufficient_quota"}})))
    from openai import RateLimitError
    with pytest.raises(RateLimitError):
        client.generate_content(model="offline-model", contents="JSON")
    assert len(calls) == 1
    with pytest.raises(ValueError, match="explicit OpenAI model"):
        client.generate_content(model="gemini-test", contents="JSON")
    assert len(calls) == 1


@pytest.mark.parametrize("status,code,reason", [
    (401, "invalid_api_key", "authentication_failed"),
    (403, "permission_denied", "permission_denied"),
    (429, "insufficient_quota", "billing_exhausted"),
])
def test_serving_auth_and_billing_fail_once_as_safe_unavailable(monkeypatch, status, code, reason):
    monkeypatch.setattr(CFG, "body_matching_mode", "off")
    calls = []
    client = transport(lambda req: (calls.append(req) or httpx.Response(status, json={"error": {
        "message": "private upstream message", "code": code, "type": code}})))
    vlm = OpenAIVLMClient(model="offline-model", client=client)
    with pytest.raises(VLMUnavailable) as failure:
        vlm.analyze(Image.new("RGB", (8, 8)), 8, 8)
    assert failure.value.reason == reason and failure.value.retryable is False
    assert failure.value.attempts == len(calls) == 1
    assert "private" not in str(failure.value)


def test_description_and_tagging_use_openai_without_gemini(monkeypatch):
    from google import genai
    monkeypatch.setattr(genai, "Client", lambda **kw: pytest.fail("Gemini network client was created"))
    captured = []
    answers = [
        {"people": [{"visible": True, "body_details": "팔 내림", "search_text": "서 있는 사람", "certainty": "high"}]},
        {"people": []},
    ]
    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=response(answers.pop(0)))
    vlm = OpenAIVLMClient(model="offline-model", client=transport(handler))
    image = Image.new("RGB", (8, 8))
    result = vlm.describe_people(image, [[0, 0, 1, 1]])
    assert result.status == "ok" and result.provider == "openai"
    assert vlm.tag_people(image, [[0, 0, 1, 1]]) == {"people": []}
    assert len(captured) == 2 and all(item["model"] == "offline-model" for item in captured)


def test_missing_key_or_explicit_model_never_reaches_network(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(CFG, "openai_model", "")
    with pytest.raises(ValueError, match="OPENAI_MODEL"):
        OpenAIVLMClient()
    from openai import OpenAIError
    with pytest.raises(OpenAIError):
        OpenAIVLMClient(model="offline-model")


def test_previous_openai_data_url_input_remains_supported():
    part = OpenAIVLMClient._to_part("data:image/png;base64,cG5nLWRhdGE=")
    assert part.inline_data.mime_type == "image/png"
    assert part.inline_data.data == b"png-data"


def test_natural_planner_uses_selected_provider_and_original_prompt(monkeypatch):
    from src.natural_pose_worker import GeminiSpecPlanner
    from src.vlm import openai_transport
    monkeypatch.setattr(CFG, "vlm_provider", "openai")
    monkeypatch.setattr(CFG, "openai_model", "offline-model")
    monkeypatch.delenv("POSE_GENERATION_PROVIDER", raising=False)
    monkeypatch.delenv("POSE_GENERATION_MODEL", raising=False)
    calls = []
    client = transport(lambda req: (calls.append(json.loads(req.content)) or httpx.Response(200, json=response({"base": "test"}))))
    monkeypatch.setattr(openai_transport, "OpenAIContentClient", lambda **kw: client)
    planner = GeminiSpecPlanner({"vocabulary": {}}, "offline-model")
    assert planner("팔을 내린다", None, [], 3) == {"base": "test"}
    assert "안전한 3D IK 목표" in calls[0]["instructions"]
    assert calls[0]["model"] == "offline-model"


def test_body_adapters_and_repair_use_openai_without_google_credentials(monkeypatch):
    from src.vlm import openai_transport
    from src.experimental.body_matching.observation import OpenAIBodyAttributeClient
    from src.experimental.body_matching.coarse import OpenAICoarseClient
    from src.skeleton_repair import build_repairer
    client = SimpleNamespace(models=None)
    monkeypatch.setattr(openai_transport, "OpenAIContentClient", lambda **kw: client)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder")
    monkeypatch.setattr(CFG, "vlm_provider", "openai")
    monkeypatch.setattr(CFG, "openai_model", "offline-model")
    assert OpenAIBodyAttributeClient("offline-model", 12).client is client
    assert OpenAICoarseClient("offline-model", 12).client is client
    reviewer = build_repairer().reviewer
    assert reviewer.metadata["provider"] == "openai" and reviewer.model == "offline-model"


@pytest.mark.parametrize("payload,status", [([], "completed"), ({}, "incomplete")])
def test_partial_or_nonobject_response_rejected(payload, status):
    client = transport(lambda req: httpx.Response(200, json=response(payload, status)))
    with pytest.raises(ValueError):
        client.generate_content(model="offline-model", contents="JSON")
