"""Responses transport for the existing bounded JSON/image call contract.

The application owns retries, deadlines and validation. This adapter translates
only transport fields; it never changes models or falls back to another provider.
Google Part/config objects remain accepted so existing prompts and parsers can
be shared while Gemini deployments keep their original transport.
"""
from __future__ import annotations

import base64
import copy
import json
import math
import os
from types import SimpleNamespace


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _parts(contents):
    if isinstance(contents, str):
        return [{"type": "input_text", "text": contents}]
    result = []
    for item in contents:
        if isinstance(item, str):
            result.append({"type": "input_text", "text": item})
        elif _get(item, "parts") is not None:
            result.extend(_parts(_get(item, "parts")))
        elif _get(item, "text") is not None:
            result.append({"type": "input_text", "text": _get(item, "text")})
        else:
            data = _get(item, "inline_data", _get(item, "inlineData"))
            mime = _get(data, "mime_type", _get(data, "mimeType"))
            raw = _get(data, "data")
            if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"} or not isinstance(raw, (str, bytes)):
                raise ValueError("unsupported OpenAI image part")
            encoded = base64.b64encode(raw).decode("ascii") if isinstance(raw, bytes) else raw
            result.append({"type": "input_image", "image_url": f"data:{mime};base64,{encoded}"})
    if not result:
        raise ValueError("empty OpenAI input")
    return result


def _strict_schema(schema):
    """Use strict decoding only for closed, fully required object contracts.

    Dynamic maps and optional fields retain JSON mode plus the exact schema in
    the instructions. Their existing application validators remain authoritative.
    Do not make optional fields required or invent nulls to satisfy a provider.
    """
    if not isinstance(schema, dict):
        return False
    if schema.get("type") == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is not False or set(schema.get("required", [])) != set(properties):
            return False
    return all(_strict_schema(v) for k, v in schema.items() if k in {"items", "additionalProperties"} and isinstance(v, dict)) and all(
        _strict_schema(v) for v in schema.get("properties", {}).values()
    ) and all(_strict_schema(v) for key in ("anyOf", "oneOf", "allOf") for v in schema.get(key, []))


class OpenAIContentClient:
    def __init__(self, *, api_key=None, timeout_ms=45000, client=None):
        if not math.isfinite(timeout_ms) or timeout_ms <= 0:
            raise ValueError("positive OpenAI timeout required")
        if client is None:
            from openai import OpenAI
            # Never nest SDK retries under the application's time/cost budget.
            client = OpenAI(api_key=api_key, timeout=timeout_ms / 1000, max_retries=0)
        self._client = client
        self.timeout_ms = timeout_ms
        self.models = self

    def generate_content(self, *, model, contents, config=None):
        if not isinstance(model, str) or not model.strip() or model.startswith("gemini"):
            raise ValueError("explicit OpenAI model required")
        options = _get(config, "http_options")
        timeout_ms = _get(options, "timeout") or self.timeout_ms
        if not math.isfinite(timeout_ms) or timeout_ms <= 0:
            raise ValueError("positive OpenAI timeout required")
        instructions = _get(config, "system_instruction") or "Return a JSON object only."
        schema = copy.deepcopy(_get(config, "response_json_schema"))
        if schema and _strict_schema(schema):
            fmt = {"type": "json_schema", "name": "standin_result", "strict": True, "schema": schema}
        else:
            fmt = {"type": "json_object"}
            if schema:
                instructions += "\nReturn JSON matching this schema:\n" + json.dumps(schema, ensure_ascii=False)
        request = dict(model=model, instructions=instructions,
                       input=[{"role": "user", "content": _parts(contents)}],
                       text={"format": fmt}, store=False,
                       timeout=timeout_ms / 1000)
        # Reasoning is an explicit OpenAI setting, never a translated Gemini
        # thinking budget. Leave unspecified models on their own defaults.
        effort = os.getenv("OPENAI_REASONING_EFFORT", "").strip()
        if effort:
            request["reasoning"] = {"effort": effort}
        maximum = _get(config, "max_output_tokens")
        if maximum:
            request["max_output_tokens"] = maximum
        response = self._client.responses.create(**request)
        if response.status != "completed":
            raise ValueError("incomplete OpenAI response")
        for item in response.output:
            for part in _get(item, "content", []) or []:
                if _get(part, "type") == "refusal":
                    raise ValueError("OpenAI response refused")
        text = response.output_text
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("OpenAI response must be a JSON object")
        usage = response.usage
        return SimpleNamespace(text=text, model_version=response.model,
            response_id=response.id, usage_metadata=SimpleNamespace(
                prompt_token_count=_get(usage, "input_tokens"),
                candidates_token_count=_get(usage, "output_tokens"),
                total_token_count=_get(usage, "total_tokens")))

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def content_client(provider, *, timeout_ms, api_key=None):
    if provider == "openai":
        return OpenAIContentClient(api_key=api_key, timeout_ms=timeout_ms)
    if provider != "gemini":
        raise ValueError("JSON provider must be gemini or openai")
    from google import genai
    from google.genai import types
    from .client import _gemini_attempt_options
    return genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"],
                        http_options=_gemini_attempt_options(types, timeout_ms))
