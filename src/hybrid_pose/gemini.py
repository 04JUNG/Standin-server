"""Single-attempt Gemini transport. No implicit retries or model fallback."""

import base64
import copy
import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from pathlib import Path

from .contracts import EVALUATION_SCHEMA, REPAIR_SCHEMA

PROMPTS = Path(__file__).with_name("prompts")
ROLES = {"evaluation": "평가용_VLM_v3.2.md", "repair": "수정용_VLM_v3.2.md"}


class GeminiStageError(RuntimeError):
    """Provider failure with a safe message (no key, payload, or response body)."""


def provider_schema(schema):
    schema = copy.deepcopy(schema)

    def strip(value):
        if isinstance(value, dict):
            for name in (
                "additionalProperties",
                "minimum",
                "maximum",
                "minItems",
                "maxItems",
            ):
                value.pop(name, None)
            for child in value.values():
                strip(child)
        elif isinstance(value, list):
            for child in value:
                strip(child)

    strip(schema)
    return schema


class GeminiReviewer:
    def __init__(self, api_key, model, timeout=180):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", model):
            raise ValueError("Explicit Gemini model ID is required")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be positive and finite")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        # Snapshot all prompt text at construction; never reload between stages.
        common = (PROMPTS / "공통_규칙_v3.2.md").read_text(encoding="utf-8")
        self.prompts = {
            stage: common + "\n\n" + (PROMPTS / name).read_text(encoding="utf-8")
            for stage, name in ROLES.items()
        }
        self.metadata = {
            "provider": "gemini",
            "model": model,
            "timeout_seconds": timeout,
            "prompt_sha256": {
                stage: hashlib.sha256(text.encode()).hexdigest()
                for stage, text in self.prompts.items()
            },
            "max_attempts_per_stage": 1,
        }

    def request(self, stage, original_png, overlay_png, context, evaluation=None):
        if stage not in ROLES or (stage == "repair") != (evaluation is not None):
            raise ValueError("Invalid stage/evaluation combination")
        parts = [{"text": self.prompts[stage]}]
        for label, data in (
            ("IMAGE 1: original rough", original_png),
            ("IMAGE 2: RTM overlay, not GT", overlay_png),
        ):
            parts.extend(
                [
                    {"text": label},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": base64.b64encode(data).decode("ascii"),
                        }
                    },
                ]
            )
        parts.append(
            {
                "text": "RTM_CONTEXT_JSON\n"
                + json.dumps(context, ensure_ascii=False, allow_nan=False)
            }
        )
        if evaluation is not None:
            parts.append(
                {
                    "text": "EVALUATION_JSON\n"
                    + json.dumps(evaluation, ensure_ascii=False, allow_nan=False)
                }
            )
        schema = EVALUATION_SCHEMA if stage == "evaluation" else REPAIR_SCHEMA
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseJsonSchema": provider_schema(schema),
            },
        }
        request = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.load(response)
        except urllib.error.HTTPError as error:
            raise GeminiStageError(
                f"Gemini HTTP {error.code}; stage not retried"
            ) from None
        except (OSError, ValueError) as error:
            raise GeminiStageError(
                f"Gemini transport/response failure ({type(error).__name__}); stage not retried"
            ) from None
        try:
            candidate = raw["candidates"][0]
            if candidate.get("finishReason") != "STOP":
                raise ValueError("Incomplete/blocked response")
            text = "".join(
                p["text"]
                for p in candidate["content"]["parts"]
                if isinstance(p.get("text"), str) and not p.get("thought")
            )
            value = json.loads(text)
        except (KeyError, IndexError, TypeError, ValueError):
            raise GeminiStageError(
                "Gemini response incomplete or invalid JSON; stage not retried"
            ) from None
        return value, raw
