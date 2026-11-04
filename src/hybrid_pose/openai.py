"""OpenAI transport for the unchanged skeleton repair prompt/validation contract."""
import json

from .gemini import GeminiReviewer, GeminiStageError
from .contracts import EVALUATION_SCHEMA, REPAIR_SCHEMA, SINGLE_SCHEMA
from ..vlm.openai_transport import OpenAIContentClient


class OpenAIReviewer(GeminiReviewer):
    def __init__(self, api_key, model, timeout=30, thinking_level="low"):
        super().__init__(api_key, model, timeout, thinking_level)
        if model.startswith("gemini"):
            raise ValueError("explicit OpenAI repair model required")
        self.client = OpenAIContentClient(api_key=api_key, timeout_ms=timeout * 1000)
        self.metadata.update(provider="openai", thinking_level=None)

    def request(self, stage, original_png, overlay_png, context, evaluation=None):
        if stage not in self.prompts or (stage == "repair") != (evaluation is not None):
            raise ValueError("Invalid stage/evaluation combination")
        schema = {"evaluation": EVALUATION_SCHEMA, "repair": REPAIR_SCHEMA,
                  "single_review_repair": SINGLE_SCHEMA}[stage]
        contents = [self.prompts[stage], "IMAGE 1: original rough",
                    {"inline_data": {"mime_type": "image/png", "data": original_png}},
                    "IMAGE 2: RTM overlay, not GT",
                    {"inline_data": {"mime_type": "image/png", "data": overlay_png}},
                    "RTM_CONTEXT_JSON\n" + json.dumps(context, ensure_ascii=False, allow_nan=False)]
        if evaluation is not None:
            contents.append("EVALUATION_JSON\n" + json.dumps(evaluation, ensure_ascii=False, allow_nan=False))
        try:
            response = self.client.generate_content(model=self.model, contents=contents,
                config={"response_json_schema": schema})
        except Exception as error:
            # SkeletonRepairer treats this safe error as unavailable evidence.
            raise GeminiStageError("OpenAI repair unavailable: " + type(error).__name__) from None
        return json.loads(response.text), {"provider": "openai", "model": response.model_version,
                                          "response_id": response.response_id}
