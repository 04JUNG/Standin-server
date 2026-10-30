"""Apply explicit environment-scoped VLM settings without reading any secrets.

Deploy uses the serving profile to replace legacy Gemini settings with OpenAI.
Explicit environment overrides remain available for rollback. Without the profile,
empty overrides preserve the current task (including standalone evaluation tasks).
The helper never copies keys/ARNs from another environment or changes evaluation.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

FIELDS = (
    "VLM_PROVIDER", "OPENAI_MODEL", "OPENAI_REASONING_EFFORT",
    "OPENAI_REQUEST_TIMEOUT_MS", "OPENAI_MAX_ATTEMPTS", "OPENAI_TOTAL_BUDGET_SECONDS",
    "BODY_VLM_PROVIDER", "BODY_VLM_MODEL", "BODY_COARSE_FAMILY_REVIEW_MODEL",
    "POSE_GENERATION_PROVIDER", "POSE_GENERATION_MODEL", "SKELETON_REPAIR_MODEL",
)


SERVING_DEFAULTS = {
    "VLM_PROVIDER": "openai",
    "OPENAI_MODEL": "gpt-6-luna",
    "OPENAI_REASONING_EFFORT": "low",
    "BODY_VLM_PROVIDER": "openai",
    "BODY_VLM_MODEL": "gpt-6-luna",
    "BODY_COARSE_FAMILY_REVIEW_MODEL": "gpt-6-luna",
    "POSE_GENERATION_PROVIDER": "openai",
    "POSE_GENERATION_MODEL": "gpt-6-luna",
    "SKELETON_REPAIR_MODEL": "gpt-6-luna",
}


def serving_overrides(overrides):
    """Choose the serving provider even when GitHub variables are unset."""
    return {**SERVING_DEFAULTS, **{
        name: overrides[name] for name in FIELDS if overrides.get(name, "").strip()
    }}


def prepare(task, overrides):
    result = copy.deepcopy(task)
    [container] = [c for c in result["containerDefinitions"] if c["name"] == "inference"]
    rows = container.get("environment", [])
    env = {row["name"]: row["value"] for row in rows}
    if len(env) != len(rows):
        raise ValueError("duplicate deployment environment names")
    for name in FIELDS:
        value = overrides.get(name, "").strip()
        if value:
            if "\n" in value or "\r" in value:
                raise ValueError("invalid VLM deployment setting: " + name)
            env[name] = value
    if not any(overrides.get(name, "").strip() for name in FIELDS):
        return result
    secrets = {row["name"] for row in container.get("secrets", [])}

    def check(provider, model):
        if provider not in {"gemini", "openai"}:
            raise ValueError("deployment requires a real VLM provider")
        key = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
        if key in env:
            raise ValueError(key + " must not be stored in plaintext deployment environment")
        if key not in secrets:
            raise ValueError(key + " must be injected through the environment's ECS secret reference")
        if not model or (provider == "openai") == model.startswith("gemini"):
            raise ValueError("explicit model matching " + provider + " required")

    provider = env.get("VLM_PROVIDER", "")
    model = env.get("OPENAI_MODEL" if provider == "openai" else "GEMINI_MODEL", "")
    check(provider, model)
    if env.get("BODY_MATCHING_MODE", "off") != "off" and env.get("BODY_CLASSIFICATION_STRATEGY") != "integrated":
        body_provider = env.get("BODY_VLM_PROVIDER", "mock")
        check(body_provider, env.get("BODY_VLM_MODEL", ""))
        family = env.get("BODY_COARSE_FAMILY_REVIEW_MODEL")
        if family:
            check(body_provider, family)
    if env.get("POSE_GENERATION_ENABLED", "1" if env.get("ROUTING_FULLBODY_ENABLED") == "1" else "0") == "1":
        generation_provider = env.get("POSE_GENERATION_PROVIDER", provider)
        generation_model = env.get("POSE_GENERATION_MODEL") or (model if generation_provider == "openai" else "gemini-3.8-flash")
        check(generation_provider, generation_model)
    repair_model = env.get("SKELETON_REPAIR_MODEL")
    if repair_model and env.get("SKELETON_REPAIR", "auto") != "off":
        check(provider, repair_model)
    if env.get("OPENAI_REASONING_EFFORT", "") not in {"", "none", "minimal", "low", "medium", "high", "xhigh", "max"}:
        raise ValueError("invalid OPENAI_REASONING_EFFORT")
    container["environment"] = [{"name": name, "value": value} for name, value in env.items()]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    parser.add_argument("--serving", action="store_true", help="apply OpenAI serving defaults before explicit overrides")
    args = parser.parse_args()
    overrides = serving_overrides(os.environ) if args.serving else os.environ
    result = prepare(json.loads(args.task.read_text()), overrides)
    args.task.write_text(json.dumps(result), encoding="utf-8")
    print("VLM deployment settings validated; secret values were not read")


if __name__ == "__main__":
    main()
