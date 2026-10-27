"""Fail closed before deploying into an existing dedicated staging service."""

import json
import re
import sys


def validate(task):
    containers = task["containerDefinitions"]
    if len(containers) != 1 or containers[0]["name"] != "scene-worker":
        raise ValueError("a dedicated scene-worker task is required")
    if task.get("runtimePlatform", {}).get("cpuArchitecture") != "X86_64":
        raise ValueError("scene worker requires X86_64")
    worker = containers[0]
    env = {v["name"]: v["value"] for v in worker.get("environment", [])}
    secrets = {v["name"]: v["valueFrom"] for v in worker.get("secrets", [])}
    if env.get("APP_ENV") != "staging" or env.get("SCENE_REFINEMENT_ENABLED") != "1":
        raise ValueError("explicit staging worker opt-in is required")
    if env.get("SCENE_VLM_PROVIDER") != "gemini" or not env.get("SCENE_VLM_MODEL"):
        raise ValueError("explicit Gemini model is required")
    if not re.fullmatch(r"[0-9a-f]{64}", env.get("SCENE_ASSET_MANIFEST_SHA256", "")):
        raise ValueError("approved asset manifest SHA is required")
    if not env.get("SCENE_ASSET_MANIFEST", "").startswith("/scene-assets/"):
        raise ValueError("mounted release manifest is required")
    for name in ("SCENE_REFINEMENT_TOKEN", "GEMINI_API_KEY"):
        if name in env or not secrets.get(name, "").startswith("arn:aws:"):
            raise ValueError("worker credentials must use ECS secrets")
    if worker.get("command") or worker.get("entryPoint"):
        raise ValueError("worker image command must not be overridden")
    if not any(
        m["containerPath"] == "/scene-assets" and m.get("readOnly") is True
        for m in worker.get("mountPoints", [])
    ):
        raise ValueError("read-only release mount is required")
    command = worker.get("healthCheck", {}).get("command", [])
    if not any("8002/readyz" in arg for arg in command):
        raise ValueError("ECS must check worker readiness")


if __name__ == "__main__":
    validate(json.load(open(sys.argv[1])))
    print(
        "Dedicated staging worker task configuration passed (assets are checked at runtime)."
    )
