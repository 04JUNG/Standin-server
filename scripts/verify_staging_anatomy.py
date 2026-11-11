"""Fail before ECS deployment if the approved staging body release was replaced."""
import argparse
import json
from pathlib import Path


def verify(task):
    if not task.get("family", "").startswith("StandinStagingApp"):
        raise ValueError("Anatomy guard requires a staging task")
    containers = {c["name"]: c for c in task["containerDefinitions"]}
    main = containers["converter"]
    init = containers["body-assets-init"]
    env = lambda c: {e["name"]: e["value"] for e in c.get("environment", [])}
    expected = {
        "BODY_ASSETS_ROLE": "converter",
        "BODY_ASSETS_ROOT": "/app/body-staging",
        "BODY_ASSETS_MANIFEST_KEY": "body-models/releases/anatomy-20261111-bf9439491bef-converter/manifest.json",
        "BODY_ASSETS_MANIFEST_SHA256": "bf9439491befc529b1a7b1072bfa46b2fb91f2d3eef93ed36b52f847a0534c3c",
    }
    for key, value in expected.items():
        if env(init).get(key) != value:
            raise ValueError(f"Staging anatomy release changed: {key}")
    for key, value in {
        "CONVERTER_CHARACTER_REGISTRY": "/app/body-staging/characters.json",
        "CONVERTER_SHAPE_REGISTRY": "/app/body-staging/shape-bakes.json",
        "BODY_MALE_BASE_V0_8_0_URI": "/app/body-staging/fbx/male-base-v0-8-0.fbx",
        "BODY_FEMALE_BASE_V0_4_0_URI": "/app/body-staging/fbx/female-base-v0-4-0.fbx",
    }.items():
        if env(main).get(key) != value:
            raise ValueError(f"Staging anatomy model changed: {key}")
    if not any(d.get("containerName") == init["name"] and d.get("condition") == "SUCCESS"
               for d in main.get("dependsOn", [])):
        raise ValueError("Missing successful body initializer dependency")
    return task


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_definition", type=Path)
    args = parser.parse_args()
    verify(json.loads(args.task_definition.read_text(encoding="utf-8-sig")))
    print("Approved staging anatomy release verified")
