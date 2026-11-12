"""Fail before ECS deployment if the approved staging body release was replaced."""
import argparse
import json
from pathlib import Path
try:
    from scripts.prepare_staging_anatomy import RELEASE, INITIALIZER
except ModuleNotFoundError:
    from prepare_staging_anatomy import RELEASE, INITIALIZER


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
        "BODY_ASSETS_MANIFEST_KEY": RELEASE['prefix'] + '/manifest.json',
        "BODY_ASSETS_MANIFEST_SHA256": RELEASE['manifestSha256'],
    }
    for key, value in expected.items():
        if env(init).get(key) != value:
            raise ValueError(f"Staging anatomy release changed: {key}")
    if init.get('command') != ['python', '-c', INITIALIZER]:
        raise ValueError('Staging body initializer changed')
    for cid in RELEASE['characterIds']:
        key = 'BODY_' + cid.upper().replace('-', '_') + '_URI'
        if env(main).get(key) != '/app/body-staging/fbx/' + cid + '.fbx':
            raise ValueError('Staging preset changed: ' + cid)
    for key, value in {
        "CONVERTER_SKIN_ENABLED": "true",
        "CONVERTER_OUTLINE_ENABLED": "true",
        "CONVERTER_SKIN_ASSET_ROOT": "/app/body-staging/skin",
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
