import copy
import pytest
from scripts.verify_staging_anatomy import verify


def fixture():
    def env(values):
        return [{"name": k, "value": v} for k, v in values.items()]
    return {"family": "StandinStagingApp-ConverterTask", "containerDefinitions": [
        {"name": "converter", "dependsOn": [{"containerName": "body-assets-init", "condition": "SUCCESS"}], "environment": env({
            "CONVERTER_CHARACTER_REGISTRY": "/app/body-staging/characters.json",
            "CONVERTER_SHAPE_REGISTRY": "/app/body-staging/shape-bakes.json",
            "BODY_MALE_BASE_V0_8_0_URI": "/app/body-staging/fbx/male-base-v0-8-0.fbx",
            "BODY_FEMALE_BASE_V0_4_0_URI": "/app/body-staging/fbx/female-base-v0-4-0.fbx",
        })},
        {"name": "body-assets-init", "environment": env({
            "BODY_ASSETS_ROLE": "converter", "BODY_ASSETS_ROOT": "/app/body-staging",
            "BODY_ASSETS_MANIFEST_KEY": "body-models/releases/anatomy-20261111-bf9439491bef-converter/manifest.json",
            "BODY_ASSETS_MANIFEST_SHA256": "bf9439491befc529b1a7b1072bfa46b2fb91f2d3eef93ed36b52f847a0534c3c",
        })},
    ]}


def test_approved_release_is_not_mutated():
    task = fixture()
    before = copy.deepcopy(task)
    assert verify(task) is task
    assert task == before


@pytest.mark.parametrize("case", ["production", "old_manifest", "missing_female", "dependency"])
def test_reject_release_regression(case):
    task = fixture()
    if case == "production":
        task["family"] = "StandinApp-ConverterTask"
    elif case == "old_manifest":
        task["containerDefinitions"][1]["environment"][2]["value"] = "body-models/old/manifest.json"
    elif case == "missing_female":
        task["containerDefinitions"][0]["environment"].pop()
    else:
        task["containerDefinitions"][0]["dependsOn"] = []
    with pytest.raises(ValueError):
        verify(task)
