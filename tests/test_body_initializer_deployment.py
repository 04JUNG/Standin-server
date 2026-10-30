import copy
import pytest

from scripts.prepare_body_initializer_deployment import prepare

IMAGE = "136889124221.dkr.ecr.ap-northeast-2.amazonaws.com/standin/inference:" + "a" * 40


def task():
    return {"family": "environment-owned", "taskRoleArn": "keep-role", "containerDefinitions": [
        {"name": "inference", "image": "old-serving", "secrets": [{"name": "OPENAI_API_KEY", "valueFrom": "keep-reference"}]},
        {"name": "body-assets-init", "image": "missing-old-digest", "command": ["python", "-c", "print('body_assets_ready')"], "mountPoints": [{"sourceVolume": "body-assets", "containerPath": "/assets"}]},
    ]}


def test_replaces_missing_initializer_image_without_changing_its_contract():
    original = task()
    before = copy.deepcopy(original)
    result = prepare(original, IMAGE)
    assert original == before
    expected = copy.deepcopy(before)
    expected["containerDefinitions"][1]["image"] = IMAGE
    assert result == expected


def test_task_without_initializer_does_not_gain_one():
    original = task()
    original["containerDefinitions"].pop()
    assert prepare(original, IMAGE) == original


@pytest.mark.parametrize("image", ["repo:latest", IMAGE[:-40] + "develop", "untrusted:tag"])
def test_rejects_mutable_or_unrelated_images(image):
    with pytest.raises(ValueError, match="immutable"):
        prepare(task(), image)


def test_rejects_an_unknown_initializer_command():
    original = task()
    original["containerDefinitions"][1]["command"] = ["sh", "-c", "unknown"]
    with pytest.raises(ValueError, match="contract"):
        prepare(original, IMAGE)


def test_refresh_precedes_render_and_service_update():
    from pathlib import Path
    import yaml
    workflow = yaml.safe_load((Path(__file__).resolve().parents[1] / ".github/workflows/deploy.yml").read_text())
    steps = workflow["jobs"]["deploy"]["steps"]
    refresh = next(i for i, step in enumerate(steps) if step.get("name") == "Refresh body initializer to the deployment image")
    render = next(i for i, step in enumerate(steps) if step.get("id") == "render")
    assert refresh < render
    assert "--image" in steps[refresh]["run"]
