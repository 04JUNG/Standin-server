"""Offline task rendering: opt-in settings must not leak across environments."""
import copy
from pathlib import Path

import pytest

from scripts.prepare_description_deployment import NAMES, prepare


def task():
    return {"taskRoleArn": "existing-role", "containerDefinitions": [
        {"name": "inference", "image": "approved-image", "environment": [
            {"name": "PARALLEL_POSE_DESCRIPTION", "value": "1"},
            {"name": "POSE_LIBRARY_URI", "value": "existing-library"}],
         "secrets": [{"name": "GEMINI_API_KEY", "valueFrom": "existing-secret"}],
         "healthCheck": {"command": ["existing-probe"]}},
        {"name": "other", "image": "other-image"}]}


def test_empty_github_variables_preserve_existing_task_exactly():
    original = task()
    assert prepare(original, {k: "" for k in NAMES}) == original


def test_explicit_zero_rolls_back_without_replacing_image_roles_secrets_or_other_settings():
    original = task()
    before = copy.deepcopy(original)
    result = prepare(original, {"PARALLEL_POSE_DESCRIPTION": "0",
                                "ROUTING_DESCRIBE_PERSON_CROPS": "0"})
    assert original == before
    env = result["containerDefinitions"][0].pop("environment")
    assert {r["name"]: r["value"] for r in env} == {
        "PARALLEL_POSE_DESCRIPTION": "0", "ROUTING_DESCRIBE_PERSON_CROPS": "0",
        "POSE_LIBRARY_URI": "existing-library"}
    before["containerDefinitions"][0].pop("environment")
    assert result == before


@pytest.mark.parametrize("name,value", [
    ("ROUTING_DESCRIBE_PERSON_CROPS", "true"),
    ("PARALLEL_POSE_DESCRIPTION", "2"),
    ("ROUTING_DESCRIBE_PERSON_WORKERS", "3"),
    ("ROUTING_DESCRIBE_BUDGET_MS", "0"),
    ("ROUTING_DESCRIBE_TIMEOUT_MS", "120001"),
    ("ROUTING_DESCRIBE_RESERVE_MS", "-1"),
    ("ROUTING_DESCRIBE_BUDGET_MS", "45000\nINJECT=1"),
    ("GEMINI_API_KEY", "1"),
])
def test_invalid_rollout_stops_before_task_registration(name, value):
    with pytest.raises(ValueError):
        prepare(task(), {name: value})


def test_duplicate_environment_and_secret_collision_are_not_silently_overwritten():
    value = task()
    value["containerDefinitions"][0]["environment"].append(
        {"name": "PARALLEL_POSE_DESCRIPTION", "value": "0"})
    with pytest.raises(ValueError, match="duplicate"):
        prepare(value, {})
    value = task()
    value["containerDefinitions"][0]["secrets"].append(
        {"name": "ROUTING_DESCRIBE_PERSON_CROPS", "valueFrom": "existing-secret"})
    with pytest.raises(ValueError, match="secret"):
        prepare(value, {"ROUTING_DESCRIBE_PERSON_CROPS": "1"})


def test_environment_scoped_overrides_are_prepared_before_image_render():
    workflow = (Path(__file__).resolve().parents[1] /
                ".github/workflows/deploy.yml").read_text()
    start = workflow.index("      - name: Prepare optional description rollout overrides")
    end = workflow.index("\n      - ", start + 1)
    step = workflow[start:end]
    assert end < workflow.index("id: render")
    env_start = step.index("        env:\n") + len("        env:\n")
    env_end = step.index("        run:", env_start)
    env = dict(line.strip().split(": ", 1) for line in step[env_start:env_end].splitlines())
    assert env == {k: "${{ vars." + k + " }}" for k in NAMES}
    assert "run: python scripts/prepare_description_deployment.py taskdef.json" in step
