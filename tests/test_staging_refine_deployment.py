"""Prevent staging rollback from changing production or unrelated runtime wiring."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("staging_refine", Path(__file__).resolve().parents[1] / "scripts/prepare_staging_refine_deployment.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


def task():
    return {"family": "StandinStagingAppInferenceTask", "cpu": "4096", "containerDefinitions": [
        {"name": "inference", "image": "original-image", "dependsOn": [{"containerName": "refine-rc4", "condition": "HEALTHY"}],
         "environment": [{"name": "REFINE_RELEASE_ENABLED", "value": "1"}, {"name": "REFINE_COLLISION_GATE", "value": "1"}],
         "secrets": [{"name": "OPENAI_API_KEY", "valueFrom": "unchanged-reference"}]},
        {"name": "refine-rc4", "image": "original-sidecar", "environment": [{"name": "REFINE_RELEASE_ENABLED", "value": "1"}]},
        {"name": "body-assets-init", "image": "original-init"},
    ]}


def test_only_main_refine_flags_change_and_policy_is_idempotent():
    original = task()
    before = deepcopy(original)
    after = deploy.prepare(original)
    expected = deepcopy(original)
    env = {r["name"]: r["value"] for r in expected["containerDefinitions"][0]["environment"]}
    env.update(deploy.DEFAULTS)
    expected["containerDefinitions"][0]["environment"] = [{"name": k, "value": v} for k, v in env.items()]
    assert original == before
    assert after == expected
    assert deploy.prepare(after) == after


@pytest.mark.parametrize("problem", ["production", "duplicate", "missing", "duplicate_env", "secret"])
def test_unsafe_target_fails_before_mutation(problem):
    value = task()
    if problem == "production": value["family"] = "StandinAppInferenceTask"
    if problem == "duplicate": value["containerDefinitions"].append(deepcopy(value["containerDefinitions"][0]))
    if problem == "missing": value["containerDefinitions"][0]["name"] = "other"
    if problem == "duplicate_env": value["containerDefinitions"][0]["environment"] *= 2
    if problem == "secret": value["containerDefinitions"][0]["secrets"].append({"name": "REFINE_RELEASE_ENABLED"})
    with pytest.raises(ValueError): deploy.prepare(value)


@pytest.mark.parametrize("override", [{"REFINE_ENABLED": "true"}, {"REFINE_DEFAULT_MODE": "unknown"}, {"REFINE_V25_SELECTOR_ENABLED": "0"}, {"OTHER": "1"}])
def test_invalid_override_is_rejected(override):
    with pytest.raises(ValueError): deploy.prepare(task(), override)


def test_emergency_disable_and_conservative_override_remain_available():
    result = deploy.prepare(task(), {"REFINE_ENABLED": "0", "REFINE_DEFAULT_MODE": "conservative"})
    assert {r["name"]: r["value"] for r in result["containerDefinitions"][0]["environment"]}["REFINE_ENABLED"] == "0"
