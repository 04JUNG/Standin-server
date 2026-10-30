import copy
import pytest

from scripts.prepare_vlm_deployment import prepare


def task():
    return {"family": "environment-owned", "taskRoleArn": "keep-role", "containerDefinitions": [{
        "name": "inference", "image": "keep-image", "environment": [
            {"name": "VLM_PROVIDER", "value": "gemini"},
            {"name": "GEMINI_MODEL", "value": "gemini-current"},
            {"name": "BODY_MATCHING_MODE", "value": "auto"},
            {"name": "BODY_VLM_PROVIDER", "value": "gemini"},
            {"name": "BODY_VLM_MODEL", "value": "gemini-small"},
        ], "secrets": [{"name": "OPENAI_API_KEY", "valueFrom": "environment-owned-secret:openaiApiKey::"},
                       {"name": "GEMINI_API_KEY", "valueFrom": "environment-owned-secret:geminiApiKey::"}]}]}


def test_empty_rollout_changes_nothing():
    original = task()
    assert prepare(original, {}) == original


def test_scoped_switch_preserves_secrets_images_and_roles():
    original = task()
    before = copy.deepcopy(original)
    result = prepare(original, {"VLM_PROVIDER": "openai", "OPENAI_MODEL": "selected-primary",
                               "BODY_VLM_PROVIDER": "openai", "BODY_VLM_MODEL": "selected-small"})
    assert before == original
    container = result["containerDefinitions"][0]
    env = {row["name"]: row["value"] for row in container.pop("environment")}
    assert env["OPENAI_MODEL"] == "selected-primary" and env["BODY_VLM_MODEL"] == "selected-small"
    original["containerDefinitions"][0].pop("environment")
    assert result == original


def test_new_provider_needs_its_own_secret_and_explicit_models():
    original = task()
    with pytest.raises(ValueError, match="model"):
        prepare(original, {"VLM_PROVIDER": "openai"})
    original["containerDefinitions"][0]["secrets"] = []
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        prepare(original, {"VLM_PROVIDER": "openai", "OPENAI_MODEL": "selected-primary"})


def test_auxiliary_gemini_model_does_not_silently_survive_openai_switch():
    with pytest.raises(ValueError, match="model"):
        prepare(task(), {"VLM_PROVIDER": "openai", "OPENAI_MODEL": "selected-primary",
                         "BODY_VLM_PROVIDER": "openai"})


def test_unrelated_or_secret_overrides_never_enter_task_environment():
    original = task()
    assert prepare(original, {"OPENAI_API_KEY": "must-not-be-copied", "STAGING_EVAL_ENABLED": "false"}) == original


def test_plaintext_key_is_rejected_even_with_a_secret_reference():
    original = task()
    original["containerDefinitions"][0]["environment"].append({"name": "OPENAI_API_KEY", "value": "do-not-log"})
    with pytest.raises(ValueError, match="plaintext"):
        prepare(original, {"VLM_PROVIDER": "openai", "OPENAI_MODEL": "selected-primary"})
