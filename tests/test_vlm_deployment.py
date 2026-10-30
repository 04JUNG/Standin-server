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


def test_scene_worker_accepts_its_own_openai_secret():
    validate = pytest.importorskip("scripts.validate_scene_task").validate
    import json
    from pathlib import Path
    value = json.loads((Path(__file__).resolve().parents[1] /
        "deploy/ecs/scene-worker-task-definition.example.json").read_text())
    container = value["containerDefinitions"][0]
    for row in container["environment"]:
        if row["name"] == "SCENE_VLM_PROVIDER":
            row["value"] = "openai"
        if row["name"] == "SCENE_VLM_MODEL":
            row["value"] = "selected-scene-model"
        if row["name"] == "SCENE_ASSET_MANIFEST_SHA256":
            row["value"] = "a" * 64
    for row in container["secrets"]:
        if row["name"] == "GEMINI_API_KEY":
            row["name"] = "OPENAI_API_KEY"
    validate(value)


def test_serving_profile_migrates_legacy_task_without_github_variables():
    from scripts.prepare_vlm_deployment import serving_overrides
    original = task()
    original["containerDefinitions"][0]["environment"].extend([
        {"name": "POSE_GENERATION_PROVIDER", "value": "gemini"},
        {"name": "POSE_GENERATION_MODEL", "value": "gemini-legacy"},
        {"name": "BODY_COARSE_FAMILY_REVIEW_MODEL", "value": "gemini-legacy"},
        {"name": "SKELETON_REPAIR_MODEL", "value": "gemini-legacy"},
        {"name": "ROUTING_FULLBODY_ENABLED", "value": "1"},
    ])
    original["containerDefinitions"].append({"name": "asset-init", "image": "keep-init"})
    before = copy.deepcopy(original)
    result = prepare(original, serving_overrides({"VLM_PROVIDER": "", "OPENAI_MODEL": ""}))
    env = {row["name"]: row["value"] for row in result["containerDefinitions"][0].pop("environment")}
    for name in ("VLM_PROVIDER", "BODY_VLM_PROVIDER", "POSE_GENERATION_PROVIDER"):
        assert env[name] == "openai"
    for name in ("OPENAI_MODEL", "BODY_VLM_MODEL", "BODY_COARSE_FAMILY_REVIEW_MODEL", "POSE_GENERATION_MODEL", "SKELETON_REPAIR_MODEL"):
        assert env[name] == "gpt-6-luna"
    assert env["OPENAI_REASONING_EFFORT"] == "low"
    assert original == before
    before["containerDefinitions"][0].pop("environment")
    assert result == before


def test_serving_profile_allows_explicit_complete_rollback():
    from scripts.prepare_vlm_deployment import serving_overrides
    overrides = {name: "gemini" for name in ("VLM_PROVIDER", "BODY_VLM_PROVIDER", "POSE_GENERATION_PROVIDER")}
    overrides.update({name: "gemini-rollback" for name in (
        "BODY_VLM_MODEL", "BODY_COARSE_FAMILY_REVIEW_MODEL", "POSE_GENERATION_MODEL", "SKELETON_REPAIR_MODEL")})
    result = prepare(task(), serving_overrides(overrides))
    env = {row["name"]: row["value"] for row in result["containerDefinitions"][0]["environment"]}
    assert env["VLM_PROVIDER"] == "gemini"


def test_deploy_uses_serving_profile_but_evaluation_does_not():
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/deploy.yml").read_text())
    steps = workflow["jobs"]["deploy"]["steps"]
    provider = next(step for step in steps if step.get("name") == "Prepare environment-scoped VLM provider")
    assert provider["run"].strip() == "python scripts/prepare_vlm_deployment.py taskdef.json --serving"
    for step in steps:
        if "evaluation" in step.get("name", "").lower():
            assert "--serving" not in step.get("run", "")
