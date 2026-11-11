"""Production promotion must keep legacy refine and deny the new worker path."""
from pathlib import Path
import yaml
from scripts.prepare_production_refine import prepare
from src.config import Config
from src.refine import REFINE_V2_CODE_VERSION


def test_production_policy_replaces_experimental_settings_and_keeps_other_runtime():
    task = {"containerDefinitions": [
        {"name": "inference", "environment": [
            {"name": "REFINE_RELEASE_ENABLED", "value": "1"},
            {"name": "REFINE_RELEASE_WORKER_URL", "value": "http://staging-worker"},
            {"name": "REFINE_DEFAULT_MODE", "value": "conservative"},
            {"name": "POSE_LIBRARY_URI", "value": "keep-production-library"},
        ]}, {"name": "body-assets-init", "environment": []}]}
    result = prepare(task)
    env = {v["name"]: v["value"] for v in result["containerDefinitions"][0]["environment"]}
    assert env["REFINE_RELEASE_ENABLED"] == "0"
    assert "REFINE_RELEASE_WORKER_URL" not in env
    assert env["REFINE_ENABLED"] == env["REFINE_V2_ENABLED"] == env["REFINE_V25_SELECTOR_ENABLED"] == "1"
    assert env["REFINE_DEFAULT_MODE"] == "aggressive"
    assert env["REFINE_V3_MODE"] == "off"
    assert env["POSE_LIBRARY_URI"] == "keep-production-library"
    assert result["containerDefinitions"][1]["environment"] == []
    assert REFINE_V2_CODE_VERSION == "v2.5.4"


def test_policy_runs_only_for_production_before_render():
    steps = yaml.safe_load((Path(__file__).parents[1] / ".github/workflows/deploy.yml").read_text())["jobs"]["deploy"]["steps"]
    i = next(i for i, s in enumerate(steps) if s.get("name") == "Pin production refine policy")
    assert steps[i]["if"] == "github.ref == 'refs/heads/main'"
    assert i < next(i for i, s in enumerate(steps) if s.get("id") == "render")
