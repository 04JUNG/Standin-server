from __future__ import annotations

import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]


def test_scene_camera_imports_from_the_converter_image_allowlist(tmp_path):
    """Catch imports that pass in the checkout but are absent from the image."""
    for line in (ROOT / "Dockerfile.converter").read_text().splitlines():
        if not line.lstrip().startswith("COPY "):
            continue
        parts = shlex.split(line, comments=True)
        if len(parts) != 3 or parts[0] != "COPY":
            continue
        source, destination = parts[1:]
        if not source.startswith("converter/") or not destination.startswith("/app/converter/"):
            continue
        target = tmp_path / destination.removeprefix("/app/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, target)
    subprocess.run(
        [sys.executable, "-I", "-c",
         "import sys; sys.path.insert(0, sys.argv[1]); "
         "from converter.scene import compose; "
         "from converter.scene_camera import framing_recipe, configure_camera, verify_camera_recipe; "
         "assert callable(compose) and callable(framing_recipe) and callable(configure_camera) and callable(verify_camera_recipe)",
         str(tmp_path)],
        check=True, cwd=tmp_path, capture_output=True, text=True,
    )


def _environment(container: dict) -> dict[str, str]:
    return {item["name"]: item["value"] for item in container["environment"]}


def test_repository_forces_lf_for_converter_integrity_hashes():
    attributes = (ROOT / ".gitattributes").read_text()
    assert "* text=auto eol=lf" in attributes


def test_converter_task_is_isolated_internal_fargate_contract():
    task = json.loads(
        (ROOT / "deploy/ecs/converter-task-definition.example.json").read_text()
    )
    assert task["family"] == "standin-converter"
    assert task["networkMode"] == "awsvpc"
    assert task["requiresCompatibilities"] == ["FARGATE"]
    assert task["runtimePlatform"]["cpuArchitecture"] == "X86_64"
    assert int(task["cpu"]) == 1024
    assert int(task["memory"]) == 2048

    assert [item["name"] for item in task["containerDefinitions"]] == ["converter"]
    container = task["containerDefinitions"][0]
    assert "/standin/converter:" in container["image"]
    assert container["user"] == "10001:10001"
    assert container["portMappings"][0]["containerPort"] == 8001
    assert container["stopTimeout"] > 30
    assert "/healthz" in " ".join(container["healthCheck"]["command"])
    assert container["healthCheck"]["startPeriod"] >= 120
    assert container["logConfiguration"]["logDriver"] == "awslogs"
    assert (
        container["logConfiguration"]["options"]["awslogs-group"]
        == "/ecs/standin/converter"
    )

    env = _environment(container)
    assert env["CONVERTER_TIMEOUT_SECONDS"] == "30"
    assert env["CONVERTER_MAX_CONCURRENT_PROCESSES"] == "1"
    assert env["CONVERTER_FORCE_EXACT_V324"] == "false"
    assert env["CONVERTER_JSON_LOGS"] == "1"
    assert env["STANDIN_MASTER_V2_URI"].startswith("s3://")
    assert env["STANDIN_MASTER_V2_URI"].endswith(
        "/characters/standin-master-v2.fbx"
    )
    assert not container.get("mountPoints")
    assert all(item["name"] != "inference" for item in task["containerDefinitions"])


def test_converter_service_network_has_no_public_ip():
    service = json.loads(
        (ROOT / "deploy/ecs/converter-service-network.example.json").read_text()
    )
    assert service["serviceName"] == "standin-converter"
    assert service["launchType"] == "FARGATE"
    assert service["healthCheckGracePeriodSeconds"] >= 120
    network = service["networkConfiguration"]["awsvpcConfiguration"]
    assert network["assignPublicIp"] == "DISABLED"
    assert network["subnets"]
    assert all("private" in subnet for subnet in network["subnets"])
    assert network["securityGroups"] == ["<converter-sg-allow-bff-and-inference>"]
    breaker = service["deploymentConfiguration"]["deploymentCircuitBreaker"]
    assert breaker == {"enable": True, "rollback": True}


def test_converter_deploy_is_gated_and_path_filtered():
    workflow = (ROOT / ".github/workflows/converter-deploy.yml").read_text()
    assert "ECR_REPOSITORY: standin/converter" in workflow
    assert "CONVERTER_AUTO_DEPLOY_ENABLED == 'true'" in workflow
    assert "branches: [main, develop]" in workflow
    assert (
        "environment: ${{ github.ref == 'refs/heads/main' && 'beta' || 'staging' }}"
        in workflow
    )
    assert "ECS_CLUSTER: ${{ vars.CONVERTER_ECS_CLUSTER }}" in workflow
    assert "ECS_SERVICE: ${{ vars.CONVERTER_ECS_SERVICE }}" in workflow
    assert "aws-region: ${{ vars.AWS_REGION }}" in workflow
    assert ":latest'" in workflow
    assert ":develop'" in workflow
    assert "Verify isolated ECR target exists" in workflow
    assert "Verify optional isolated ECS target exists" in workflow
    assert workflow.count("if: env.ECS_SERVICE != ''") == 4
    assert 'file: Dockerfile.converter' in workflow
    assert 'container-name: converter' in workflow
    assert '"converter/**"' in workflow
    assert '"converter_api/**"' in workflow
    assert '"deploy/ecs/**"' in workflow
    assert "container-name: inference" not in workflow


def _job_condition(filename: str, job: str) -> str:
    workflow = (ROOT / ".github/workflows" / filename).read_text()
    block = re.search(
        rf"(?ms)^  {re.escape(job)}:\n(.*?)(?=^  [\w-]+:|\Z)",
        workflow.split("\njobs:\n", 1)[1],
    ).group(1)
    condition = re.search(r"(?m)^    if: (.+)$", block).group(1)
    if condition == ">-":
        condition = " ".join(
            re.search(r"(?m)^    if: >-\n((?:      .+\n)+)", block)
            .group(1).splitlines()
        )
    return condition


def _enabled(condition, event, branch, target, deploy_staging, auto):
    # Evaluate the actual simple job guards, including short-circuit operators,
    # so a missing branch/event/target guard fails the routing matrix below.
    expression = condition.replace("&&", " and ").replace("||", " or ")
    expression = expression.replace("vars.", "variables.")
    return bool(eval(expression, {"__builtins__": {}}, {
        "github": SimpleNamespace(event_name=event, ref="refs/heads/" + branch),
        "inputs": SimpleNamespace(target=target, deploy_staging=deploy_staging),
        "variables": SimpleNamespace(CONVERTER_AUTO_DEPLOY_ENABLED=str(auto).lower()),
    }))


def test_manual_target_routing_never_deploys_worker_outside_develop():
    converter = _job_condition("converter-deploy.yml", "deploy")
    worker = _job_condition("converter-deploy.yml", "scene_worker")
    callee = _job_condition("scene-worker-deploy.yml", "staging")
    # Expected caller jobs. The worker's second guard must independently deny
    # main/feature refs even if a caller is accidentally broadened later.
    cases = [
        ("push", "develop", "", False, False, set()),
        ("push", "develop", "", False, True, {"converter"}),
        ("push", "main", "", False, True, {"converter"}),
        ("workflow_dispatch", "develop", "converter", True, False, {"converter"}),
        ("workflow_dispatch", "develop", "converter", False, False, set()),
        ("workflow_dispatch", "develop", "scene-worker", True, False, {"worker"}),
        ("workflow_dispatch", "develop", "scene-worker", True, True, {"worker"}),
        ("workflow_dispatch", "develop", "scene-worker", False, True, {"worker"}),
        ("workflow_dispatch", "main", "scene-worker", True, True, set()),
        ("workflow_dispatch", "feature", "scene-worker", True, True, set()),
        ("pull_request", "develop", "scene-worker", True, False, set()),
    ]
    for event, branch, target, deploy_staging, auto, expected in cases:
        args = event, branch, target, deploy_staging, auto
        actual = {
            name for name, condition in (("converter", converter), ("worker", worker))
            if _enabled(condition, *args)
        }
        assert actual == expected, args
        assert _enabled(callee, *args) == (branch == "develop"), args


def test_worker_delegation_uses_same_commit_and_explicit_staging_opt_in():
    caller = (ROOT / ".github/workflows/converter-deploy.yml").read_text()
    worker = (ROOT / ".github/workflows/scene-worker-deploy.yml").read_text()
    delegation = caller.split("  scene_worker:\n", 1)[1]
    assert "default: converter" in caller
    assert "uses: ./.github/workflows/scene-worker-deploy.yml" in delegation
    assert "deploy_staging: ${{ inputs.deploy_staging }}" in delegation
    assert "secrets: inherit" not in delegation
    assert "  workflow_call:\n    inputs:\n      deploy_staging:\n        type: boolean" in worker
    assert "    environment: staging" in worker
    assert "group: scene-worker-staging" in worker
    assert "SCENE_AWS_DEPLOY_ROLE" in worker
    assert "CONVERTER_AWS_DEPLOY_ROLE" not in worker
    assert worker.count("if: inputs.deploy_staging") == 2
    assert "wait-for-service-stability: true" in worker
    assert "python scripts/validate_scene_task.py /tmp/scene-task.json" in worker
    triggers = worker.split("\npermissions:", 1)[0]
    assert "\n  push:" not in triggers
    assert "\n  pull_request:" not in triggers


def test_converter_ci_exercises_real_skin_screen_endpoint():
    workflow = (ROOT / '.github/workflows/converter-ci.yml').read_text()
    assert 'http://127.0.0.1:8001/screen-hand-contact' in workflow
    assert '--skin-screen "$ARTIFACT_ROOT/http-skin-screen.json"' in workflow
    dockerfile = (ROOT / 'Dockerfile.converter').read_text()
    assert 'COPY converter/hand_contact.py /app/converter/hand_contact.py' in dockerfile


def test_converter_ci_smokes_dual_artifact_bundle_endpoint():
    workflow = (ROOT / ".github/workflows/converter-ci.yml").read_text()
    assert "http://127.0.0.1:8001/convert-bundle" in workflow
    assert "--form artifact_kind=base" in workflow
    assert '--form "expected_bvh_sha256=$FINAL_BVH_SHA256"' in workflow
    assert '--bundle "$ARTIFACT_ROOT/http-smoke.zip"' in workflow
    assert '--bundle-headers "$ARTIFACT_ROOT/bundle-response.headers"' in workflow


def test_converter_ci_smokes_refine_thumbnail_endpoint_with_real_blender():
    """headless 컨테이너에서 EEVEE/Cycles가 실제로 그림을 쓰는지는 CI만 증명한다."""
    workflow = (ROOT / ".github/workflows/converter-ci.yml").read_text()
    assert "http://127.0.0.1:8001/render-thumbnail" in workflow
    assert "--form view=front" in workflow
    assert '--thumbnail "$ARTIFACT_ROOT/http-smoke-thumbnail.png"' in workflow
    assert '--thumbnail-headers "$ARTIFACT_ROOT/thumbnail-response.headers"' in workflow
    dockerfile = (ROOT / "Dockerfile.converter").read_text()
    assert "COPY converter/thumbnail_render.py /app/converter/thumbnail_render.py" in dockerfile
    requirements = (ROOT / "requirements-converter.txt").read_text()
    assert "Pillow==" in requirements


def test_required_pr_checks_workflow_is_not_path_filtered():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    pull_request_trigger = workflow.split("  pull_request:", 1)[1].split(
        "  push:", 1
    )[0]
    assert "paths:" not in pull_request_trigger
    assert "name: PR checks" in workflow


def test_converter_image_defines_runtime_healthcheck_and_json_logs():
    dockerfile = (ROOT / "Dockerfile.converter").read_text()
    assert "HEALTHCHECK --interval=30s" in dockerfile
    assert "http://127.0.0.1:8001/healthz" in dockerfile
    assert "CONVERTER_JSON_LOGS=1" in dockerfile
    assert "DEPLOYMENT_VERSION=$DEPLOYMENT_VERSION" in dockerfile
