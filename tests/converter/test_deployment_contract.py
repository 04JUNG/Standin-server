from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _environment(container: dict) -> dict[str, str]:
    return {item["name"]: item["value"] for item in container["environment"]}


def test_converter_task_is_isolated_internal_fargate_contract():
    task = json.loads(
        (ROOT / "deploy/ecs/converter-task-definition.example.json").read_text()
    )
    assert task["family"] == "standin-converter"
    assert task["networkMode"] == "awsvpc"
    assert task["requiresCompatibilities"] == ["FARGATE"]
    assert task["runtimePlatform"]["cpuArchitecture"] == "X86_64"
    assert int(task["cpu"]) >= 2048
    assert int(task["memory"]) >= 4096

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
    assert env["CONVERTER_JSON_LOGS"] == "1"
    assert env["STANDIN_MASTER_V2_URI"].startswith("/characters/")
    character_mount = next(
        mount for mount in container["mountPoints"]
        if mount["containerPath"] == "/characters"
    )
    assert character_mount["readOnly"] is True
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
    assert network["securityGroups"] == ["<converter-sg-allow-bff-only>"]
    breaker = service["deploymentConfiguration"]["deploymentCircuitBreaker"]
    assert breaker == {"enable": True, "rollback": True}


def test_converter_deploy_is_gated_and_path_filtered():
    workflow = (ROOT / ".github/workflows/converter-deploy.yml").read_text()
    assert "ECR_REPOSITORY: standin/converter" in workflow
    assert "CONVERTER_AUTO_DEPLOY_ENABLED == 'true'" in workflow
    assert 'file: Dockerfile.converter' in workflow
    assert 'container-name: converter' in workflow
    assert '"converter/**"' in workflow
    assert '"converter_api/**"' in workflow
    assert '"deploy/ecs/**"' in workflow
    assert "container-name: inference" not in workflow


def test_converter_ci_smokes_dual_artifact_bundle_endpoint():
    workflow = (ROOT / ".github/workflows/converter-ci.yml").read_text()
    assert "http://127.0.0.1:8001/convert-bundle" in workflow
    assert "--form artifact_kind=base" in workflow
    assert '--form "expected_bvh_sha256=$FINAL_BVH_SHA256"' in workflow
    assert '--bundle "$ARTIFACT_ROOT/http-smoke.zip"' in workflow
    assert '--bundle-headers "$ARTIFACT_ROOT/bundle-response.headers"' in workflow


def test_converter_image_defines_runtime_healthcheck_and_json_logs():
    dockerfile = (ROOT / "Dockerfile.converter").read_text()
    assert "HEALTHCHECK --interval=30s" in dockerfile
    assert "http://127.0.0.1:8001/healthz" in dockerfile
    assert "CONVERTER_JSON_LOGS=1" in dockerfile
    assert "DEPLOYMENT_VERSION=$DEPLOYMENT_VERSION" in dockerfile
