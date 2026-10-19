"""ECS must replace inherited version overrides when deploying a new image."""
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("workflow,container", [
    ("deploy.yml", "inference"),
    ("converter-deploy.yml", "converter"),
])
def test_runtime_version_is_set_on_the_container_render_step(workflow, container):
    text = (ROOT / ".github/workflows" / workflow).read_text()
    render = text.split("uses: aws-actions/amazon-ecs-render-task-definition@v1", 1)[1]
    render = render.split("\n      - ", 1)[0]
    assert f"container-name: {container}" in render
    # A Docker build-arg alone does not replace a stale task-definition env var.
    assert "image: ${{ steps.ecr.outputs.registry }}/${{ env.ECR_REPOSITORY }}:${{ github.sha }}" in render
    assert "environment-variables: |" in render
    assignments = render.split("environment-variables: |", 1)[1].splitlines()
    versions = [line.strip() for line in assignments
                if line.strip().startswith("DEPLOYMENT_VERSION=")]
    assert versions == ["DEPLOYMENT_VERSION=${{ github.sha }}"]
