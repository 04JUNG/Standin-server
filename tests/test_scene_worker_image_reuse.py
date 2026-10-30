"""A two-stage bootstrap must not overwrite an already published SHA image."""

import os
from pathlib import Path
import subprocess
import re
import textwrap

import pytest

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/scene-worker-deploy.yml"


@pytest.mark.parametrize("case,expected", [("exists", "true"), ("missing", "false"), ("denied", None)])
def test_only_missing_image_allows_build(tmp_path, case, expected):
    workflow = WORKFLOW.read_text()
    blocks = re.split(r"(?m)^      - name: ", workflow)
    by_name = {block.splitlines()[0]: block for block in blocks[1:]}
    code = textwrap.dedent(by_name["Check pinned commit image"].split("        run: |\n", 1)[1])
    # Isolate fixed scratch files used by the workflow on a disposable CI runner.
    code = code.replace("/tmp/scene-image", str(tmp_path / "scene-image"))
    aws = tmp_path / "aws"
    aws.write_text("""#!/bin/bash
case "$TEST_IMAGE_CASE" in
  exists) echo '{"imageDetails":[{}]}' ;;
  missing) echo 'An error occurred (ImageNotFoundException) when calling DescribeImages' >&2; exit 254 ;;
  denied) echo 'An error occurred (AccessDeniedException) when calling DescribeImages' >&2; exit 254 ;;
esac
""")
    aws.chmod(0o755)
    output = tmp_path / "output"
    result = subprocess.run(["bash", "-e", "-c", code], capture_output=True, text=True,
                            env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
                                 "GITHUB_OUTPUT": str(output), "ECR_REPOSITORY": "standin/scene-worker",
                                 "GITHUB_SHA": "a" * 40, "TEST_IMAGE_CASE": case})
    if expected is None:
        assert result.returncode != 0
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert output.read_text().strip() == "exists=" + expected
    assert "if: steps.existing.outputs.exists != 'true'" in by_name["Build images from the same commit"]
    assert "if: steps.existing.outputs.exists != 'true'" in by_name["Push immutable commit tag"]
    assert "if: steps.existing.outputs.exists == 'true'" in by_name["Load already published commit image"]
    assert "if:" not in by_name["Verify worker image imports"]
