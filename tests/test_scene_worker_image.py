"""A deployment retry must reuse an immutable image, and fail closed on AWS errors."""

from types import SimpleNamespace

import pytest

from scripts.scene_worker_image import REPOSITORY, resolve_image


COMMIT = "c" * 40
DIGEST = "sha256:" + "d" * 64


def client(*, mutability="IMMUTABLE", images=None, error=None):
    def describe_images(**kwargs):
        assert kwargs == {"repositoryName": REPOSITORY, "imageIds": [{"imageTag": COMMIT}]}
        if error:
            raise error
        return {"imageDetails": images if images is not None else [{"imageTags": [COMMIT], "imageDigest": DIGEST}]}

    return SimpleNamespace(
        describe_repositories=lambda **kwargs: {"repositories": [{"imageTagMutability": mutability}]},
        describe_images=describe_images,
    )


def test_retry_reuses_existing_commit_digest():
    assert resolve_image(client(), REPOSITORY, COMMIT) == DIGEST


def aws_error(code):
    error = RuntimeError("AWS request failed")
    error.response = {"Error": {"Code": code}}
    return error


def test_only_image_not_found_allows_first_build():
    assert resolve_image(client(error=aws_error("ImageNotFoundException")), REPOSITORY, COMMIT) is None


@pytest.mark.parametrize("code", ["AccessDeniedException", "RepositoryNotFoundException", "ThrottlingException"])
def test_aws_failure_is_not_a_cache_miss(code):
    with pytest.raises(RuntimeError):
        resolve_image(client(error=aws_error(code)), REPOSITORY, COMMIT)


@pytest.mark.parametrize("mutability", ["MUTABLE", "IMMUTABLE_WITH_EXCLUSION", None])
def test_repository_must_protect_every_commit_tag(mutability):
    with pytest.raises(ValueError, match="immutable"):
        resolve_image(client(mutability=mutability), REPOSITORY, COMMIT)


@pytest.mark.parametrize("images", [[], [{"imageTags": ["other"], "imageDigest": DIGEST}], [{"imageTags": [COMMIT], "imageDigest": "not-a-digest"}]])
def test_bad_lookup_cannot_be_deployed(images):
    with pytest.raises(ValueError):
        resolve_image(client(images=images), REPOSITORY, COMMIT)


@pytest.mark.parametrize("repository,commit", [("standin/inference", COMMIT), (REPOSITORY, "develop"), (REPOSITORY, "c" * 39)])
def test_other_targets_and_moving_tags_are_rejected(repository, commit):
    with pytest.raises(ValueError):
        resolve_image(client(), repository, commit)


def test_workflow_reuses_images_and_deploys_the_resolved_digest():
    from pathlib import Path

    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/scene-worker-deploy.yml").read_text()
    assert workflow.count("if: steps.lookup.outputs.found != 'true'") == 2
    assert "--require" in workflow
    assert "@${{ steps.image.outputs.digest }}" in workflow
    assert "if: github.ref == 'refs/heads/develop'" in workflow
