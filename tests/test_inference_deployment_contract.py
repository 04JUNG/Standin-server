"""Inference deployment contract for the full Human-Art rescue promotion."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_deploy_defaults_to_full_cascade_with_strict_startup():
    workflow = (ROOT / ".github" / "workflows" / "deploy.yml").read_text(
        encoding="utf-8"
    )

    assert "APP_ENV=production" in workflow
    assert "POSE_BACKEND=rtmlib" in workflow
    assert "vars.POSE_MODEL_VARIANT || 'cascade'" in workflow
    assert "vars.POSE_CANARY_STAGE || 'canary-100'" in workflow
    assert "POSE_STRICT=1" in workflow


def test_deploy_requires_existing_manifest_and_keeps_fast_rollback():
    workflow = (ROOT / ".github" / "workflows" / "deploy.yml").read_text(
        encoding="utf-8"
    )

    assert "Verify Human-Art promotion prerequisites" in workflow
    assert 'select(.name == "POSE_MODEL_MANIFEST"' in workflow
    assert "POSE_MODEL_VARIANT=current-x" in workflow
    assert "POSE_CANARY_STAGE=off" in workflow
