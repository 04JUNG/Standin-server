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


def test_deploy_preserves_infra_owned_model_source_and_keeps_fast_rollback():
    workflow = (ROOT / ".github" / "workflows" / "deploy.yml").read_text(
        encoding="utf-8"
    )
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "Verify Human-Art model source" in workflow
    assert 'select(.name == "POSE_MODEL_URI"' in workflow
    assert "POSE_MODEL_URI=${{ vars.POSE_MODEL_URI }}" not in workflow
    assert "POSE_MODELS_ROOT=/app/data/pose-models" not in workflow
    assert "POSE_MODELS_ROOT=/app/data/pose-models" in dockerfile
    assert "POSE_MODEL_VARIANT=current-x" in workflow
    assert "POSE_CANARY_STAGE=off" in workflow


def test_pose_bundle_runtime_stays_exactly_pinned():
    """번들 계약은 런타임 버전 문자열 완전일치를 요구한다.

    tests/test_pose_contract.py의 manifest 픽스처는 runtime 블록을
    installed_version()으로 만들어 자기 자신과 비교하므로, 범위 핀이
    올려버린 버전을 잡지 못한다. 실제로 onnxruntime이 1.28.0 → 1.29.0으로
    올라가면서 staging 기동이 깨졌다(이미지 e5dc04e).
    """
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    for package in ("onnxruntime", "rtmlib"):
        line = next(
            candidate
            for candidate in requirements.splitlines()
            if candidate.strip().startswith(package)
        )
        assert "==" in line, (
            f"{package}는 번들 manifest의 runtime과 완전일치해야 한다. "
            f"범위 핀은 재빌드만으로 cascade 기동을 깬다: {line!r}"
        )
