from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.hybrid_pose import frontend
from src.pose_contract import PoseContractError


def fake_model(license_review="pending", status="candidate"):
    return Mock(
        bundle=SimpleNamespace(calibration={"skeleton_kpt_threshold": 0.35}),
        runtime_identity=Mock(
            return_value={"license_review": license_review, "status": status}
        ),
    )


def test_default_humanart_uses_manifest_threshold():
    model = fake_model()
    with patch.object(
        frontend, "CFG", SimpleNamespace(is_production=False)
    ), patch.object(frontend, "HumanArtPoseModel", return_value=model) as ctor:
        actual, threshold = frontend.build_frontend()
        assert actual is model and threshold == 0.35
        ctor.assert_called_once_with(
            manifest_path=None, calibration_owner="manifest-rescue"
        )


def test_current_x_remains_explicit_option():
    with patch.object(frontend, "RTMPoseModel", return_value=object()) as ctor:
        model, threshold = frontend.build_frontend("current-x")
        assert model is ctor.return_value and threshold == 0.3


def test_initialization_failure_does_not_fallback():
    with patch.object(
        frontend, "HumanArtPoseModel", side_effect=PoseContractError("Missing bundle")
    ), patch.object(frontend, "RTMPoseModel") as old:
        with pytest.raises(PoseContractError):
            frontend.build_frontend()
        old.assert_not_called()


@pytest.mark.parametrize(
    "license_review,status", [("pending", "promoted"), ("approved", "candidate")]
)
def test_production_approval_is_not_bypassed(license_review, status):
    with patch.object(
        frontend, "CFG", SimpleNamespace(is_production=True)
    ), patch.object(
        frontend, "HumanArtPoseModel", return_value=fake_model(license_review, status)
    ):
        with pytest.raises(PoseContractError):
            frontend.build_frontend()


def test_approved_production_bundle_is_allowed():
    with patch.object(
        frontend, "CFG", SimpleNamespace(is_production=True)
    ), patch.object(
        frontend, "HumanArtPoseModel", return_value=fake_model("approved", "canary")
    ):
        assert frontend.build_frontend()[1] == 0.35


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        frontend.build_frontend("unknown")
