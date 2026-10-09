"""Strict frontend selection for the opt-in hybrid CLI, not /analyze."""

from ..config import CFG
from ..pose import RTMPoseModel
from ..pose_humanart import HumanArtPoseModel
from ..pose_contract import PoseContractError

DEFAULT_FRONTEND = "humanart-m"


def build_frontend(variant=DEFAULT_FRONTEND, manifest_path=None):
    """Return (real model, model-owned localization threshold).

    No mock/current-X fallback and no manifest/canary/license approval bypass.
    This extraction-only path does not own downstream search calibration.
    """
    if variant == "current-x":
        return RTMPoseModel(), 0.3
    if variant != "humanart-m":
        raise ValueError("Unknown hybrid pose frontend")
    model = HumanArtPoseModel(
        manifest_path=manifest_path, calibration_owner="manifest-rescue"
    )
    if CFG.is_production:
        identity = model.runtime_identity()
        if identity.get("license_review") != "approved" or identity.get(
            "status"
        ) not in {"shadow", "canary", "promoted"}:
            raise PoseContractError(
                "Human-Art production requires approved license and rollout manifest"
            )
    threshold = float(model.bundle.calibration["skeleton_kpt_threshold"])
    return model, threshold
