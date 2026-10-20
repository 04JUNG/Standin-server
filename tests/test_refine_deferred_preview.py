import numpy as np
import pytest
import api.app as api_app
from api.models import RefineRequest
from src.config import CFG
from tests.test_smoke import _synthetic_bvh
from tests.test_refine_v2 import _bvh_with_rotation, _target_kp


@pytest.mark.parametrize("allowed", [False, True])
def test_deferred_preview_preserves_solver_result_without_rendering(monkeypatch, tmp_path, allowed):
    base = _synthetic_bvh(str(tmp_path), "base.bvh")
    target = _bvh_with_rotation(str(tmp_path), "target.bvh", "LeftArm", 15.0)
    keypoints, scores = _target_kp(target)
    monkeypatch.setitem(api_app.STATE, "db_path", str(tmp_path / "unused.db"))
    monkeypatch.setattr(api_app, "get_pose_meta", lambda *_: {"bvh_path": base, "set_id": None})
    monkeypatch.setattr(CFG, "refine_v2_enabled", True)
    monkeypatch.setattr(CFG, "refine_observability_gate", False)
    calls = []
    monkeypatch.setattr(api_app, "_refine_thumbnail", lambda **kw: calls.append(kw))
    result = api_app.refine(RefineRequest(
        pose_id="pose", view="front", keypoints=np.asarray(keypoints).tolist(), scores=scores.tolist(),
        refine_allowed=allowed, refinable_limbs=["left_arm"], lower_body_observed=False,
        skeleton_state="valid", coverage_class="full", slot_origin="vlm",
        skeleton_source="full_image", gap_type="unknown", render_thumbnail=False,
    ))
    assert result.refined is allowed, result.reason
    assert bool(result.bvh) is allowed
    assert result.thumbnail is None
    assert calls == []
