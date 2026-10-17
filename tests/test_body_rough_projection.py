"""Real failure modes: clothing, ownership, cropped windows, pose mismatch, ties."""

import copy
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw
from src.experimental.body_matching.rough_evidence import (
    PARTS,
    SAMPLES,
    parse_visibility,
    extract_contours,
)
from src.experimental.body_matching.rough_segmentation import extract_mask_contours
from src.experimental.body_matching.rough_pipeline import RoughBodyPipeline
from src.experimental.body_matching.rough_matching import (
    rank_profiles,
    pose_compatibility,
    verify_projection_group,
    sha256,
    compare_rough,
    part_labels,
)


def fixture():
    points = np.zeros((17, 2))
    points[5] = (20, 20)
    points[6] = (60, 20)
    points[11] = (25, 120)
    points[12] = (55, 120)
    mask = np.zeros(17, bool)
    mask[[5, 6, 11, 12]] = True
    im = Image.new("RGB", (140, 150), "white")
    ImageDraw.Draw(im).rectangle((15, 20, 65, 120), outline="black", width=2)
    sm = np.zeros((150, 140), bool)
    sm[20:121, 15:66] = True
    vis = dict(
        ownership_ambiguous=False,
        parts={
            p: dict(
                state="exposed" if p == "torso" else "uncertain",
                evidence="clear outline" if p == "torso" else "not visible",
            )
            for p in PARTS
        },
    )
    return im, points, mask, vis, {"torso": dict(mask=sm, predicted_iou=0.95)}


def test_clipped_search_window_keeps_clear_boundaries():
    im, p, m, v, s = fixture()
    e, *_ = extract_contours(im, p, m, v, ownership_verified=True)
    assert e["accepted_parts"] == ["torso"]
    assert len(e["parts"]["torso"]["samples"]) == 9


@pytest.mark.parametrize(
    "state", ["loose_clothing", "occluded", "out_of_frame", "symbolic", "uncertain"]
)
def test_hidden_and_symbolic_regions_never_become_body_evidence(state):
    im, p, m, v, s = fixture()
    v["parts"]["torso"]["state"] = state
    e, observed, excluded = extract_mask_contours(
        im, p, m, v, s, ownership_verified=True
    )
    assert not e["accepted_parts"] and not observed.any()
    assert bool(excluded.any()) == (state in {"loose_clothing", "occluded"})


def test_mask_needs_original_ink_support():
    im, p, m, v, s = fixture()
    accepted, *_ = extract_mask_contours(im, p, m, v, s, ownership_verified=True)
    assert accepted["accepted_parts"] == ["torso"]
    im.paste("white", (0, 0, *im.size))
    e, *_ = extract_mask_contours(im, p, m, v, s, ownership_verified=True)
    assert not e["accepted_parts"]


@pytest.mark.parametrize("iou", [0.3, float("nan"), float("inf")])
def test_segmentation_confidence_is_not_body_evidence(iou):
    im, p, m, v, s = fixture()
    s["torso"]["predicted_iou"] = iou
    e, *_ = extract_mask_contours(im, p, m, v, s, ownership_verified=True)
    assert not e["accepted_parts"]


def test_neighbor_ownership_rejects_boundaries():
    im, p, m, v, s = fixture()
    near = [(16, y) for y in range(20, 121, 2)]
    e, *_ = extract_mask_contours(
        im, p, m, v, s, ownership_verified=True, other_points=near
    )
    assert not e["accepted_parts"]


def test_unowned_case_calls_no_remote_or_segmentation():
    class Fail:
        def analyze(self, *args):
            raise AssertionError("remote called")

        def segment(self, *args):
            raise AssertionError("segmenter called")

    im, p, m, _, _ = fixture()
    pipe = RoughBodyPipeline(segmenter=Fail(), visibility_client=Fail())
    row, masks = pipe.observe(
        im,
        dict(case_id="a", points=p.tolist(), mask=m.tolist(), ownership_verified=False),
    )
    assert not row["segmentation"]["accepted_parts"] and not masks["observed"].any()


def test_visibility_identity_and_missing_part_fail_closed():
    _, _, _, v, _ = fixture()
    raw = dict(
        person_id="a",
        ownership_ambiguous=False,
        parts={str(i): v["parts"][p] for i, p in enumerate(PARTS, 1)},
    )
    assert parse_visibility({"people": [raw]}, ["a"])["a"]["parts"] == v["parts"]
    with pytest.raises(ValueError):
        parse_visibility({"people": [raw, raw]}, ["a"])
    raw["parts"].pop("9")
    with pytest.raises(ValueError):
        parse_visibility({"people": [raw]}, ["a"])


def profiles(parts=("torso", "left_upper_arm", "right_upper_arm")):
    ts = [round(float(t), 4) for t in SAMPLES]
    observed = {
        p: dict(accepted=True, samples=[dict(t=t, width=0.2) for t in ts])
        for p in parts
    }
    refs = {
        b: {p: {t: width for t in ts} for p in parts}
        for b, width in [("a", 0.2), ("b", 0.4), ("c", 0.6)]
    }
    return observed, refs


def test_clear_half_body_does_not_require_hidden_legs():
    q, r = profiles()
    out = rank_profiles(q, r, 200)
    assert not out["reason_codes"] and out["ranking"][0]["body_id"] == "a"
    assert out["regions"] == ["arms", "torso"]


def test_arm_only_cannot_determine_whole_fine_body():
    q, r = profiles(("left_upper_arm", "right_upper_arm", "left_forearm"))
    assert (
        "insufficient_independent_regions" in rank_profiles(q, r, 200)["reason_codes"]
    )


def test_near_tie_is_not_resolved_by_id_or_missing_regions():
    q, r = profiles()
    r["b"] = copy.deepcopy(r["a"])
    r["c"]["left_calf"] = {0.5: 10}
    out = rank_profiles(q, r, 200)
    assert (
        out["margin"] == 0 and "margin_below_contour_uncertainty" in out["reason_codes"]
    )
    assert "left_calf" not in out["common_parts"]


def test_identical_station_support_for_every_candidate():
    q, r = profiles()
    r["c"]["torso"] = {0.2: 0.2}
    out = rank_profiles(q, r, 200)
    assert "torso" not in out["common_parts"]
    assert "insufficient_independent_regions" in out["reason_codes"]


def test_part_ablation_detects_unstable_winner():
    q, r = profiles()
    for p in r["a"]:
        r["a"][p] = {t: (0.2 if p == "torso" else 0.3) for t in r["a"][p]}
    for p in r["b"]:
        r["b"][p] = {t: (0.5 if p == "torso" else 0.2) for t in r["b"][p]}
    assert "part_ablation_changes_winner" in rank_profiles(q, r, 200)["reason_codes"]


def test_pose_gate_does_not_confuse_changed_limb_length_with_direction():
    _, p, m, _, _ = fixture()
    p[7] = [20, 70]
    m[7] = True
    ref = dict(keypoints=p.copy(), valid_joint_mask=m)
    ref["keypoints"][7] = [20, 90]
    assert pose_compatibility(p, m, ref, ["torso", "left_upper_arm"])["passed"]
    ref["keypoints"][7] = [80, 20]
    assert not pose_compatibility(p, m, ref, ["torso", "left_upper_arm"])["passed"]


def group(tmp_path):
    _, p, m, _, _ = fixture()
    pose = tmp_path / "pose.bvh"
    pose.write_text("fixture")
    im = tmp_path / "parts.png"
    Image.new("RGBA", (10, 10), (255, 0, 0, 255)).save(im)
    root = Path(__file__).resolve().parents[1]
    job = dict(
        key="k", pose_id="p", view="front", pose_sha256=sha256(pose), bvh_path=str(pose)
    )
    assets = [dict(body_id=x, asset_sha256=x * 64) for x in ("a", "b")]
    rec = []
    for a in assets:
        rec.append(
            dict(
                **a,
                **{k: job[k] for k in ("key", "pose_id", "view", "pose_sha256")},
                image=str(im),
                image_sha256=sha256(im),
                parts=list(PARTS),
                mesh_finite=True,
                camera={"yaw": 0},
                renderer_sha256=sha256(
                    root / "scripts/render_body_pose_projections.py"
                ),
                retarget_sha256=sha256(root / "converter/retarget.py"),
                colors_linear=[[1, 0, 0]],
                keypoints=p.tolist(),
                valid_joint_mask=m.tolist()
            )
        )
    return rec, assets, job


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "view", "pose", "camera", "image", "code", "asset"],
)
def test_projection_identity_hashes_and_shared_camera(tmp_path, mutation):
    rec, a, j = group(tmp_path)
    verify_projection_group(rec, a, j)
    if mutation == "missing":
        rec.pop()
    elif mutation == "duplicate":
        rec[1] = copy.deepcopy(rec[0])
    elif mutation == "view":
        rec[1]["view"] = "side"
    elif mutation == "pose":
        rec[1]["pose_sha256"] = "changed"
    elif mutation == "camera":
        rec[1]["camera"]["yaw"] = 25
    elif mutation == "image":
        rec[1]["image_sha256"] = "changed"
    elif mutation == "code":
        rec[1]["renderer_sha256"] = "changed"
    elif mutation == "asset":
        rec[1]["asset_sha256"] = "changed"
    with pytest.raises(ValueError):
        verify_projection_group(rec, a, j)


def test_transparent_and_unknown_palette_pixels_are_not_body():
    im = Image.new("RGBA", (3, 1))
    im.putdata([(255, 0, 0, 255), (255, 0, 0, 0), (100, 100, 100, 255)])
    assert part_labels(im, [[1, 0, 0]]).tolist() == [[1, 0, 0]]


def test_every_result_keeps_runtime_selection_unset():
    result = compare_rough(
        dict(case_id="a", ownership_verified=True, segmentation={"accepted_parts": []}),
        [],
        [],
        [],
    )
    assert (
        result["status"] == "held"
        and result["selected_body_id"] is None
        and not result["auto_apply"]
    )
