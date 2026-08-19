"""Fixed-mask blind-render regression tests."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from standin_eval.refine_evaluator import EVALUATOR_VERSION, query_evidence
from standin_eval.refine_render import (
    RENDERER_VERSION,
    normalize_pose,
    project_bvh,
    render_blind_artifact,
    shared_bounds,
)
from tests.test_smoke import _synthetic_bvh, _target_kp


def test_metric_and_renderer_algorithm_versions_are_bumped():
    assert EVALUATOR_VERSION == "refine-external-v1.1"
    assert RENDERER_VERSION == "coco17-blind-svg-v2"


def test_renderer_normalization_matches_frozen_evaluator_coordinates():
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        keypoints, scores = _target_kp(base)
    scores = np.asarray(scores, dtype=np.float64)
    scores[9] = 0.2
    evidence = query_evidence(keypoints, scores, score_threshold=0.3)
    assert evidence["valid"], evidence

    normalized, render_scores = normalize_pose(
        keypoints,
        scores,
        score_threshold=evidence["score_threshold"],
        valid_mask=evidence["target_valid_mask"],
    )

    expected = np.asarray(evidence["normalized_target"], dtype=np.float64)
    valid = np.asarray(evidence["target_valid_mask"], dtype=bool)
    assert np.allclose(
        normalized[valid], expected[valid], atol=1e-12, rtol=0.0,
    )
    assert np.allclose(normalized[~valid], 0.0)
    assert render_scores[9] == 0.0
    assert np.allclose(normalized[9], 0.0)


def test_legacy_renderer_call_derives_the_evaluator_default_mask():
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        keypoints, scores = _target_kp(base)
    scores = np.asarray(scores, dtype=np.float64)
    scores[9] = 0.29
    legacy_points, legacy_scores = normalize_pose(keypoints, scores)
    evidence = query_evidence(keypoints, scores)
    explicit_points, explicit_scores = normalize_pose(
        keypoints,
        scores,
        score_threshold=evidence["score_threshold"],
        valid_mask=evidence["target_valid_mask"],
    )
    assert np.array_equal(legacy_points, explicit_points)
    assert np.array_equal(legacy_scores, explicit_scores)
    assert legacy_scores[9] == 0.0


def test_scores_from_point_one_through_point_two_nine_are_not_rendered():
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        keypoints, scores = _target_kp(base)
        scores = np.asarray(scores, dtype=np.float64)
        scores[7], scores[9], scores[13] = 0.1, 0.2, 0.29
        scores[15] = 0.3  # threshold is inclusive, as in query_evidence
        evidence = query_evidence(keypoints, scores, score_threshold=0.3)
        assert evidence["valid"], evidence
        assert not any(evidence["target_valid_mask"][i] for i in (7, 9, 13))
        assert evidence["target_valid_mask"][15]

        target = normalize_pose(
            keypoints,
            scores,
            score_threshold=evidence["score_threshold"],
            valid_mask=evidence["target_valid_mask"],
        )
        target_pose = project_bvh(
            base,
            "front",
            score_threshold=evidence["score_threshold"],
            valid_mask=evidence["target_valid_mask"],
        )
        safety_pose = project_bvh(base, "three_quarter")
        output = Path(directory) / "blind.svg"
        render_blind_artifact(
            artifact_path=base,
            target_keypoints=keypoints,
            target_scores=scores,
            target_view="front",
            safety_view="three_quarter",
            target_bounds=shared_bounds([target, target_pose]),
            safety_bounds=shared_bounds([safety_pose]),
            output_path=output,
            score_threshold=evidence["score_threshold"],
            target_valid_mask=evidence["target_valid_mask"],
        )

        root = ET.parse(output).getroot()

    elements = list(root.iter())
    red_circles = [
        element for element in elements
        if element.tag.endswith("circle") and element.attrib.get("fill") == "#d24b40"
    ]
    red_lines = [
        element for element in elements
        if element.tag.endswith("line") and element.attrib.get("stroke") == "#d24b40"
    ]
    # COCO body has 12 drawn joints. Three low-score joints and their four
    # incident edges are absent; the score==0.3 ankle remains visible.
    assert len(red_circles) == 9
    assert len(red_lines) == 8
    metadata = next(
        element.text for element in elements if element.tag.endswith("metadata")
    )
    assert RENDERER_VERSION in metadata
    assert "score_threshold=0.3" in metadata
    for index in (7, 9, 13):
        assert metadata.split("target_valid_mask=", 1)[1][index] == "0"
    assert metadata.split("target_valid_mask=", 1)[1][15] == "1"


def test_missing_query_can_render_a_blind_safety_card():
    with tempfile.TemporaryDirectory() as directory:
        base = _synthetic_bvh(directory, "base.bvh")
        output = Path(directory) / "missing-target.svg"
        render_blind_artifact(
            artifact_path=base,
            target_keypoints=[], target_scores=[],
            target_view="front", safety_view="side",
            target_bounds=(-1.0, -1.0, 1.0, 1.0),
            safety_bounds=(-1.0, -1.0, 1.0, 1.0),
            output_path=output, allow_missing_target=True,
        )
        svg = output.read_text(encoding="utf-8")
        assert "query skeleton unavailable" in svg
        assert "target_available=false" in svg


if __name__ == "__main__":
    import traceback

    functions = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failed = 0
    for function in functions:
        try:
            function()
            print(f"PASS {function.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {function.__name__}")
            traceback.print_exc()
    print(f"\n{len(functions) - failed}/{len(functions)} passed")
    raise SystemExit(1 if failed else 0)
