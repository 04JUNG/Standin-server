"""Stage 0 evaluation harness contracts (pytest-free runner included)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from unittest import SkipTest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from standin_eval.cache import ContentAddressedCache, pose_cache_key, vlm_cache_key
from standin_eval.compare import compare_runs, semantic_compare
from standin_eval.dataset import (
    init_dataset,
    load_dataset,
    seal_dataset,
    validate_dataset,
)
from standin_eval.fixtures import (
    ReplayPose, ReplayVLM, capture_pose_fixture, capture_vlm_fixture,
    serialize_skeleton,
)
from standin_eval.http_runner import run_http
from standin_eval.labels import create_label_pool, validate_pool_labels
from standin_eval.matching import MatchPolicy, match_people
from standin_eval.metrics import compute_run_metrics
from standin_eval.replay_runner import run_replay
from standin_eval.refine_runner import run_refine_pairs
from standin_eval.util import read_json, read_jsonl, utc_now, write_json, write_jsonl


def _png(path: Path, color=(255, 255, 255)) -> None:
    from PIL import Image

    Image.new("RGB", (100, 120), color=color).save(path, format="PNG")


def _dataset(directory: str, people: int = 1):
    root = Path(directory)
    images = root / "images"
    images.mkdir()
    first = images / "first.png"
    duplicate = images / "duplicate.png"
    _png(first)
    duplicate.write_bytes(first.read_bytes())
    output = init_dataset(
        "test-v1", [images], eval_root=root / "evaluation", purpose="engineering"
    )
    dataset = load_dataset(output)
    assert len(dataset.cuts) == 1
    cut = dataset.cuts[0]
    cut.update({
        "scene_group_id": "scene-1",
        "artist_id": "artist-1",
        "project_id": "project-1",
        "num_people_gt": people,
        "expected_route": "core",
    })
    persons = []
    for index in range(people):
        x1 = 5 + index * 45
        persons.append({
            "schema_version": 1,
            "person_id": f"{cut['cut_id']}:p{index + 1:02d}",
            "cut_id": cut["cut_id"],
            "bbox_xyxy": [x1, 10, x1 + 40, 110],
            "bbox_source": "manual-test",
            "eligible": True,
            "out_of_scope": False,
            "scale_class": "near",
            "difficulty": "hard" if index else "easy",
        })
    write_jsonl(output / "cuts.jsonl", [cut])
    write_jsonl(output / "persons.jsonl", persons)
    dataset = load_dataset(output)
    seal_dataset(dataset)
    return load_dataset(output)


def test_dataset_init_deduplicates_and_seals_gt():
    with tempfile.TemporaryDirectory() as directory:
        dataset = _dataset(directory, people=2)
        assert dataset.manifest["counts"]["files"] == 2
        assert dataset.manifest["counts"]["unique_image_contents"] == 1
        assert dataset.manifest["counts"]["target_persons"] == 2
        assert not [issue for issue in validate_dataset(dataset) if issue.level == "error"]
        from standin_eval.dataset import dataset_stats

        stats = dataset_stats(dataset)
        assert stats["artists"] == stats["projects"] == stats["scene_groups"] == 1


def test_unresolved_dataset_stats_do_not_invent_groups():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        images = root / "images"
        images.mkdir()
        _png(images / "one.png")
        dataset = load_dataset(init_dataset(
            "unresolved-v1", [images], eval_root=root / "evaluation"
        ))
        from standin_eval.dataset import dataset_stats

        stats = dataset_stats(dataset)
        assert stats["artists"] == stats["projects"] == stats["scene_groups"] == 0


def test_semantic_compare_uses_exact_and_numeric_tolerance():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first, second = root / "first", root / "second"
        first.mkdir()
        second.mkdir()
        artifacts = {
            "cut_results.jsonl": [{"cut_id": "c", "status": "ok", "route": "core", "vlm_count": 1, "detector_count": 1, "count_confidence": "high", "people_count": 1, "latency_ms": 10}],
            "predictions.jsonl": [{"cut_id": "c", "prediction_id": "p", "person_index": 0, "box_xyxy": [0, 0, 10, 10], "skeleton_state": "valid", "skeleton_source": "full", "coverage_class": "full", "valid_joint_count": 17, "confidence": "high", "candidate_count": 1}],
            "matches.jsonl": [{"cut_id": "c", "person_id": "g", "prediction_id": "p", "match_status": "matched", "expected_route": "core", "predicted_route": "core", "iou": 0.5, "normalized_center_distance": 0.1}],
            "candidates.jsonl": [{"cut_id": "c", "person_id": "g", "prediction_id": "p", "rank": 1, "pose_id": "pose", "view": "front", "family_id": "pose", "candidate_artifact_id": "sha256:x", "display_filter_status": "eligible", "surfaced": True, "distance": 0.2}],
        }
        for name, rows in artifacts.items():
            write_jsonl(first / name, rows)
            changed = [dict(row) for row in rows]
            if name == "cut_results.jsonl":
                changed[0]["latency_ms"] = 999
            if name == "candidates.jsonl":
                changed[0]["distance"] = 0.2000004
            write_jsonl(second / name, changed)
        assert semantic_compare(first, second, numeric_tolerance=1e-6)["status"] == "equal"
        rows = read_jsonl(second / "candidates.jsonl")
        rows[0]["pose_id"] = "regressed"
        write_jsonl(second / "candidates.jsonl", rows)
        result = semantic_compare(first, second, numeric_tolerance=1e-6)
        assert result["status"] == "different"
        assert result["artifacts"]["candidates.jsonl"]["field_mismatch_count"] == 1


def test_hungarian_matching_is_one_to_one_and_keeps_miss_rows():
    gt = [
        {"person_id": "left", "bbox_xyxy": [0, 0, 40, 100]},
        {"person_id": "right", "bbox_xyxy": [60, 0, 100, 100]},
        {"person_id": "missing", "bbox_xyxy": [120, 0, 160, 100]},
    ]
    predictions = [
        {"prediction_id": "right-pred", "box_xyxy": [62, 0, 99, 100]},
        {"prediction_id": "left-pred", "box_xyxy": [1, 0, 39, 100]},
    ]
    rows = match_people("cut", gt, predictions, "core", "core", MatchPolicy())
    matched = {row["person_id"]: row["prediction_id"] for row in rows if row["match_status"] == "matched"}
    assert matched == {"left": "left-pred", "right": "right-pred"}
    missed = [row for row in rows if row["match_status"] == "missed"]
    assert len(missed) == 1 and missed[0]["person_id"] == "missing"


def test_fixed_denominator_includes_missing_person():
    with tempfile.TemporaryDirectory() as directory:
        dataset = _dataset(directory, people=2)
        cut = dataset.cuts[0]
        first, second = dataset.persons
        cut_results = [{
            "cut_id": cut["cut_id"], "status": "ok", "route": "core",
            "vlm_count": 2, "latency_ms": 100, "within_time_budget": True,
        }]
        predictions = [{
            "cut_id": cut["cut_id"], "prediction_id": "pred-0",
            "skeleton_state": "valid", "coverage_class": "full",
        }]
        matches = [
            {"cut_id": cut["cut_id"], "person_id": first["person_id"], "prediction_id": "pred-0", "match_status": "matched"},
            {"cut_id": cut["cut_id"], "person_id": second["person_id"], "prediction_id": None, "match_status": "missed"},
        ]
        candidates = [{
            "cut_id": cut["cut_id"], "person_id": first["person_id"],
            "prediction_id": "pred-0", "rank": 1, "pose_id": "pose",
            "candidate_artifact_id": "sha256:one", "surfaced": True,
        }]
        key = (dataset.dataset_id, first["person_id"], "sha256:one", 1)
        labels = {key: {
            "dataset_id": dataset.dataset_id, "person_id": first["person_id"],
            "candidate_artifact_id": "sha256:one", "rubric_version": 1,
            "usefulness": "reference", "appearance": "allow",
        }}
        report = compute_run_metrics(
            dataset, cut_results, predictions, matches, candidates, labels
        )
        assert report["denominator"]["target_persons"] == 2
        assert report["search"]["candidate_coverage_at_5"]["numerator"] == 1
        assert report["search"]["candidate_coverage_at_5"]["denominator"] == 2
        assert report["diagnostics"]["failure_funnel"]["person_localization"] == 1


def test_content_addressed_cache_checks_payload():
    with tempfile.TemporaryDirectory() as directory:
        cache = ContentAddressedCache(directory)
        key = vlm_cache_key(
            image_sha256="image", provider="gemini", model="model",
            prompt_sha256="prompt", decoding={"temperature": 0},
            response_schema_version="1", preprocessing_version="1",
            sdk_version="1", sample_index=0,
        )
        cache.put("vlm", key, {"raw": "value"})
        result = cache.get("vlm", key)
        assert result.status == "success" and result.payload == {"raw": "value"}
        assert pose_cache_key(
            image_sha256="image", operation="crop", crop_pixel_sha256="crop",
            bbox_xyxy=[0, 0, 10, 10], padding=0.1, backend="rtmlib",
            weights_sha256="weights", preprocessing_version="1",
            runtime_provider="cpu", inference_parameters={},
        ) != pose_cache_key(
            image_sha256="image", operation="crop", crop_pixel_sha256="other",
            bbox_xyxy=[0, 0, 10, 10], padding=0.1, backend="rtmlib",
            weights_sha256="weights", preprocessing_version="1",
            runtime_provider="cpu", inference_parameters={},
        )


def test_replay_adapters_reproduce_normalized_inputs():
    import numpy as np
    from src.pose import MockPoseModel
    from src.schema import Action, Relationship, Shot, View
    from src.vlm.client import MockVLMClient

    vlm = MockVLMClient().analyze("full_half standing front 1p", 100, 120)
    payload = {
        "analysis": {
            "num_people": vlm.num_people,
            "shot": Shot.FULL_HALF.value,
            "action": Action.STANDING.value,
            "view": View.FRONT.value,
            "relationship": Relationship.SOLO.value,
            "approx_boxes": [{
                "x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2,
                "source": box.source, "score": box.score,
            } for box in vlm.approx_boxes],
            "dialogue": None, "raw": {},
        },
        "reranks": [],
    }
    replayed = ReplayVLM(payload).analyze(None, 100, 120)
    assert replayed.num_people == 1 and replayed.shot == Shot.FULL_HALF

    skeleton = MockPoseModel().estimate(None, vlm.approx_boxes, 100, 120)[0]
    pose = ReplayPose({
        "self_detecting": False,
        "calls": [{
            "operation": "full", "boxes": None, "img_w": 100, "img_h": 120,
            "outputs": [serialize_skeleton(skeleton)],
        }],
    })
    output = pose.estimate(None, vlm.approx_boxes, 100, 120)
    assert len(output) == 1 and np.allclose(output[0].keypoints, skeleton.keypoints)
    assert pose.unused_calls == 0



def _require_pose_db() -> str:
    """포즈 DB가 있는 환경에서만 도는 테스트임을 명시한다.

    data/는 Mixamo·CMU 원본 재배포 금지 정책으로 레포에 커밋하지 않으므로
    (.gitignore) CI 체크아웃에는 존재하지 않는다. 라이브러리를 내려받은
    로컬·평가 환경에서만 실제 capture/replay를 검증한다.
    """
    db_path = "data/poses.db"
    if not Path(db_path).exists():
        raise SkipTest(f"pose library DB unavailable: {db_path}")
    return db_path

def test_mock_fixture_capture_and_replay_end_to_end():
    _require_pose_db()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset = _dataset(directory, people=1)
        fixture = capture_vlm_fixture(
            dataset, fixture_id="fixture-test", cache_root=root / "cache",
            requested_provider="mock", model_cache_root=root / "model-cache",
        )
        capture_pose_fixture(
            dataset, fixture, db_path="data/poses.db", requested_backend="mock",
            model_cache_root=root / "model-cache",
        )
        run = run_replay(
            dataset, fixture=fixture, db_path="data/poses.db",
            output_root=root / "runs", run_id="replay-test", note="mock replay",
        )
        manifest = read_json(run / "manifest.json")
        assert manifest["mode"] == "replay" and manifest["counts"]["ok"] == 1
        assert read_jsonl(run / "candidates.jsonl")
        assert read_jsonl(run / "cut_results.jsonl")[0]["latency_kind"].startswith("replay")
        timing = read_jsonl(run / "timings.jsonl")[0]
        assert timing["kind"].startswith("replay")
        assert "pipeline_total" in timing["server_spans_ms"]


def test_fixture_capture_reuses_content_addressed_model_cache():
    _require_pose_db()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset = _dataset(directory, people=1)
        model_cache = root / "model-cache"
        first = capture_vlm_fixture(
            dataset, fixture_id="fixture-a", cache_root=root / "fixtures",
            requested_provider="mock", model_cache_root=model_cache,
        )
        second = capture_vlm_fixture(
            dataset, fixture_id="fixture-b", cache_root=root / "fixtures",
            requested_provider="mock", model_cache_root=model_cache,
            cache_miss="error",
        )
        assert read_json(first / "manifest.json")["vlm"]["cache"]["captures"] == 1
        assert read_json(second / "manifest.json")["vlm"]["cache"]["hits"] == 1
        capture_pose_fixture(
            dataset, first, db_path="data/poses.db", requested_backend="mock",
            model_cache_root=model_cache,
        )
        capture_pose_fixture(
            dataset, second, db_path="data/poses.db", requested_backend="mock",
            model_cache_root=model_cache, cache_miss="error",
        )
        assert read_json(second / "manifest.json")["pose"]["cache"]["hits"] == 1


def _write_fake_run(path: Path, dataset, run_id: str, candidate_artifact_id: str) -> None:
    path.mkdir(parents=True)
    write_json(path / "manifest.json", {
        "schema_version": 1, "metric_schema_version": 1, "run_id": run_id,
        "mode": "http", "dataset": {
            "dataset_id": dataset.dataset_id, "root": str(dataset.root),
            "cut_manifest_sha256": dataset.actual_cut_hash,
            "gt_sha256": dataset.actual_person_hash,
            "rubric_version": 1,
        },
        "artifacts": {"db": {"sha256": "db"}, "bvh": {"sha256": "bvh"},
                      "thumbnails": {"sha256": "thumb"}, "renderer_version": "v1"},
        "fixture_id": None,
    })
    cut, person = dataset.cuts[0], dataset.persons[0]
    write_jsonl(path / "candidates.jsonl", [{
        "cut_id": cut["cut_id"], "person_id": person["person_id"],
        "prediction_id": "pred", "rank": 1, "pose_id": f"hidden-{run_id}",
        "view": "front", "distance": 0.1,
        "candidate_artifact_id": candidate_artifact_id,
        "artifact_identity_complete": True,
        "thumbnail_local_path": "/tmp/preview.png",
        "surfaced": True,
    }])


def _complete_fake_run(path: Path, dataset) -> None:
    cut, person = dataset.cuts[0], dataset.persons[0]
    write_jsonl(path / "cut_results.jsonl", [{
        "cut_id": cut["cut_id"], "status": "ok", "route": "core",
        "vlm_count": 1, "detector_count": 1, "count_confidence": "high",
        "people_count": 1, "latency_ms": 10.0, "within_time_budget": True,
    }])
    write_jsonl(path / "predictions.jsonl", [{
        "cut_id": cut["cut_id"], "prediction_id": "pred", "person_index": 0,
        "box_xyxy": person["bbox_xyxy"], "skeleton_state": "valid",
        "skeleton_source": "full_image", "coverage_class": "full",
        "valid_joint_count": 17, "confidence": "high", "candidate_count": 1,
    }])
    write_jsonl(path / "matches.jsonl", [{
        "cut_id": cut["cut_id"], "person_id": person["person_id"],
        "prediction_id": "pred", "match_status": "matched", "iou": 1.0,
        "normalized_center_distance": 0.0,
    }])


def test_label_pool_hides_run_rank_pose_and_validates_completeness():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset = _dataset(directory, people=1)
        first, second = root / "run-a", root / "run-b"
        _write_fake_run(first, dataset, "run-a", "sha256:same")
        _write_fake_run(second, dataset, "run-b", "sha256:same")
        pool = create_label_pool(
            [first, second], output_root=root / "pools", pool_id="pool-test"
        )
        items = read_jsonl(pool / "items.jsonl")
        assert len(items) == 1
        assert not ({"run_id", "rank", "pose_id", "distance", "view"} & set(items[0]))
        provenance = read_jsonl(pool / "provenance.private.jsonl")
        assert len(provenance[0]["sources"]) == 2
        labels = read_jsonl(pool / "labels_template.jsonl")
        labels[0].update({
            "usefulness": "direct", "appearance": "allow", "labeler_id": "artist"
        })
        labels_path = root / "labels.jsonl"
        write_jsonl(labels_path, labels)
        result = validate_pool_labels(pool, labels_path)
        assert result["complete"]


def test_compare_runs_applies_labels_compatibility_and_semantic_diff():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset = _dataset(directory, people=1)
        first, second = root / "run-a", root / "run-b"
        _write_fake_run(first, dataset, "run-a", "sha256:same")
        _write_fake_run(second, dataset, "run-b", "sha256:same")
        for run in (first, second):
            rows = read_jsonl(run / "candidates.jsonl")
            rows[0]["pose_id"] = "same-pose"
            write_jsonl(run / "candidates.jsonl", rows)
        _complete_fake_run(first, dataset)
        _complete_fake_run(second, dataset)
        person = dataset.persons[0]
        labels = root / "labels.jsonl"
        write_jsonl(labels, [{
            "dataset_id": dataset.dataset_id, "person_id": person["person_id"],
            "candidate_artifact_id": "sha256:same", "rubric_version": 1,
            "usefulness": "reference", "appearance": "allow",
        }])
        output = compare_runs(
            first, second, labels_path=labels, output_root=root / "comparisons"
        )
        result = read_json(output / "comparison.json")
        assert result["status"] == "complete"
        assert result["semantic_replay"]["status"] == "equal"
        assert result["assist_success_at_5"]["equal"] == 1
        manifest = read_json(second / "manifest.json")
        manifest["config"] = {"surface_policy": "high_confidence"}
        write_json(second / "manifest.json", manifest)
        try:
            compare_runs(
                first, second, labels_path=labels,
                output_root=root / "incompatible-comparisons",
            )
            raise AssertionError("surface policy mismatch should be incomparable")
        except ValueError as exc:
            assert "surface_policy differs" in str(exc)


class _APIHandler(BaseHTTPRequestHandler):
    png = b"\x89PNG\r\n\x1a\n"

    def log_message(self, format, *args):
        pass

    def _json(self, value, status=200):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/healthz":
            return self._json({"ok": True})
        if self.path == "/openapi.json":
            return self._json({"paths": {"/analyze": {"post": {}}, "/healthz": {"get": {}}}})
        if self.path.startswith("/pose/pose/thumbnail"):
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(self.png)))
            self.end_headers()
            self.wfile.write(self.png)
            return
        return self._json({"error": "missing"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path != "/analyze":
            return self._json({"error": "missing"}, 404)
        return self._json({
            "route": "core", "count_confidence": "high",
            "detector_count": 1, "vlm_count": 1,
            "people": [{
                "index": 0, "box": [5, 10, 45, 110], "tags": {},
                "skeleton": {"schema_version": "coco17-v1", "keypoints": [[0, 0]] * 17, "scores": [0.9] * 17},
                "candidates": [{
                    "pose_id": "pose", "view": "front", "distance": 0.1,
                    "tags": {}, "bvh_url": "/pose/pose/bvh",
                    "thumbnail_url": "/pose/pose/thumbnail?view=front",
                }],
                "confidence": "high", "skeleton_state": "valid",
                "skeleton_source": "full_image", "coverage_class": "full",
                "quality_trace": {}, "quality_reasons": [],
            }],
            "notes": [], "image": {"width": 100, "height": 120},
            "inference_metadata": {
                "deployment_version": "test", "vlm_provider": "mock",
                "vlm_model": "mock", "pose_backend": "mock",
                "pose_model_version": "test", "pose_library_version": "test",
                "feature_version": 1,
            },
        })


def test_http_runner_writes_complete_artifact_shape_and_incomplete_report():
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as directory:
        dataset = _dataset(directory, people=1)
        payload = {
            "route": "core", "count_confidence": "high",
            "detector_count": 1, "vlm_count": 1,
            "people": [{
                "index": 0, "box": [5, 10, 45, 110], "tags": {},
                "skeleton": {"schema_version": "coco17-v1", "keypoints": [[0, 0]] * 17, "scores": [0.9] * 17},
                "candidates": [{
                    "pose_id": "pose", "view": "front", "distance": 0.1,
                    "tags": {}, "bvh_url": "/pose/pose/bvh",
                    "thumbnail_url": "/pose/pose/thumbnail?view=front",
                }],
                "confidence": "high", "skeleton_state": "valid",
                "skeleton_source": "full_image", "coverage_class": "full",
                "quality_trace": {}, "quality_reasons": [],
            }],
            "notes": [], "image": {"width": 100, "height": 120},
            "inference_metadata": {
                "deployment_version": "test", "vlm_provider": "mock",
                "vlm_model": "mock", "pose_backend": "mock",
                "pose_model_version": "test", "pose_library_version": "test",
                "feature_version": 1,
            },
        }

        def fake_request(url, **kwargs):
            if url.endswith("healthz"):
                return 200, b"{}", {"ok": True}
            if url.endswith("openapi.json"):
                return 200, b"{}", {"paths": {"/analyze": {"post": {}}}}
            return 200, json.dumps(payload).encode(), payload

        with patch("standin_eval.http_runner._request_json", side_effect=fake_request), patch(
            "standin_eval.http_runner._fetch_binary", return_value=(200, _APIHandler.png)
        ):
            run = run_http(
                dataset, target="http://standin.test", output_root=Path(directory) / "runs",
                run_id="http-test", requested_vlm="mock", requested_pose="mock",
                db_path=None, bvh_dir=None, thumbnail_dir=None,
            )
        assert read_json(run / "manifest.json")["counts"]["ok"] == 1
        assert len(read_jsonl(run / "cut_results.jsonl")) == 1
        assert len(read_jsonl(run / "matches.jsonl")) == 1
        assert len(read_jsonl(run / "candidates.jsonl")) == 1
        assert read_jsonl(run / "timings.jsonl")[0]["kind"].startswith("live")
        assert read_json(run / "report.json")["status"] == "incomplete"


def test_refine_pair_runner_checks_gated_fallback_identity():
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        (source / "responses").mkdir(parents=True)
        write_json(source / "manifest.json", {
            "run_id": "source-run", "mode": "http", "dataset": {"dataset_id": "d"},
            "artifacts": {}, "requested_backend": {}, "actual_backend": {},
        })
        write_jsonl(source / "candidates.jsonl", [{
            "cut_id": "cut", "person_id": "person", "prediction_id": "cut:pred:0",
            "rank": 1, "pose_id": "pose", "view": "front", "distance": 0.1,
            "bvh_url": "/pose/pose/bvh",
        }])
        write_json(source / "responses" / "cut.json", {"people": [{
            "refine_allowed": True, "refinable_limbs": ["left_arm"],
            "keypoints": [[0.0, 0.0]] * 17, "scores": [0.9] * 17,
        }]})
        with patch(
            "standin_eval.refine_runner._request_json",
            return_value=(200, b"{}", {
                "pose_id": "pose", "view": "front", "refined": False,
                "reason": "no_gain", "bvh_url": "/pose/pose/bvh",
                "loss_base": 0.1, "loss_final": 0.1, "gain": 0.0,
                "backend": "numpy", "limbs": [], "limb_decisions": {},
            }),
        ), patch(
            "standin_eval.refine_runner._fetch_binary", return_value=(200, b"same-bvh")
        ):
            run = run_refine_pairs(
                target="http://standin.test", from_run=source,
                output_root=root / "runs", run_id="refine-test",
            )
        pair = read_jsonl(run / "refine_pairs.jsonl")[0]
        assert pair["attempted"] and not pair["refined"]
        assert pair["fallback_identity_ok"] is True
        assert read_json(run / "manifest.json")["counts"]["fallback_identity_failures"] == 0


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
