"""Offline A/B1 shadow evaluator boundaries and scoring contracts."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from experiments.b1_pose_facts.shadow_eval import (
    CANDIDATE_LABELS,
    load_frozen_queries,
    run_shadow_evaluation,
    score_run,
)
from src.features import normalize_skeleton
from src.repo import build_db
from src.schema import Action, LibraryEntry, Relationship, Shot, View


_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "/x8AAusB9Wl2nLkAAAAASUVORK5CYII="
)
_VIEWS = (View.FRONT, View.THREE_QUARTER, View.SIDE, View.BACK)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_pose(*, knees_bent: bool = False, arm_bent: bool = False) -> np.ndarray:
    points = np.zeros((17, 2), dtype=np.float32)
    points[5], points[6] = (35, 35), (65, 35)
    points[7], points[8] = (25, 50), (75, 50)
    points[9], points[10] = (20, 70), (80, 70)
    points[11], points[12] = (40, 65), (60, 65)
    points[13], points[14] = (40, 90), (60, 90)
    points[15], points[16] = (40, 120), (60, 120)
    if knees_bent:
        points[13], points[14] = (30, 78), (70, 78)
        points[15], points[16] = (42, 82), (58, 82)
    if arm_bent:
        points[10] = (72, 38)
    return points


def _entries() -> list[LibraryEntry]:
    scores = np.ones(17, dtype=np.float32)
    poses = {
        "stand": _raw_pose(),
        "sit": _raw_pose(knees_bent=True),
        "gesture": _raw_pose(arm_bent=True),
    }
    rows: list[LibraryEntry] = []
    for pose_id, keypoints in poses.items():
        feature = normalize_skeleton(keypoints, scores)
        for view in _VIEWS:
            rows.append(LibraryEntry(
                pose_id=pose_id,
                view=view,
                feature=feature.copy(),
                tags={
                    "shot": Shot.FULL_HALF.value,
                    "action": Action.OTHER.value,
                    "relationship": Relationship.SOLO.value,
                },
                bvh_path=f"{pose_id}.bvh",
                meta={"source": "shadow-test", "license": "n/a"},
            ))
    return rows


def _write_query(directory: Path, *, unit_id: str = "cut-001",
                 keypoints: np.ndarray | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    image = directory / "cut.png"
    image.write_bytes(_PNG)
    points = _raw_pose(knees_bent=True) if keypoints is None else keypoints
    scores = np.ones(17, dtype=np.float32)
    payload = {
        "unit_id": unit_id,
        "image": str(image),
        "image_sha256": "sha256:" + hashlib.sha256(_PNG).hexdigest(),
        "keypoints": points.tolist(),
        "scores": scores.tolist(),
        "query_evidence": {
            "valid": True,
            "error": None,
            "score_threshold": 0.3,
            "target_valid_mask": [True] * 17,
        },
    }
    path = directory / "query.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_thumbnails(directory: Path) -> None:
    directory.mkdir(parents=True)
    for entry in _entries():
        (directory / f"{entry.pose_id}__{entry.view.value}.png").write_bytes(_PNG)


def _complete_answers(run: Path, *, candidate_label: str = "editable") -> Path:
    assert candidate_label in CANDIDATE_LABELS
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    mapping = json.loads(
        (run / "review/mapping.hidden.json").read_text(encoding="utf-8")
    )
    answers = {
        "schema_version": 1,
        "run_id": mapping["run_id"],
        "mapping_sha256": manifest["review"]["mapping_sha256"],
        "reviewer_id": "test-reviewer",
        "units": {
            unit["query_code"]: {
                "support_label": "non_stand",
                "candidates": {
                    code: candidate_label for code in unit["candidates"]
                },
            }
            for unit in mapping["units"]
        },
    }
    path = run.parent / "answers.json"
    path.write_text(json.dumps(answers), encoding="utf-8")
    return path


def test_frozen_query_deduplicates_identical_units_and_rejects_conflicts():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = _write_query(root / "one")
        second_dir = root / "two"
        second_dir.mkdir()
        second = second_dir / "query.json"
        second.write_bytes(first.read_bytes())
        queries = load_frozen_queries([first, second])
        assert len(queries) == 1
        assert len(queries[0].source_paths) == 2

        payload = json.loads(second.read_text(encoding="utf-8"))
        payload["keypoints"][16][0] += 3
        second.write_text(json.dumps(payload), encoding="utf-8")
        try:
            load_frozen_queries([first, second])
        except ValueError as exc:
            assert "conflicting frozen queries" in str(exc)
        else:
            raise AssertionError("conflicting duplicate unit was accepted")


def test_shadow_run_is_isolated_auditable_and_refuses_overwrite():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        db = root / "inputs/poses.db"
        build_db(_entries(), str(db))
        query = _write_query(root / "inputs/query")
        thumbs = root / "inputs/thumbs"
        _write_thumbnails(thumbs)
        output_root = root / "experiment"
        output_root.mkdir()
        output = output_root / "run-001"
        before_db = _sha256(db)
        before_query = _sha256(query)

        completed = run_shadow_evaluation(
            query_roots=[query.parent],
            db_path=db,
            thumbnail_root=thumbs,
            output_path=output,
            allowed_output_root=output_root,
            top_k=2,
        )
        assert completed == output.resolve()
        assert _sha256(db) == before_db
        assert _sha256(query) == before_query
        assert (output / "manifest.json").is_file()
        assert (output / "records.json").is_file()
        assert (output / "summary.json").is_file()
        assert (output / "REPORT.md").is_file()
        assert (output / "review/index.html").is_file()
        assert (output / "review/mapping.hidden.json").is_file()

        public_html = (output / "review/index.html").read_text(encoding="utf-8")
        assert "pose_id" not in public_html
        assert '"conditions"' not in public_html
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["input"]["evaluated_units"] == 1
        assert all(summary["invariants"].values())

        try:
            run_shadow_evaluation(
                query_roots=[query], db_path=db, thumbnail_root=thumbs,
                output_path=output, allowed_output_root=output_root, top_k=2,
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing run directory was overwritten")

        try:
            run_shadow_evaluation(
                query_roots=[query], db_path=db, thumbnail_root=thumbs,
                output_path=root / "outside", allowed_output_root=output_root,
                top_k=2,
            )
        except ValueError as exc:
            assert "escapes isolated experiment root" in str(exc)
        else:
            raise AssertionError("output boundary escape was accepted")


def test_complete_blind_answers_score_and_incomplete_answers_fail_closed():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        db = root / "inputs/poses.db"
        build_db(_entries(), str(db))
        query = _write_query(root / "inputs/query")
        thumbs = root / "inputs/thumbs"
        _write_thumbnails(thumbs)
        output_root = root / "experiment"
        output_root.mkdir()
        run = run_shadow_evaluation(
            query_roots=[query], db_path=db, thumbnail_root=thumbs,
            output_path=output_root / "run-001",
            allowed_output_root=output_root, top_k=2,
        )
        answers = _complete_answers(run)
        result = score_run(
            run_path=run, answers_path=answers,
            allowed_output_root=output_root,
        )
        assert result["conditions"]["baseline"]["top1_workable_rate"] == 1.0
        assert result["promotion_checks"]["a_minimal_support"]["status"] == "INSUFFICIENT"
        assert result["promotion_checks"]["b1_pose_fact_rerank"]["status"] == "INSUFFICIENT"
        score_output = Path(result["output_directory"])
        assert (score_output / "score_records.json").is_file()
        assert (score_output / "score_summary.json").is_file()
        assert (score_output / "SCORE_REPORT.md").is_file()

        try:
            score_run(
                run_path=run, answers_path=answers,
                allowed_output_root=output_root,
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("an existing answer score was overwritten")

        incomplete = json.loads(answers.read_text(encoding="utf-8"))
        incomplete["units"] = {}
        incomplete_path = root / "incomplete.json"
        incomplete_path.write_text(json.dumps(incomplete), encoding="utf-8")
        try:
            score_run(
                run_path=run, answers_path=incomplete_path,
                allowed_output_root=output_root,
            )
        except ValueError as exc:
            assert "answers unit set mismatch" in str(exc)
        else:
            raise AssertionError("incomplete blind answers were scored")


if __name__ == "__main__":
    import traceback

    functions = [value for name, value in sorted(globals().items())
                 if name.startswith("test_") and callable(value)]
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
