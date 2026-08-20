#!/usr/bin/env python3
"""Build an immutable staging intake from ``data/pose-dedup-v3``.

The builder never mutates ``data/bvh``.  It cross-deduplicates incoming
single-frame BVHs against the active library, removes the already approved CMU
geometry exclusions from the staging baseline, creates verified mirror pairs,
and writes a provenance manifest plus a complete candidate BVH directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any
import warnings

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.mirror_bvh import mirror_bvh  # noqa: E402
from src.bvh import fk, load_coco17, parse_bvh  # noqa: E402


SCHEMA_VERSION = 1
DEFAULT_BATCH_ID = "pose-dedup-v3-20260818"
DEFAULT_THRESHOLD_DEGREES = 15.0

BONES = [
    ("Hips", "Spine"),
    ("Spine", "Spine1"),
    ("Spine1", "Neck"),
    ("Neck", "Head"),
    ("Spine1", "LeftShoulder"),
    ("LeftShoulder", "LeftArm"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("Spine1", "RightShoulder"),
    ("RightShoulder", "RightArm"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    ("Hips", "LeftUpLeg"),
    ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFoot"),
    ("LeftFoot", "LeftToeBase"),
    ("Hips", "RightUpLeg"),
    ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFoot"),
    ("RightFoot", "RightToeBase"),
]


def _swap_side(name: str) -> str:
    if name.startswith("Left"):
        return name.replace("Left", "Right", 1)
    if name.startswith("Right"):
        return name.replace("Right", "Left", 1)
    return name


MIRROR_INDICES = [BONES.index((_swap_side(a), _swap_side(b))) for a, b in BONES]
REQUIRED_JOINTS = {name for bone in BONES for name in bone}
SAMPLE_PATTERN = re.compile(r"__(?P<kind>[pn])(?P<ordinal>\d+)_f(?P<frame>\d+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _descriptor(path: Path) -> np.ndarray:
    joints, frames = parse_bvh(str(path))
    if len(frames) != 1:
        raise ValueError(f"single-frame BVH required: {path} has {len(frames)}")
    positions = fk(joints, frames[0])
    points = {
        joint[0].split(":")[-1]: np.asarray(positions[index], dtype=float)
        for index, joint in enumerate(joints)
    }
    missing = sorted(REQUIRED_JOINTS - set(points))
    if missing:
        raise ValueError(f"canonical joints missing in {path}: {missing}")

    lateral = points["LeftUpLeg"] - points["RightUpLeg"]
    lateral_xz = np.array([lateral[0], 0.0, lateral[2]])
    norm = float(np.linalg.norm(lateral_xz))
    if norm < 1e-9:
        lateral = points["LeftShoulder"] - points["RightShoulder"]
        lateral_xz = np.array([lateral[0], 0.0, lateral[2]])
        norm = float(np.linalg.norm(lateral_xz))
    if norm < 1e-9:
        raise ValueError(f"lateral axis is degenerate: {path}")
    cosine, sine = lateral_xz[0] / norm, lateral_xz[2] / norm
    canonical_yaw = np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
    )

    directions = []
    for start, end in BONES:
        vector = points[end] - points[start]
        length = float(np.linalg.norm(vector))
        if length < 1e-9:
            # 기존 CMU 축약 리그에는 Spine1/Neck 등이 같은 위치인 파일이 있다.
            # pose-dedup-v3를 만든 기존 metric과 동일하게 해당 뼈만 영벡터로 둔다.
            directions.append(np.zeros(3, dtype=float))
        else:
            directions.append((canonical_yaw @ vector) / length)
    return np.asarray(directions, dtype=np.float64)


def _mirrored(descriptor: np.ndarray) -> np.ndarray:
    output = descriptor[MIRROR_INDICES].copy()
    output[:, 0] *= -1.0
    return output


def _angular_distance(left: np.ndarray, right: np.ndarray) -> float:
    dot = np.clip(np.sum(left * right, axis=1), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)).mean())


def _nearest(
    descriptor: np.ndarray,
    pool: list[tuple[Path, np.ndarray, np.ndarray]],
) -> tuple[float, str]:
    return min(
        (
            min(
                _angular_distance(descriptor, candidate),
                _angular_distance(descriptor, candidate_mirror),
            ),
            path.name,
        )
        for path, candidate, candidate_mirror in pool
    )


def _excluded_pose_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    output: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            output.update(json.loads(row["member_pose_ids"]))
    return output


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_record(
    pose: dict[str, Any],
    *,
    makehuman: dict[str, dict[str, Any]],
    ual1_source: dict[str, Any],
    ual2_source: dict[str, Any],
    g1_files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source = pose["src"]
    clip = pose["clip"]
    sample = SAMPLE_PATTERN.search(pose["orig_stem"])
    sample_ordinal = int(sample.group("ordinal")) if sample else None
    selected_frame = int(sample.group("frame")) if sample else None
    selection_kind = sample.group("kind") if sample else None

    if source == "UAL1":
        original_path = Path(ual1_source["source_fbx"])
        return {
            "provider": "quaternius",
            "collection_id": "universal_animation_library_1_standard",
            "source_clip_id": f"ual1:{clip}",
            "native_clip_id": clip,
            "original_title": clip,
            "original_filename": original_path.name,
            "original_path": original_path.as_posix(),
            "derived_artifact_path": pose["origin"],
            "source_url": None,
            "source_sha256": f"sha256:{ual1_source['source_sha256']}",
            "license_id": "CC0-1.0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "product_bvh_export": "yes",
            "delivery_mode": "original",
            "attribution_required": False,
            "sample_ordinal": sample_ordinal,
            "selected_frame_index": selected_frame,
            "frame_index_base": "one",
            "selection_kind": selection_kind,
        }
    if source == "UAL2":
        original_path = Path(ual2_source["input_dir"]) / f"{clip}.fbx"
        return {
            "provider": "quaternius",
            "collection_id": "universal_animation_library_2_standard",
            "source_clip_id": f"ual2:{clip}",
            "native_clip_id": clip,
            "original_title": clip,
            "original_filename": original_path.name,
            "original_path": original_path.as_posix(),
            "derived_artifact_path": pose["origin"],
            "source_url": None,
            "source_sha256": _sha256(original_path),
            "license_id": "CC0-1.0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "product_bvh_export": "yes",
            "delivery_mode": "original",
            "attribution_required": False,
            "sample_ordinal": sample_ordinal,
            "selected_frame_index": selected_frame,
            "frame_index_base": "one",
            "selection_kind": selection_kind,
        }
    if source == "g1":
        source_file = g1_files[clip]
        original_path = Path(source_file["source_path"])
        return {
            "provider": "g1_moves",
            "collection_id": "g1_moves",
            "source_clip_id": f"g1:{clip}",
            "native_clip_id": clip,
            "original_title": clip,
            "original_filename": source_file["source_file"],
            "original_path": original_path.as_posix(),
            "derived_artifact_path": pose["origin"],
            "source_url": None,
            "source_sha256": f"sha256:{source_file['source_sha256']}",
            "fps": source_file.get("source_fps"),
            "license_id": "redistribution-confirmed-2026-08",
            "license_url": None,
            "product_bvh_export": "yes",
            "delivery_mode": "original",
            "attribution_required": False,
            "sample_ordinal": sample_ordinal,
            "selected_frame_index": selected_frame,
            "frame_index_base": "zero",
            "selection_kind": selection_kind,
        }
    if source == "MH":
        item = makehuman[pose["orig_stem"]]
        node = item.get("source_url", "").rstrip("/").split("/")[-1] or None
        return {
            "provider": "makehuman_community",
            "collection_id": "makehuman_poses_01",
            "source_clip_id": f"makehuman:{pose['orig_stem']}",
            "native_clip_id": node or pose["orig_stem"],
            "original_title": pose["orig_stem"],
            "original_filename": Path(item["output"]).name,
            "original_path": pose["origin"],
            "derived_artifact_path": pose["origin"],
            "source_url": item.get("source_url"),
            "source_sha256": (
                f"sha256:{item['source_sha256']}" if item.get("source_sha256") else None
            ),
            "license_id": item["license"],
            "license_url": (
                "https://creativecommons.org/licenses/by/4.0/"
                if item["license"] == "CC-BY-4.0"
                else "https://creativecommons.org/publicdomain/zero/1.0/"
            ),
            "product_bvh_export": "yes",
            "delivery_mode": "original",
            "attribution_required": item["license"] == "CC-BY-4.0",
            "author": item.get("author"),
            "sample_ordinal": None,
            "selected_frame_index": None,
            "frame_index_base": None,
            "selection_kind": "static_asset",
        }
    raise ValueError(f"unsupported source prefix: {source}")


def _member_record(
    *,
    path: Path,
    manifest_path: Path,
    original_pose_id: str,
    variant: str,
    source: dict[str, Any],
    nearest_distance: float,
    source_bundle_path: str,
    threshold_degrees: float,
) -> dict[str, Any]:
    pose_id = path.stem
    return {
        "record_type": "pose_intake_member",
        "schema_version": SCHEMA_VERSION,
        "pose_id": pose_id,
        "bvh": {
            "path": _repo_path(manifest_path),
            "sha256": _sha256(path),
            "frame_count": 1,
        },
        "source": source,
        "extraction": {
            "sample_ordinal": source.get("sample_ordinal"),
            "selected_frame_index": source.get("selected_frame_index"),
            "frame_index_base": source.get("frame_index_base"),
            "selection_kind": source.get("selection_kind"),
            "bundle_path": source_bundle_path,
        },
        "grouping": {
            "mirror_group_id": f"mirror:{original_pose_id}",
            "semantic_unit_id": f"pose:{original_pose_id}",
            "pose_family_id": source["source_clip_id"],
            "variant_kind": variant,
            "mirror_of": original_pose_id if variant == "mirrored" else None,
        },
        "dedup": {
            "metric": "20 canonical bone unit-direction mean angular distance",
            "nearest_active_degrees": round(nearest_distance, 6),
            "threshold_degrees": threshold_degrees,
        },
        "derivation": {
            "operations": (["mirror_x"] if variant == "mirrored" else []),
            "parent_pose_id": original_pose_id if variant == "mirrored" else None,
        },
        "eligibility": {
            "geometry_index": True,
            "semantic_index": True,
            "release": source["product_bvh_export"] == "yes",
        },
    }


def _validate_bvh(path: Path) -> None:
    joints, frames = parse_bvh(str(path))
    if not joints or len(frames) != 1:
        raise ValueError(f"invalid single-frame BVH: {path}")
    keypoints, scores = load_coco17(str(path))
    if not np.isfinite(keypoints).all() or not np.isfinite(scores).all():
        raise ValueError(f"non-finite COCO mapping: {path}")
    if np.any(scores[5:17] < 0.3):
        raise ValueError(f"incomplete COCO body mapping: {path}")


def build(args: argparse.Namespace) -> dict[str, Any]:
    incoming_dir = args.incoming_dir.resolve()
    incoming_bvh_dir = incoming_dir / "bvh"
    active_dir = args.active_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable intake: {output_dir}")

    package = _load_json(incoming_dir / "manifest.json")
    package_rows = {row["file"]: row for row in package["poses"]}
    incoming_paths = sorted(incoming_bvh_dir.glob("*.bvh"))
    if set(package_rows) != {path.name for path in incoming_paths}:
        raise ValueError("incoming BVHs and package manifest rows do not match")

    excluded_ids = _excluded_pose_ids(args.exclusions_csv)
    all_active_paths = sorted(active_dir.glob("*.bvh"))
    active_paths = sorted(
        path for path in all_active_paths if path.stem not in excluded_ids
    )
    removed_ids = {path.stem for path in all_active_paths} - {
        path.stem for path in active_paths
    }
    if removed_ids != excluded_ids:
        missing = sorted(excluded_ids - removed_ids)
        unexpected = sorted(removed_ids - excluded_ids)
        raise ValueError(
            f"geometry exclusion mismatch: missing={missing}, unexpected={unexpected}"
        )
    active_pool = []
    for path in active_paths:
        descriptor = _descriptor(path)
        active_pool.append((path, descriptor, _mirrored(descriptor)))

    incoming_descriptors: dict[str, np.ndarray] = {}
    for path in incoming_paths:
        _validate_bvh(path)
        incoming_descriptors[path.name] = _descriptor(path)

    internal_minimum = min(
        min(
            _angular_distance(incoming_descriptors[left.name], incoming_descriptors[right.name]),
            _angular_distance(incoming_descriptors[left.name], _mirrored(incoming_descriptors[right.name])),
        )
        for index, left in enumerate(incoming_paths)
        for right in incoming_paths[index + 1 :]
    )
    decisions = []
    for path in incoming_paths:
        distance, nearest_name = _nearest(incoming_descriptors[path.name], active_pool)
        decisions.append(
            {
                "file": path.name,
                "source": package_rows[path.name]["src"],
                "source_clip": package_rows[path.name]["clip"],
                "nearest_active_file": nearest_name,
                "nearest_active_degrees": distance,
                "nearest_incoming_file": None,
                "nearest_incoming_degrees": None,
                "decision": (
                    "pending_internal_dedup"
                    if distance >= args.threshold_degrees
                    else "rejected_cross_library_duplicate"
                ),
            }
        )

    accepted_pool: list[dict[str, Any]] = []
    for decision in sorted(
        (row for row in decisions if row["decision"] == "pending_internal_dedup"),
        key=lambda row: (-row["nearest_active_degrees"], row["file"]),
    ):
        descriptor = incoming_descriptors[decision["file"]]
        nearest_internal: tuple[float, str] | None = None
        for kept in accepted_pool:
            kept_descriptor = incoming_descriptors[kept["file"]]
            distance = min(
                _angular_distance(descriptor, kept_descriptor),
                _angular_distance(descriptor, _mirrored(kept_descriptor)),
            )
            if nearest_internal is None or distance < nearest_internal[0]:
                nearest_internal = (distance, kept["file"])
        if nearest_internal and nearest_internal[0] < args.threshold_degrees:
            decision["nearest_incoming_degrees"] = nearest_internal[0]
            decision["nearest_incoming_file"] = nearest_internal[1]
            decision["decision"] = "rejected_internal_duplicate"
        else:
            decision["decision"] = "accepted"
            accepted_pool.append(decision)

    accepted = [row for row in decisions if row["decision"] == "accepted"]
    rejected = [row for row in decisions if row["decision"] != "accepted"]
    rejected_cross = [
        row for row in rejected if row["decision"] == "rejected_cross_library_duplicate"
    ]
    rejected_internal = [
        row for row in rejected if row["decision"] == "rejected_internal_duplicate"
    ]
    makehuman_manifest = _load_json(args.makehuman_manifest)
    makehuman = {row["id"]: row for row in makehuman_manifest["included"]}
    ual1_manifest = _load_json(args.ual1_manifest)
    ual2_manifest = _load_json(args.ual2_manifest)
    g1_manifest = _load_json(args.g1_manifest)
    g1_input_dir = Path(g1_manifest["source"]["input_dir"])
    g1_files = {
        Path(row["source_file"]).stem: {
            **row,
            "source_path": (g1_input_dir / row["source_file"]).as_posix(),
        }
        for row in g1_manifest["files"]
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=str(output_dir.parent))
    )
    try:
        originals_dir = temporary / "originals"
        accepted_dir = temporary / "accepted"
        candidate_dir = temporary / "candidate_bvh"
        reports_dir = temporary / "reports"
        attribution_dir = temporary / "attribution"
        for directory in (originals_dir, accepted_dir, candidate_dir, reports_dir):
            directory.mkdir(parents=True, exist_ok=True)

        for path in active_paths:
            shutil.copy2(path, candidate_dir / path.name)

        member_records = []
        mirror_roundtrip_max = 0.0
        attribution_required = False
        for decision in accepted:
            source_path = incoming_bvh_dir / decision["file"]
            original_path = originals_dir / source_path.name
            accepted_original = accepted_dir / source_path.name
            shutil.copy2(source_path, original_path)
            shutil.copy2(source_path, accepted_original)

            mirror_path = accepted_dir / f"{source_path.stem}_mirror.bvh"
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Gimbal lock detected")
                mirror_result = mirror_bvh(accepted_original, mirror_path)
            mirror_roundtrip_max = max(
                mirror_roundtrip_max,
                float(mirror_result["roundtrip_coco_max_error"]),
            )
            _validate_bvh(mirror_path)
            shutil.copy2(accepted_original, candidate_dir / accepted_original.name)
            shutil.copy2(mirror_path, candidate_dir / mirror_path.name)

            package_pose = package_rows[source_path.name]
            source = _source_record(
                package_pose,
                makehuman=makehuman,
                ual1_source=ual1_manifest["source"],
                ual2_source=ual2_manifest["source"],
                g1_files=g1_files,
            )
            attribution_required = attribution_required or bool(
                source["attribution_required"]
            )
            source_bundle_path = _repo_path(source_path)
            member_records.extend(
                [
                    _member_record(
                        path=accepted_original,
                        manifest_path=output_dir / "accepted" / accepted_original.name,
                        original_pose_id=source_path.stem,
                        variant="original",
                        source=source,
                        nearest_distance=decision["nearest_active_degrees"],
                        source_bundle_path=source_bundle_path,
                        threshold_degrees=args.threshold_degrees,
                    ),
                    _member_record(
                        path=mirror_path,
                        manifest_path=output_dir / "accepted" / mirror_path.name,
                        original_pose_id=source_path.stem,
                        variant="mirrored",
                        source=source,
                        nearest_distance=decision["nearest_active_degrees"],
                        source_bundle_path=source_bundle_path,
                        threshold_degrees=args.threshold_degrees,
                    ),
                ]
            )

        if attribution_required:
            attribution_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                args.makehuman_attribution,
                attribution_dir / "MakeHuman_ATTRIBUTION.md",
            )

        for path in sorted(candidate_dir.glob("*.bvh")):
            _validate_bvh(path)

        header = {
            "record_type": "intake_header",
            "schema_version": SCHEMA_VERSION,
            "batch_id": args.batch_id,
            "source_bundle": _repo_path(incoming_dir),
            "source_bundle_manifest_sha256": _sha256(incoming_dir / "manifest.json"),
            "policy": {
                "dedup_threshold_degrees": args.threshold_degrees,
                "dedup_metric": "20 canonical bone unit-direction mean angular distance",
                "mirror_aware": True,
                "active_geometry_exclusions_applied": True,
            },
            "counts": {
                "incoming_originals": len(incoming_paths),
                "rejected_cross_library_duplicates": len(rejected_cross),
                "rejected_internal_duplicates": len(rejected_internal),
                "rejected_originals_total": len(rejected),
                "accepted_originals": len(accepted),
                "generated_mirrors": len(accepted),
                "new_pose_members": len(member_records),
                "baseline_pose_members": len(active_paths),
                "baseline_excluded_pose_members": len(excluded_ids),
                "candidate_pose_members": len(list(candidate_dir.glob("*.bvh"))),
            },
            "validation": {
                "incoming_internal_minimum_degrees": round(internal_minimum, 6),
                "mirror_roundtrip_coco_max_error": mirror_roundtrip_max,
                "all_candidate_bvhs_single_frame_and_coco17_valid": True,
            },
        }
        with (temporary / "manifest.v1.jsonl").open("w", encoding="utf-8") as stream:
            for row in [header, *sorted(member_records, key=lambda item: item["pose_id"])]:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

        audit = {
            **header,
            "record_type": "intake_audit",
            "accepted_by_source": {
                source: sum(row["source"] == source for row in accepted)
                for source in sorted({row["source"] for row in decisions})
            },
            "rejected_by_source": {
                source: sum(row["source"] == source for row in rejected)
                for source in sorted({row["source"] for row in decisions})
            },
            "rejected_duplicates": sorted(
                rejected, key=lambda row: (row["nearest_active_degrees"], row["file"])
            ),
        }
        (reports_dir / "intake-audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with (reports_dir / "cross-library-dedup.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(decisions[0]))
            writer.writeheader()
            writer.writerows(
                sorted(decisions, key=lambda row: (row["nearest_active_degrees"], row["file"]))
            )

        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return _load_json(output_dir / "reports/intake-audit.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument("--incoming-dir", type=Path, default=Path("data/pose-dedup-v3"))
    parser.add_argument("--active-dir", type=Path, default=Path("data/bvh"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/library_work") / DEFAULT_BATCH_ID,
    )
    parser.add_argument(
        "--exclusions-csv",
        type=Path,
        default=Path("data/semantic/excluded_source_clips.csv"),
    )
    parser.add_argument(
        "--makehuman-manifest",
        type=Path,
        default=Path("data/makehuman-sitting-poses01/manifest.json"),
    )
    parser.add_argument(
        "--ual1-manifest",
        type=Path,
        default=Path("data/ual1-keyposes/manifest.json"),
    )
    parser.add_argument(
        "--ual2-manifest",
        type=Path,
        default=Path("data/ual2_bvh_candidates/manifest.json"),
    )
    parser.add_argument(
        "--g1-manifest",
        type=Path,
        default=Path("data/g1-moves/manifest.json"),
    )
    parser.add_argument(
        "--makehuman-attribution",
        type=Path,
        default=Path("data/makehuman-sitting-poses01/ATTRIBUTION.md"),
    )
    parser.add_argument("--threshold-degrees", type=float, default=DEFAULT_THRESHOLD_DEGREES)
    return parser.parse_args()


def main() -> int:
    result = build(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
