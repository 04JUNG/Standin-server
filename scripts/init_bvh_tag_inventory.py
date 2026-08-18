"""BVH 의미 태깅 전의 결정적 인벤토리 JSONL을 만든다.

이 스크립트는 자세의 의미를 확정하지 않는다. 파일·BVH 무결성, 원본/미러 관계,
파일명에서 읽을 수 있는 *제안 힌트*만 기록한다. 출처와 라이선스는 외부 원장으로
확인하기 전까지 미확정 상태를 유지한다.

실행:
    python scripts/init_bvh_tag_inventory.py
    python scripts/init_bvh_tag_inventory.py \
        --bvh-dir data/bvh \
        --output data/semantic/inventory.v1.jsonl

출력 파일은 생성 산출물이다. 사람이 직접 수정하지 않고, 의미 태그 제안과 검수 결정은
각각 proposals/decisions JSONL에 append-only로 남긴다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from collections import Counter


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.bvh import load_coco17, parse_bvh  # noqa: E402
from src.schema import COCO17  # noqa: E402


SCHEMA_VERSION = 1
INVENTORY_VERSION = 1
FILENAME_PARSER_VERSION = 1
BODY_INDICES = tuple(range(5, 17))
DERIVATION_SUFFIXES = ("_ground", "_legfix", "_legstraight")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _without_mirror(stem: str) -> tuple[str, bool]:
    if stem.endswith("_mirror"):
        return stem[: -len("_mirror")], True
    return stem, False


def _split_words(value: str) -> str:
    value = value.replace("_", " ")
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return " ".join(value.split())


def _filename_evidence(stem: str) -> dict:
    """파일명은 정답이 아니라 provenance 복구와 의미 태깅의 seed만 만든다."""
    base, _ = _without_mirror(stem)
    seed_base = base
    derivation_hints = []
    changed = True
    while changed:
        changed = False
        for suffix in DERIVATION_SUFFIXES:
            if seed_base.endswith(suffix):
                derivation_hints.append(suffix[1:])
                seed_base = seed_base[: -len(suffix)]
                changed = True

    pattern = "opaque"
    label_hint = None
    collection_hint = None
    provider_hint = None
    native_subject_id_hint = None
    native_clip_id_hint = None
    source_clip_id_hint = None
    source_frame_hint = None
    sample_ordinal_hint = None
    retarget_profile_hint = None

    match = re.fullmatch(r"cmu_(\d+)_(\d+)_(\d+)", seed_base)
    if match:
        subject, trial, frame = match.groups()
        pattern = "cmu_subject_trial_frame"
        collection_hint = "cmu"
        provider_hint = "cmu"
        native_subject_id_hint = subject
        native_clip_id_hint = trial
        source_clip_id_hint = f"filename_hint:cmu:{subject}:{trial}"
        source_frame_hint = int(frame)
    else:
        match = re.fullmatch(r"rokoko_(.+)_mixamo_(\d+)", seed_base)
        if match:
            clip, frame = match.groups()
            pattern = "rokoko_clip_frame"
            collection_hint = "rokoko"
            provider_hint = "rokoko"
            label_hint = _split_words(clip)
            source_clip_id_hint = f"filename_hint:rokoko:{clip}"
            source_frame_hint = int(frame)
            retarget_profile_hint = "mixamorig"
        else:
            match = re.fullmatch(r"(.+?)_(\d{5})", seed_base)
            if match:
                label, frame = match.groups()
                pattern = "named_clip_frame"
                label_hint = _split_words(label)
                source_clip_id_hint = f"filename_hint:named:{label}"
                source_frame_hint = int(frame)
            else:
                match = re.fullmatch(r"(.+?)_(\d{2})", seed_base)
                if match:
                    label, sample = match.groups()
                    pattern = "named_clip_sample"
                    label_hint = _split_words(label)
                    source_clip_id_hint = f"filename_hint:named:{label}"
                    source_frame_hint = None
                    sample_ordinal_hint = int(sample)
                elif seed_base:
                    pattern = "named_unversioned"
                    label_hint = _split_words(seed_base)

    return {
        "parser_version": FILENAME_PARSER_VERSION,
        "raw_stem": stem,
        "pattern": pattern,
        "label_en_hint": label_hint,
        "original_title_hint": label_hint,
        "source_collection_hint": collection_hint,
        "provider_hint": provider_hint,
        "native_subject_id_hint": native_subject_id_hint,
        "native_clip_id_hint": native_clip_id_hint,
        "source_clip_id_hint": source_clip_id_hint,
        "selected_frame_index_hint": source_frame_hint,
        "frame_index_base_hint": "unknown" if source_frame_hint is not None else None,
        "sample_ordinal_hint": sample_ordinal_hint,
        "retarget_profile_hint": retarget_profile_hint,
        "derivation_hints": sorted(derivation_hints),
        "evidence_status": "proposed",
    }


def _inspect_bvh(path: Path) -> tuple[dict, dict]:
    errors: list[str] = []
    warnings: list[str] = []
    metadata = {
        "frame_count": None,
        "joint_count": None,
        "root_joint": None,
        "joint_name_profile": None,
    }
    parse_status = "failed"
    mapping_status = "not_run"

    try:
        joints, frames = parse_bvh(str(path))
        parse_status = "pass"
        metadata.update(
            {
                "frame_count": int(len(frames)),
                "joint_count": int(len(joints)),
                "root_joint": joints[0][0] if joints else None,
                "joint_name_profile": (
                    "namespaced"
                    if any(":" in joint[0] for joint in joints)
                    else "unprefixed"
                ),
            }
        )
        if len(frames) != 1:
            warnings.append("single_frame_required")
    except Exception as exc:  # 손상 파일 하나 때문에 전체 inventory를 잃지 않는다.
        errors.append(f"parse_error:{type(exc).__name__}:{exc}")

    if parse_status == "pass":
        try:
            _, scores = load_coco17(str(path), frame=0)
            missing = [COCO17[index] for index in BODY_INDICES if scores[index] < 0.3]
            if missing:
                mapping_status = "failed"
                errors.append("missing_body_joints:" + ",".join(missing))
            else:
                mapping_status = "pass"
        except Exception as exc:
            mapping_status = "failed"
            errors.append(f"mapping_error:{type(exc).__name__}:{exc}")

    return metadata, {
        "parse_status": parse_status,
        "mapping_status": mapping_status,
        "warnings": warnings,
        "errors": errors,
    }


def _member_record(path: Path, all_pose_ids: set[str]) -> dict:
    pose_id = path.stem
    canonical_id, mirrored = _without_mirror(pose_id)
    expected_original = canonical_id if mirrored else None
    expected_mirror = f"{canonical_id}_mirror"
    mirror_exists = (
        expected_original in all_pose_ids if mirrored else expected_mirror in all_pose_ids
    )

    bvh_sha256 = _sha256(path)
    bvh_metadata, validation = _inspect_bvh(path)
    filename_evidence = _filename_evidence(pose_id)
    warnings = validation["warnings"]

    if mirrored and not mirror_exists:
        warnings.append("orphan_mirror")
    if filename_evidence["label_en_hint"] is None:
        warnings.append("opaque_filename")
    warnings.extend(
        ["source_unverified", "license_unresolved", "semantic_annotation_missing"]
    )

    geometry_eligible = (
        validation["parse_status"] == "pass"
        and validation["mapping_status"] == "pass"
        and bvh_metadata["frame_count"] == 1
    )
    blockers = ["source_unverified", "license_unresolved", "semantic_annotation_not_accepted"]
    if not geometry_eligible:
        blockers.insert(0, "bvh_geometry_validation_failed")
    if mirrored and not mirror_exists:
        blockers.append("orphan_mirror_review_required")

    return {
        "record_type": "pose_member_inventory",
        "schema_version": SCHEMA_VERSION,
        "pose_id": pose_id,
        "bvh": {
            "relative_path": _display_path(path),
            "sha256": bvh_sha256,
            **bvh_metadata,
        },
        "provenance_refs": {
            "source_clip_id": None,
            "pose_lineage_id": f"lineage:{pose_id}",
            "verification_status": "unverified",
        },
        "grouping": {
            "mirror_group_id": f"mirror:{canonical_id}",
            "semantic_unit_id": f"pose:{canonical_id}",
            "pose_family_id": None,
            "variant": {
                "kind": "mirrored" if mirrored else "original",
                "mirror_of": expected_original,
                "paired_member_present": mirror_exists,
            },
            "set": None,
        },
        "filename_evidence": filename_evidence,
        "posecode_evidence": None,
        "validation": {
            **validation,
            "warnings": sorted(set(warnings)),
        },
        "index_state": "quarantine" if validation["errors"] else "needs_review",
        "eligibility": {
            "geometry_engineering": geometry_eligible,
            "semantic_index": False,
            "release": False,
            "blockers": blockers,
        },
        "review": {"status": "generated", "reviewed_fields": []},
        "content_revision": 1,
    }


def _library_version(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["pose_id"]):
        digest.update(record["pose_id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(record["bvh"]["sha256"].encode("ascii"))
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def _mark_duplicate_content(records: list[dict]) -> int:
    by_hash = Counter(record["bvh"]["sha256"] for record in records)
    duplicate_hashes = {value for value, count in by_hash.items() if count > 1}
    for record in records:
        if record["bvh"]["sha256"] in duplicate_hashes:
            record["validation"]["warnings"].append("duplicate_content_hash")
            record["validation"]["warnings"] = sorted(
                set(record["validation"]["warnings"])
            )
            record["eligibility"]["blockers"].append("duplicate_content_review_required")
    return len(duplicate_hashes)


def _header(bvh_dir: Path, records: list[dict], duplicate_hash_groups: int, partial: bool) -> dict:
    mirrored = [r for r in records if r["grouping"]["variant"]["kind"] == "mirrored"]
    complete_pairs = sum(
        1
        for record in records
        if record["grouping"]["variant"]["kind"] == "original"
        and record["grouping"]["variant"]["paired_member_present"]
    )
    orphan_mirrors = sum(
        1
        for record in mirrored
        if not record["grouping"]["variant"]["paired_member_present"]
    )
    return {
        "record_type": "inventory_header",
        "schema_version": SCHEMA_VERSION,
        "inventory_version": INVENTORY_VERSION,
        "generator": {
            "name": "scripts/init_bvh_tag_inventory.py",
            "version": 1,
            "filename_parser_version": FILENAME_PARSER_VERSION,
        },
        "input": {"bvh_dir": _display_path(bvh_dir), "partial_inventory": partial},
        "pose_library_version": _library_version(records),
        "counts": {
            "pose_members": len(records),
            "semantic_units": len({r["grouping"]["semantic_unit_id"] for r in records}),
            "mirrored_members": len(mirrored),
            "complete_mirror_pairs": complete_pairs,
            "orphan_mirrors": orphan_mirrors,
            "parse_failures": sum(
                r["validation"]["parse_status"] != "pass" for r in records
            ),
            "mapping_failures": sum(
                r["validation"]["mapping_status"] != "pass" for r in records
            ),
            "duplicate_hash_groups": duplicate_hash_groups,
        },
    }


def _write_jsonl(output: Path, rows: list[dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                stream.write("\n")
        os.replace(temp_name, output)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the pre-tagging BVH inventory without inventing semantics."
    )
    parser.add_argument("--bvh-dir", default="data/bvh")
    parser.add_argument("--output", default="data/semantic/inventory.v1.jsonl")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="diagnostic subset only; the header marks the output as partial",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bvh_dir = Path(args.bvh_dir)
    if not bvh_dir.is_dir():
        raise SystemExit(f"BVH directory does not exist: {bvh_dir}")

    all_paths = sorted(
        (path for path in bvh_dir.iterdir() if path.is_file() and path.suffix.lower() == ".bvh"),
        key=lambda path: path.name,
    )
    pose_ids = {path.stem for path in all_paths}
    paths = all_paths
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No root-level .bvh files found: {bvh_dir}")

    records = [_member_record(path, pose_ids) for path in paths]
    duplicate_hash_groups = _mark_duplicate_content(records)
    header = _header(bvh_dir, records, duplicate_hash_groups, partial=args.limit > 0)
    output = Path(args.output)
    _write_jsonl(output, [header, *records])

    counts = header["counts"]
    print(
        f"[inventory] {counts['pose_members']} poses / {counts['semantic_units']} semantic units "
        f"/ {counts['complete_mirror_pairs']} mirror pairs -> {output}"
    )
    print(
        f"[validation] parse_failures={counts['parse_failures']} "
        f"mapping_failures={counts['mapping_failures']} "
        f"orphan_mirrors={counts['orphan_mirrors']} "
        f"duplicate_hash_groups={counts['duplicate_hash_groups']}"
    )
    print(f"[version] {header['pose_library_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
