"""Offline BVH semantic tagging catalog builder.

The builder creates reviewable proposal artifacts from the deterministic
inventory.  It intentionally does not create a production ``pose_semantics.db``:
captions, aliases, actions, props, and contextual facts require explicit human
decisions before the dense index can be built.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any, Iterable

import numpy as np

from .bvh import load_coco17, parse_bvh
from .posecode import (
    ATOM_SCHEMA_VERSION,
    COORDINATE_PROFILE,
    POSECODE_VERSION,
    common_neutral_atoms,
    measure_posecode,
    mirror_atom_report,
    render_posecode_documents,
)


SEMANTIC_SCHEMA_VERSION = 1
SEMANTIC_VOCAB_VERSION = 1
CATALOG_BUILDER_VERSION = 3
PROPOSAL_RENDERER_VERSION = 2
PASSAGE_TEMPLATE_VERSION = 1


@dataclass(frozen=True)
class CatalogPaths:
    inventory: Path
    bvh_dir: Path
    raw_dir: Path
    output_dir: Path
    cmu_catalog_html: Path | None = None
    cmu_catalog_captured_at: str | None = None
    exclusions_path: Path | None = None


class _CMUCatalogParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_row = False
        self._in_cell = False
        self._cells: list[str] = []
        self._cell_parts: list[str] = []
        self._hrefs: list[str] = []
        self.records: dict[tuple[str, str], dict[str, Any]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        if lower == "tr":
            self._in_row = True
            self._cells = []
            self._hrefs = []
        elif lower in {"td", "th"} and self._in_row:
            self._in_cell = True
            self._cell_parts = []
        elif lower == "a" and self._in_row:
            href = dict(attrs).get("href")
            if href:
                self._hrefs.append(href)

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower in {"td", "th"} and self._in_cell:
            self._cells.append(" ".join("".join(self._cell_parts).split()))
            self._in_cell = False
            self._cell_parts = []
        elif lower == "tr" and self._in_row:
            self._finish_row()
            self._in_row = False

    def _finish_row(self) -> None:
        amc = None
        for href in self._hrefs:
            match = re.fullmatch(r"/subjects/(\d+)/(\d+)_(\d+)\.amc", href)
            if match:
                amc = match
                break
        if amc is None or len(self._cells) < 3:
            return
        subject_path, subject_file, trial = amc.groups()
        if int(subject_path) != int(subject_file):
            return
        title = self._cells[2].strip()
        fps = None
        for cell in reversed(self._cells[3:]):
            if re.fullmatch(r"\d+(?:\.\d+)?", cell):
                fps = float(cell)
                if fps.is_integer():
                    fps = int(fps)
                break
        key = (str(int(subject_path)), str(int(trial)))
        self.records[key] = {
            "subject_id": subject_path,
            "clip_id": trial,
            "title": title or None,
            "fps": fps,
            "artifact_uri": f"https://mocap.cs.cmu.edu/subjects/{subject_path}/{subject_file}_{trial}.amc",
        }


def parse_cmu_catalog(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    parser = _CMUCatalogParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return parser.records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
    return rows


def read_inventory(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _read_jsonl(path)
    if not rows or rows[0].get("record_type") != "inventory_header":
        raise ValueError(f"inventory header missing: {path}")
    header, members = rows[0], rows[1:]
    if any(row.get("record_type") != "pose_member_inventory" for row in members):
        raise ValueError("inventory contains an unexpected record type")
    pose_ids = [row["pose_id"] for row in members]
    if len(pose_ids) != len(set(pose_ids)):
        raise ValueError("inventory contains duplicate pose_id values")
    expected = int(header.get("counts", {}).get("pose_members", -1))
    if expected != len(members):
        raise ValueError(f"inventory count mismatch: header={expected}, rows={len(members)}")
    return header, members


def _source_identity(member: dict[str, Any], raw_dir: Path) -> tuple[str, str]:
    evidence = member["filename_evidence"]
    pattern = evidence["pattern"]
    if pattern == "cmu_subject_trial_frame":
        subject = evidence["native_subject_id_hint"]
        clip = evidence["native_clip_id_hint"]
        return f"cmu:{subject}_{clip}", "cmu"
    if pattern == "rokoko_clip_frame":
        hint = evidence["source_clip_id_hint"].removeprefix("filename_hint:rokoko:")
        return f"rokoko:{hint}", "rokoko"
    label = evidence.get("label_en_hint") or evidence["raw_stem"]
    if (raw_dir / f"{label}.bvh").is_file():
        return f"local_action_raw:{label}", "local_action_raw"
    return f"local_named:{label}", "local_named"


def _frame_time(path: Path) -> float | None:
    match = re.search(r"(?im)^\s*Frame\s+Time:\s*([0-9.eE+-]+)\s*$", path.read_text(errors="replace"))
    if not match:
        return None
    value = float(match.group(1))
    return value if value > 0 else None


def build_source_clips(
    members: list[dict[str, Any]],
    *,
    raw_dir: Path,
    cmu_records: dict[tuple[str, str], dict[str, Any]],
    cmu_snapshot_ref: str | None,
    cmu_captured_at: str | None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    representative: dict[str, tuple[str, dict[str, Any]]] = {}
    pose_to_source: dict[str, str] = {}
    for member in members:
        source_clip_id, kind = _source_identity(member, raw_dir)
        pose_to_source[member["pose_id"]] = source_clip_id
        representative.setdefault(source_clip_id, (kind, member))

    output: list[dict[str, Any]] = []
    for source_clip_id, (kind, member) in sorted(representative.items()):
        evidence = member["filename_evidence"]
        if kind == "cmu":
            subject_hint = evidence["native_subject_id_hint"]
            clip_hint = evidence["native_clip_id_hint"]
            catalog = cmu_records.get((str(int(subject_hint)), str(int(clip_hint))))
            verified_fields = [
                "provider",
                "collection.id",
                "native_ids.subject_id",
                "native_ids.clip_id",
            ]
            unresolved = ["native_ids.asset_id", "original.sha256", "license_ref"]
            if catalog:
                verified_fields.extend(
                    ["original.filename", "original.catalog_uri", "original.artifact_uri", "original.fps"]
                )
                if catalog["title"]:
                    verified_fields.append("original.title")
                else:
                    unresolved.append("original.title")
            else:
                unresolved.extend(
                    [
                        "original.title",
                        "original.filename",
                        "original.artifact_uri",
                        "original.fps",
                    ]
                )
            subject_file = catalog["subject_id"] if catalog else subject_hint
            clip_file = catalog["clip_id"] if catalog else clip_hint
            output.append(
                {
                    "record_type": "source_clip",
                    "schema_version": 1,
                    "source_clip_id": source_clip_id,
                    "provider": "cmu_graphics_lab",
                    "collection": {"id": "cmu_motion_capture_database", "version": None},
                    "native_ids": {"subject_id": subject_hint, "asset_id": None, "clip_id": clip_hint},
                    "original": {
                        "title": catalog["title"] if catalog else None,
                        "filename": f"{subject_file}_{clip_file}.amc" if catalog else None,
                        "catalog_uri": (
                            f"https://mocap.cs.cmu.edu/search.php?subjectnumber={int(subject_hint)}&trinum={int(clip_hint)}"
                        ),
                        "artifact_uri": catalog["artifact_uri"] if catalog else None,
                        "sha256": None,
                        "fps": catalog["fps"] if catalog else None,
                    },
                    "license_ref": None,
                    "catalog_evidence": {
                        "status": "verified" if catalog else "not_found_in_snapshot",
                        "snapshot_ref": cmu_snapshot_ref,
                        "captured_at": cmu_captured_at,
                    },
                    "verification": {
                        "status": "catalog_verified_file_unverified" if catalog else "filename_ids_only",
                        "verified_fields": verified_fields,
                        "unresolved_fields": sorted(set(unresolved)),
                    },
                }
            )
            continue

        label = evidence.get("label_en_hint") or evidence["raw_stem"]
        raw_path = raw_dir / f"{label}.bvh"
        has_raw = kind == "local_action_raw" and raw_path.is_file()
        frame_time = _frame_time(raw_path) if has_raw else None
        output.append(
            {
                "record_type": "source_clip",
                "schema_version": 1,
                "source_clip_id": source_clip_id,
                "provider": None,
                "collection": {"id": kind, "version": None},
                "native_ids": {"subject_id": None, "asset_id": None, "clip_id": None},
                "original": {
                    "title": None,
                    "local_label": label,
                    "filename": raw_path.name if has_raw else None,
                    "catalog_uri": None,
                    "artifact_uri": _repo_path(raw_path) if has_raw else None,
                    "sha256": _sha256_file(raw_path) if has_raw else None,
                    "fps": round(1.0 / frame_time, 6) if frame_time else None,
                },
                "license_ref": None,
                "catalog_evidence": {"status": "missing", "snapshot_ref": None, "captured_at": None},
                "verification": {
                    "status": "local_artifact_verified_external_origin_unknown" if has_raw else "filename_hint_only",
                    "verified_fields": (
                        [
                            "collection.id",
                            "original.local_label",
                            "original.filename",
                            "original.artifact_uri",
                            "original.sha256",
                            "original.fps",
                        ]
                        if has_raw
                        else ["collection.id", "original.local_label"]
                    ),
                    "unresolved_fields": [
                        "provider",
                        "native_ids",
                        "original.title",
                        "original.catalog_uri",
                        "license_ref",
                    ],
                },
            }
        )
    return output, pose_to_source


def _joint_signature(joints: list[Any]) -> tuple[Any, ...]:
    return tuple((joint[0], joint[1], tuple(joint[3])) for joint in joints)


def _match_raw_frame(derived_path: Path, raw_path: Path, frame_hint: int | None) -> dict[str, Any]:
    if frame_hint is None or not raw_path.is_file():
        return {"status": "unverified", "frame_index": None, "frame_index_base": "unknown", "evidence": []}
    try:
        derived_joints, derived_frames = parse_bvh(str(derived_path))
        raw_joints, raw_frames = parse_bvh(str(raw_path))
    except Exception as exc:
        return {
            "status": "parse_failed",
            "frame_index": None,
            "frame_index_base": "unknown",
            "evidence": [],
            "warning": f"raw_match_parse_error:{type(exc).__name__}",
        }
    if len(derived_frames) != 1 or _joint_signature(derived_joints) != _joint_signature(raw_joints):
        return {"status": "hierarchy_or_frame_mismatch", "frame_index": None, "frame_index_base": "unknown", "evidence": []}
    candidates = [(frame_hint, "zero"), (frame_hint - 1, "one")]
    for index, base in candidates:
        if index < 0 or index >= len(raw_frames):
            continue
        if np.allclose(derived_frames[0], raw_frames[index], rtol=0.0, atol=1e-6):
            return {
                "status": "verified",
                "frame_index": int(index),
                "frame_index_base": base,
                "evidence": [
                    {
                        "kind": "exact_channel_match",
                        "ref": f"{_repo_path(raw_path)}#frame={index};base=zero",
                        "supports": [
                            "source_clip_id",
                            "extraction.selected_frame_index",
                            "extraction.frame_index_base",
                            "derivation.parent_artifact_sha256",
                        ],
                        "detail": f"all {raw_frames.shape[1]} channel values equal",
                    }
                ],
            }
    return {"status": "frame_values_differ", "frame_index": None, "frame_index_base": "unknown", "evidence": []}


def _existing_library_numbers(lineage_path: Path, registry_path: Path) -> dict[str, str]:
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        assignments = registry.get("assignments")
        if not isinstance(assignments, dict):
            raise ValueError(f"invalid library number registry: {registry_path}")
        return {str(key): str(value) for key, value in assignments.items()}
    mapping: dict[str, str] = {}
    for row in _read_jsonl(lineage_path):
        if row.get("record_type") == "pose_lineage" and row.get("library_no"):
            mapping[row["pose_id"]] = row["library_no"]
    return mapping


def build_pose_lineage(
    members: list[dict[str, Any]],
    source_clips: list[dict[str, Any]],
    pose_to_source: dict[str, str],
    *,
    bvh_dir: Path,
    raw_dir: Path,
    existing_path: Path,
    registry_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = {row["source_clip_id"]: row for row in source_clips}
    library_numbers = _existing_library_numbers(existing_path, registry_path)
    used_numbers = [int(value.split("-")[-1]) for value in library_numbers.values() if re.fullmatch(r"BVH-\d{6}", value)]
    next_number = max(used_numbers, default=0) + 1
    for pose_id in sorted(member["pose_id"] for member in members):
        if pose_id not in library_numbers:
            library_numbers[pose_id] = f"BVH-{next_number:06d}"
            next_number += 1

    rows: list[dict[str, Any]] = []
    by_pose = {member["pose_id"]: member for member in members}
    for member in sorted(members, key=lambda item: item["pose_id"]):
        pose_id = member["pose_id"]
        evidence = member["filename_evidence"]
        source_clip_id = pose_to_source[pose_id]
        source = sources[source_clip_id]
        source_label = source["original"].get("local_label")
        raw_path = raw_dir / f"{source_label}.bvh" if source_label else Path("__missing__")
        bvh_path = bvh_dir / f"{pose_id}.bvh"
        is_mirror = member["grouping"]["variant"]["kind"] == "mirrored"
        match = (
            {"status": "derived_mirror", "frame_index": None, "frame_index_base": "unknown", "evidence": []}
            if is_mirror
            else _match_raw_frame(bvh_path, raw_path, evidence.get("selected_frame_index_hint"))
        )

        operations: list[dict[str, Any]] = []
        if evidence.get("selected_frame_index_hint") is not None:
            operations.append({"type": "extract_frame", "status": "verified" if match["status"] == "verified" else "inferred"})
        for operation in evidence.get("derivation_hints", []):
            operations.append({"type": operation, "status": "filename_inferred"})
        if evidence.get("retarget_profile_hint"):
            operations.append({"type": "retarget", "status": "filename_inferred"})
        if is_mirror:
            operations.append({"type": "mirror", "status": "paired_member_present" if member["grouping"]["variant"]["paired_member_present"] else "orphan"})

        fps = source["original"].get("fps")
        verified_index = match.get("frame_index")
        rows.append(
            {
                "record_type": "pose_lineage",
                "schema_version": 1,
                "pose_lineage_id": f"lineage:{pose_id}",
                "library_no": library_numbers[pose_id],
                "pose_id": pose_id,
                "bvh_filename": bvh_path.name,
                "bvh_sha256": member["bvh"]["sha256"],
                "source_clip_id": source_clip_id,
                "extraction": {
                    "selected_frame_index": verified_index,
                    "selected_frame_index_hint": evidence.get("selected_frame_index_hint"),
                    "frame_index_base": match.get("frame_index_base", "unknown"),
                    "source_fps": fps,
                    "source_time_seconds": (
                        round(verified_index / float(fps), 6) if verified_index is not None and fps else None
                    ),
                    "sample_ordinal": evidence.get("sample_ordinal_hint"),
                },
                "derivation": {
                    "parent_pose_id": member["grouping"]["variant"].get("mirror_of"),
                    "parent_artifact_sha256": (
                        by_pose[member["grouping"]["variant"]["mirror_of"]]["bvh"]["sha256"]
                        if is_mirror and member["grouping"]["variant"].get("mirror_of") in by_pose
                        else _sha256_file(raw_path)
                        if match["status"] == "verified"
                        else None
                    ),
                    "operations": operations,
                    "conversion_recipe_id": None,
                    "retarget_profile": evidence.get("retarget_profile_hint"),
                },
                "evidence": match.get("evidence", []),
                "verification": {
                    "catalog_match_status": source["catalog_evidence"]["status"],
                    "file_lineage_status": match["status"],
                    "warnings": [
                        item
                        for item in [
                            match.get("warning"),
                            "external_provider_and_license_unresolved" if source["provider"] is None else None,
                            "selected_frame_base_unverified" if evidence.get("selected_frame_index_hint") is not None and verified_index is None else None,
                        ]
                        if item
                    ],
                },
            }
        )
    registry = {
        "record_type": "library_number_registry",
        "schema_version": 1,
        "prefix": "BVH",
        "allocation_policy": "append_only_never_reuse",
        "next_number": next_number,
        "assignments": dict(sorted(library_numbers.items())),
    }
    return rows, registry


def _source_label(source: dict[str, Any]) -> str | None:
    return source["original"].get("title") or source["original"].get("local_label")


def _source_is_excluded(source: dict[str, Any]) -> bool:
    policy = source.get("library_policy") or {}
    return policy.get("state") == "pending_removal"


def apply_source_exclusions(
    source_clips: list[dict[str, Any]],
    exclusions_path: Path | None,
) -> dict[str, Any] | None:
    """Overlay an explicit project-owner exclusion policy without deleting BVHs."""
    if exclusions_path is None:
        return None
    document = json.loads(exclusions_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("unsupported library exclusion schema")
    source_ids = document.get("source_clip_ids")
    if not isinstance(source_ids, list) or len(source_ids) != len(set(source_ids)):
        raise ValueError("library exclusions must contain unique source_clip_ids")
    known = {row["source_clip_id"] for row in source_clips}
    missing = sorted(set(source_ids) - known)
    if missing:
        raise ValueError(f"library exclusions reference unknown sources: {missing}")
    effect = document.get("default_effect") or {}
    if any(effect.get(field) is not False for field in ("semantic_index", "geometry_index", "release")):
        raise ValueError("excluded sources must disable semantic, geometry, and release eligibility")
    decision = document.get("decision") or {}
    for source in source_clips:
        if source["source_clip_id"] not in source_ids:
            continue
        source["library_policy"] = {
            "state": "pending_removal",
            "policy_id": document["policy_id"],
            "semantic_index": False,
            "geometry_index": False,
            "release": False,
            "reason_codes": decision.get("reason_codes", []),
            "disposition": decision.get("disposition", "pending_physical_deletion"),
            "requested_by": decision.get("requested_by"),
            "requested_at": decision.get("requested_at"),
        }
    return document


def _source_display(source: dict[str, Any]) -> str:
    """Return the short provider label shown in human review artifacts."""
    if source["provider"] == "cmu_graphics_lab":
        return "CMU"
    return source["provider"] or "unknown"


def _proposal_priority(member_rows: list[dict[str, Any]], mirror_report: dict[str, Any] | None, source: dict[str, Any]) -> str:
    if _source_is_excluded(source):
        return "PX"
    if any(row["validation"]["errors"] for row in member_rows):
        return "P0"
    if any(not row["grouping"]["variant"]["paired_member_present"] and row["grouping"]["variant"]["kind"] == "mirrored" for row in member_rows):
        return "P0"
    # Paired mirror atom differences are canonicalized from the original member
    # below. They are retained as diagnostics, not sent to human review.
    # A missing action name is not a search blocker: deterministic BVH posecode
    # remains searchable. Human review is reserved for structural failures,
    # contradictions, and hard contextual facets.
    return "P2"


def build_proposals(
    members: list[dict[str, Any]],
    source_clips: list[dict[str, Any]],
    pose_to_source: dict[str, str],
    *,
    bvh_dir: Path,
    existing_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    sources = {row["source_clip_id"]: row for row in source_clips}
    posecodes: dict[str, dict[str, Any]] = {}
    for member in sorted(members, key=lambda item: item["pose_id"]):
        pose_id = member["pose_id"]
        bvh_path = bvh_dir / f"{pose_id}.bvh"
        joints, _ = load_coco17(str(bvh_path), frame=0)
        posecodes[pose_id] = measure_posecode(joints, provenance_ref=_repo_path(bvh_path))

    groups: dict[str, list[dict[str, Any]]] = {}
    for member in members:
        groups.setdefault(member["grouping"]["semantic_unit_id"], []).append(member)

    old_rows = [row for row in _read_jsonl(existing_path) if row.get("record_type") == "semantic_proposal"]
    by_id = {row["proposal_id"]: row for row in old_rows}
    revisions: dict[str, int] = {}
    for row in old_rows:
        revisions[row["semantic_unit_id"]] = max(revisions.get(row["semantic_unit_id"], 0), int(row.get("content_revision", 0)))

    current: dict[str, dict[str, Any]] = {}
    mirror_pass = 0
    mirror_canonicalized = 0
    priorities: dict[str, int] = {}
    for semantic_unit_id, member_rows in sorted(groups.items()):
        member_rows = sorted(member_rows, key=lambda item: item["pose_id"])
        member_ids = [row["pose_id"] for row in member_rows]
        source_ids = {pose_to_source[pose_id] for pose_id in member_ids}
        source_clip_id = sorted(source_ids)[0]
        source = sources[source_clip_id]
        original_row = next((row for row in member_rows if row["grouping"]["variant"]["kind"] == "original"), None)
        mirror_row = next((row for row in member_rows if row["grouping"]["variant"]["kind"] == "mirrored"), None)
        mirror_report = None
        if original_row and mirror_row:
            mirror_report = mirror_atom_report(posecodes[original_row["pose_id"]], posecodes[mirror_row["pose_id"]])
            if mirror_report["status"] == "pass":
                mirror_pass += 1
            else:
                mirror_canonicalized += 1
                mirror_report["resolution"] = {
                    "status": "canonicalized",
                    "method": "original_posecode_side_neutralization",
                    "search_tag_review_required": False,
                }

        # A mirror file represents the same semantic pose with sides swapped.
        # Use the original member as the canonical source for unit-level search
        # atoms so small conversion/threshold differences cannot delete or add a
        # semantic search tag. Raw atoms and continuous measurements for both
        # files stay in member_posecodes for diagnostics.
        if original_row and mirror_row:
            common_atoms = common_neutral_atoms([posecodes[original_row["pose_id"]]["observed_atoms"]])
        else:
            common_atoms = common_neutral_atoms([posecodes[pose_id]["observed_atoms"] for pose_id in member_ids])
        rendered = render_posecode_documents(common_atoms)
        label = _source_label(source)
        excluded = _source_is_excluded(source)
        aliases_en = [label] if label else []
        priority = _proposal_priority(member_rows, mirror_report, source)
        priorities[priority] = priorities.get(priority, 0) + 1
        warnings: list[str] = []
        if label is None and not excluded:
            warnings.extend(["action_name_missing", "observed_only_search"])
        elif label is None:
            warnings.append("action_name_missing")
        if excluded:
            warnings.append("source_excluded_pending_removal")
        if source["provider"] != "cmu_graphics_lab":
            warnings.append("license_unresolved")
        if (
            source["provider"] != "cmu_graphics_lab"
            and source["verification"]["status"]
            not in {"catalog_verified_file_unverified", "local_artifact_verified_external_origin_unknown"}
        ):
            warnings.append("source_unverified")
        if mirror_report and mirror_report["status"] != "pass":
            warnings.append("mirror_posecode_canonicalized")
        if not original_row:
            warnings.append("orphan_mirror")

        fingerprint = _sha256_json(
            {
                "members": [(row["pose_id"], row["bvh"]["sha256"]) for row in member_rows],
                "source_clip_id": source_clip_id,
                "source_semantic_seed": label,
                "source_verification": source["verification"]["status"],
                "library_policy": source.get("library_policy"),
            }
        )
        run_payload = {
            "input_fingerprint": fingerprint,
            "catalog_builder_version": CATALOG_BUILDER_VERSION,
            "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
            "semantic_vocab_version": SEMANTIC_VOCAB_VERSION,
            "filename_parser_version": member_rows[0]["filename_evidence"]["parser_version"],
            "posecode_version": POSECODE_VERSION,
            "coordinate_profile": COORDINATE_PROFILE,
            "proposal_renderer_version": PROPOSAL_RENDERER_VERSION,
            "prompt_version": None,
            "model_id": None,
        }
        proposal_id = _sha256_json(run_payload)
        if proposal_id in by_id:
            current[semantic_unit_id] = by_id[proposal_id]
            continue

        proposal = {
            "record_type": "semantic_proposal",
            "schema_version": SEMANTIC_SCHEMA_VERSION,
            "proposal_id": proposal_id,
            "semantic_unit_id": semantic_unit_id,
            "semantic_unit_type": "pose_variant_group",
            "member_pose_ids": member_ids,
            "source_clip_id": source_clip_id,
            "input_fingerprint": fingerprint,
            "semantic": {
                "caption_ko": rendered["ko"],
                "caption_en": rendered["en"],
                "aliases_ko": [],
                "aliases_en": aliases_en,
                "coarse_action": "unknown",
                "fine_actions": [],
                "motion_phase": "unknown",
                "posture": [],
                "gesture": [],
                "interaction": {"kind": "solo", "set_id": None, "set_role": None},
                "intended_props": [],
                "style_context": [],
                "unit_atoms": common_atoms,
            },
            "member_posecodes": {pose_id: posecodes[pose_id] for pose_id in member_ids},
            "mirror_validation": mirror_report,
            "field_evidence": {
                "semantic.caption_ko": [{"source": "posecode_renderer", "ref": f"posecode-v{POSECODE_VERSION}"}],
                "semantic.caption_en": [{"source": "posecode_renderer", "ref": f"posecode-v{POSECODE_VERSION}"}],
                "semantic.aliases_en": ([{"source": "source_catalog_or_filename", "ref": source_clip_id}] if label else []),
                "semantic.unit_atoms": [{"source": "bvh_rule_group", "ref": semantic_unit_id}],
                "member_posecodes": [{"source": "bvh_rule", "ref": f"posecode-v{POSECODE_VERSION}"}],
            },
            "generator": {
                "run_key": proposal_id,
                "name": "scripts/build_semantic_tagging.py",
                "version": CATALOG_BUILDER_VERSION,
                "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
                "semantic_vocab_version": SEMANTIC_VOCAB_VERSION,
                "atom_schema_version": ATOM_SCHEMA_VERSION,
                "filename_parser_version": member_rows[0]["filename_evidence"]["parser_version"],
                "posecode_version": POSECODE_VERSION,
                "coordinate_profile": COORDINATE_PROFILE,
                "proposal_renderer_version": PROPOSAL_RENDERER_VERSION,
                "vlm_provider": None,
                "vlm_model_id": None,
                "prompt_version": None,
            },
            "validation": {"errors": [], "warnings": sorted(set(warnings)), "review_priority": priority},
            "workflow_status": (
                "rejected"
                if excluded
                else "needs_review"
                if priority in {"P0", "P1"}
                else "auto_verified_observed_tags"
            ),
            "content_revision": revisions.get(semantic_unit_id, 0) + 1,
        }
        old_rows.append(proposal)
        by_id[proposal_id] = proposal
        current[semantic_unit_id] = proposal

    stats = {
        "posecode_members": len(posecodes),
        "member_observed_atoms": sum(len(item["observed_atoms"]) for item in posecodes.values()),
        "unit_observed_atoms": sum(len(proposal["semantic"]["unit_atoms"]) for proposal in current.values()),
        "mirror_pairs_pass": mirror_pass,
        "mirror_pairs_canonicalized": mirror_canonicalized,
        "mirror_pairs_needing_review": 0,
        "action_name_missing_units": sum(
            _source_label(sources[p["source_clip_id"]]) is None
            and not _source_is_excluded(sources[p["source_clip_id"]])
            for p in current.values()
        ),
        "excluded_units": sum(
            _source_is_excluded(sources[p["source_clip_id"]]) for p in current.values()
        ),
        "excluded_pose_members": sum(
            len(p["member_pose_ids"])
            for p in current.values()
            if _source_is_excluded(sources[p["source_clip_id"]])
        ),
        "auto_verified_observed_tag_units": sum(
            p["workflow_status"] == "auto_verified_observed_tags" for p in current.values()
        ),
        "review_priorities": priorities,
        "proposal_history_rows": len(old_rows),
        "current_proposals": len(current),
    }
    return old_rows, current, stats


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_review_queues(
    output_dir: Path,
    current: dict[str, dict[str, Any]],
    source_clips: list[dict[str, Any]],
    lineage: list[dict[str, Any]],
) -> None:
    review_fields = [
        "priority", "workflow_status", "semantic_unit_id", "member_pose_ids", "preview_path",
        "source_provider", "source_clip_id", "action_name_status", "filename_seed",
        "caption_ko_proposed", "caption_ko_edited", "aliases_ko_edited",
        "coarse_action", "fine_actions", "motion_phase", "intended_props", "style_context",
        "unit_atoms_json", "member_observed_atoms_summary", "warnings", "decision", "reason_codes",
        "reviewer", "notes",
    ]
    review_rows = []
    source_by_id = {row["source_clip_id"]: row for row in source_clips}
    for unit_id, proposal in sorted(current.items(), key=lambda item: (item[1]["validation"]["review_priority"], item[0])):
        source = source_by_id[proposal["source_clip_id"]]
        excluded = _source_is_excluded(source)
        preview_pose_id = next(
            (pose_id for pose_id in proposal["member_pose_ids"] if not pose_id.endswith("_mirror")),
            proposal["member_pose_ids"][0],
        )
        preview_candidate = Path(__file__).resolve().parents[1] / "data/thumbs" / f"{preview_pose_id}__front.png"
        member_summary = {
            pose_id: {
                "atom_count": len(value["observed_atoms"]),
                "measurements": len(value["measurements"]),
            }
            for pose_id, value in proposal["member_posecodes"].items()
        }
        review_rows.append(
            {
                "priority": proposal["validation"]["review_priority"],
                "workflow_status": proposal["workflow_status"],
                "semantic_unit_id": unit_id,
                "member_pose_ids": json.dumps(proposal["member_pose_ids"], ensure_ascii=False, sort_keys=True),
                "preview_path": _repo_path(preview_candidate) if preview_candidate.is_file() else "",
                "source_provider": _source_display(source),
                "source_clip_id": proposal["source_clip_id"],
                "action_name_status": (
                    "excluded" if excluded else "present" if _source_label(source) else "missing"
                ),
                "filename_seed": _source_label(source) or "",
                "caption_ko_proposed": proposal["semantic"]["caption_ko"],
                "caption_ko_edited": "",
                "aliases_ko_edited": "",
                "coarse_action": proposal["semantic"]["coarse_action"],
                "fine_actions": "[]",
                "motion_phase": proposal["semantic"]["motion_phase"],
                "intended_props": "[]",
                "style_context": "[]",
                "unit_atoms_json": json.dumps(
                    proposal["semantic"]["unit_atoms"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                "member_observed_atoms_summary": json.dumps(
                    member_summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                "warnings": ";".join(proposal["validation"]["warnings"]),
                "decision": "reject" if excluded else "",
                "reason_codes": (
                    ";".join(source["library_policy"].get("reason_codes", [])) if excluded else ""
                ),
                "reviewer": source.get("library_policy", {}).get("requested_by") or "",
                "notes": "pending physical deletion" if excluded else "",
            }
        )
    _atomic_write_csv(output_dir / "review_queue.csv", review_fields, review_rows)

    lineage_by_pose = {row["pose_id"]: row for row in lineage}
    missing_action_fields = [
        "no", "source_provider", "source_clip_id", "semantic_unit_ids", "member_pose_ids",
        "library_numbers", "preview_paths", "action_name",
    ]
    missing_action_rows = []
    proposals_by_source: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for unit_id, proposal in current.items():
        proposals_by_source.setdefault(proposal["source_clip_id"], []).append((unit_id, proposal))
    missing_sources = sorted(
        (
            source
            for source in source_clips
            if _source_label(source) is None and not _source_is_excluded(source)
        ),
        key=lambda source: source["source_clip_id"],
    )
    for number, source in enumerate(missing_sources, 1):
        entries = sorted(proposals_by_source.get(source["source_clip_id"], []))
        pose_ids = [pose_id for _, proposal in entries for pose_id in proposal["member_pose_ids"]]
        preview_paths = []
        for _, proposal in entries:
            preview_pose_id = next(
                (pose_id for pose_id in proposal["member_pose_ids"] if not pose_id.endswith("_mirror")),
                proposal["member_pose_ids"][0],
            )
            preview = Path(__file__).resolve().parents[1] / "data/thumbs" / f"{preview_pose_id}__front.png"
            if preview.is_file():
                preview_paths.append(_repo_path(preview))
        missing_action_rows.append(
            {
                "no": number,
                "source_provider": _source_display(source),
                "source_clip_id": source["source_clip_id"],
                "semantic_unit_ids": json.dumps([unit_id for unit_id, _ in entries], ensure_ascii=False),
                "member_pose_ids": json.dumps(pose_ids, ensure_ascii=False),
                "library_numbers": json.dumps(
                    [lineage_by_pose[pose_id]["library_no"] for pose_id in pose_ids], ensure_ascii=False
                ),
                "preview_paths": json.dumps(preview_paths, ensure_ascii=False),
                "action_name": "",
            }
        )
    _atomic_write_csv(
        output_dir / "missing_action_names.csv", missing_action_fields, missing_action_rows
    )

    excluded_fields = [
        "source_provider", "source_clip_id", "semantic_unit_ids", "member_pose_ids",
        "library_numbers", "reason_codes", "disposition", "requested_by", "requested_at",
    ]
    excluded_rows = []
    for source in sorted(
        (source for source in source_clips if _source_is_excluded(source)),
        key=lambda source: source["source_clip_id"],
    ):
        entries = sorted(proposals_by_source.get(source["source_clip_id"], []))
        pose_ids = [pose_id for _, proposal in entries for pose_id in proposal["member_pose_ids"]]
        policy = source["library_policy"]
        excluded_rows.append(
            {
                "source_provider": _source_display(source),
                "source_clip_id": source["source_clip_id"],
                "semantic_unit_ids": json.dumps([unit_id for unit_id, _ in entries], ensure_ascii=False),
                "member_pose_ids": json.dumps(pose_ids, ensure_ascii=False),
                "library_numbers": json.dumps(
                    [lineage_by_pose[pose_id]["library_no"] for pose_id in pose_ids], ensure_ascii=False
                ),
                "reason_codes": ";".join(policy.get("reason_codes", [])),
                "disposition": policy.get("disposition", ""),
                "requested_by": policy.get("requested_by") or "",
                "requested_at": policy.get("requested_at") or "",
            }
        )
    _atomic_write_csv(output_dir / "excluded_source_clips.csv", excluded_fields, excluded_rows)

    provenance_fields = [
        "library_no", "pose_id", "bvh_filename", "source_clip_id", "provider", "collection_id",
        "native_subject_id", "native_asset_id", "native_clip_id", "original_title", "original_filename",
        "selected_frame_index", "selected_frame_index_hint", "frame_index_base", "source_fps",
        "sample_ordinal", "retarget_profile", "operations", "source_uri", "source_sha256",
        "catalog_snapshot_ref", "catalog_match_status", "file_lineage_status", "reviewer", "notes",
    ]
    provenance_rows = []
    for row in sorted(lineage, key=lambda item: item["library_no"]):
        source = source_by_id[row["source_clip_id"]]
        provenance_rows.append(
            {
                "library_no": row["library_no"],
                "pose_id": row["pose_id"],
                "bvh_filename": row["bvh_filename"],
                "source_clip_id": row["source_clip_id"],
                "provider": _source_display(source),
                "collection_id": source["collection"]["id"] or "",
                "native_subject_id": source["native_ids"]["subject_id"] or "",
                "native_asset_id": source["native_ids"]["asset_id"] or "",
                "native_clip_id": source["native_ids"]["clip_id"] or "",
                "original_title": source["original"].get("title") or source["original"].get("local_label") or "",
                "original_filename": source["original"].get("filename") or "",
                "selected_frame_index": row["extraction"]["selected_frame_index"],
                "selected_frame_index_hint": row["extraction"]["selected_frame_index_hint"],
                "frame_index_base": row["extraction"]["frame_index_base"],
                "source_fps": row["extraction"]["source_fps"],
                "sample_ordinal": row["extraction"]["sample_ordinal"],
                "retarget_profile": row["derivation"]["retarget_profile"] or "",
                "operations": json.dumps(
                    row["derivation"]["operations"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                "source_uri": source["original"].get("catalog_uri") or source["original"].get("artifact_uri") or "",
                "source_sha256": source["original"].get("sha256") or "",
                "catalog_snapshot_ref": source["catalog_evidence"].get("snapshot_ref") or "",
                "catalog_match_status": row["verification"]["catalog_match_status"],
                "file_lineage_status": row["verification"]["file_lineage_status"],
                "reviewer": "",
                "notes": "",
            }
        )
    _atomic_write_csv(output_dir / "provenance_review_queue.csv", provenance_fields, provenance_rows)


def build_review_index(
    path: Path,
    *,
    inventory_header: dict[str, Any],
    source_clips: list[dict[str, Any]],
    lineage: list[dict[str, Any]],
    current: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        connection = sqlite3.connect(temporary)
        connection.executescript(
            """
            PRAGMA foreign_keys=ON;
            CREATE TABLE meta (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
            CREATE TABLE source_clips (source_clip_id TEXT PRIMARY KEY, record_json TEXT NOT NULL);
            CREATE TABLE pose_lineage (
                pose_id TEXT PRIMARY KEY,
                library_no TEXT NOT NULL UNIQUE,
                source_clip_id TEXT NOT NULL,
                record_json TEXT NOT NULL
            );
            CREATE TABLE semantic_units (
                semantic_unit_id TEXT PRIMARY KEY,
                source_clip_id TEXT NOT NULL,
                review_status TEXT NOT NULL,
                priority TEXT NOT NULL,
                proposal_id TEXT NOT NULL,
                proposal_json TEXT NOT NULL
            );
            CREATE TABLE pose_members (
                pose_id TEXT PRIMARY KEY,
                semantic_unit_id TEXT NOT NULL,
                posecode_version INTEGER NOT NULL,
                coordinate_profile TEXT NOT NULL,
                measurements_json TEXT NOT NULL
            );
            CREATE TABLE semantic_atoms (
                atom_id INTEGER PRIMARY KEY AUTOINCREMENT,
                semantic_unit_id TEXT NOT NULL,
                pose_id TEXT,
                scope TEXT NOT NULL,
                predicate TEXT NOT NULL,
                subject TEXT,
                relation TEXT,
                object TEXT,
                axis TEXT,
                value TEXT,
                measure REAL,
                measure_unit TEXT,
                bucket TEXT,
                polarity TEXT NOT NULL,
                evidence_state TEXT NOT NULL,
                provenance_json TEXT NOT NULL
            );
            CREATE INDEX idx_atoms_predicate ON semantic_atoms(predicate, subject, relation, value, bucket);
            CREATE TABLE text_documents (
                document_id TEXT PRIMARY KEY,
                semantic_unit_id TEXT NOT NULL,
                language TEXT NOT NULL,
                document_type TEXT NOT NULL,
                text TEXT NOT NULL,
                text_sha256 TEXT NOT NULL,
                review_status TEXT NOT NULL,
                provenance_json TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE text_documents_fts USING fts5(
                document_id UNINDEXED,
                semantic_unit_id UNINDEXED,
                language UNINDEXED,
                text,
                tokenize='unicode61'
            );
            CREATE TABLE semantic_embeddings (
                document_id TEXT PRIMARY KEY,
                embedding_blob BLOB,
                embedding_version TEXT
            );
            """
        )
        meta = {
            "artifact_type": "tagging_review_index",
            "production_ready": False,
            "semantic_endpoint_compatible": False,
            "embedding_status": "not_built_pending_action_names_and_model_pin",
            "pose_library_version": inventory_header["pose_library_version"],
            "pose_members": inventory_header["counts"]["pose_members"],
            "semantic_units": len(current),
            "accepted_units": 0,
            "auto_verified_observed_tag_units": sum(
                proposal["workflow_status"] == "auto_verified_observed_tags"
                for proposal in current.values()
            ),
            "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
            "semantic_vocab_version": SEMANTIC_VOCAB_VERSION,
            "atom_schema_version": ATOM_SCHEMA_VERSION,
            "posecode_version": POSECODE_VERSION,
            "coordinate_profile": COORDINATE_PROFILE,
            "passage_template_version": PASSAGE_TEMPLATE_VERSION,
        }
        connection.executemany(
            "INSERT INTO meta(key,value_json) VALUES(?,?)",
            [(key, json.dumps(value, ensure_ascii=False, sort_keys=True)) for key, value in meta.items()],
        )
        connection.executemany(
            "INSERT INTO source_clips VALUES(?,?)",
            [(row["source_clip_id"], json.dumps(row, ensure_ascii=False, sort_keys=True)) for row in source_clips],
        )
        connection.executemany(
            "INSERT INTO pose_lineage VALUES(?,?,?,?)",
            [
                (row["pose_id"], row["library_no"], row["source_clip_id"], json.dumps(row, ensure_ascii=False, sort_keys=True))
                for row in lineage
            ],
        )
        for unit_id, proposal in sorted(current.items()):
            connection.execute(
                "INSERT INTO semantic_units VALUES(?,?,?,?,?,?)",
                (
                    unit_id,
                    proposal["source_clip_id"],
                    proposal["workflow_status"],
                    proposal["validation"]["review_priority"],
                    proposal["proposal_id"],
                    json.dumps(proposal, ensure_ascii=False, sort_keys=True),
                ),
            )
            for atom in proposal["semantic"].get("unit_atoms", []):
                connection.execute(
                    """INSERT INTO semantic_atoms(
                    semantic_unit_id,pose_id,scope,predicate,subject,relation,object,axis,value,
                    measure,measure_unit,bucket,polarity,evidence_state,provenance_json
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        unit_id, None, "unit", atom["predicate"], atom.get("subject"),
                        atom.get("relation"), atom.get("object"), atom.get("axis"), atom.get("value"),
                        atom.get("measure"), atom.get("measure_unit"), atom.get("bucket"), atom["polarity"],
                        atom["evidence_state"], json.dumps(atom["provenance"], ensure_ascii=False, sort_keys=True),
                    ),
                )
            for pose_id, posecode in proposal["member_posecodes"].items():
                connection.execute(
                    "INSERT INTO pose_members VALUES(?,?,?,?,?)",
                    (
                        pose_id,
                        unit_id,
                        posecode["posecode_version"],
                        posecode["coordinate_profile"],
                        json.dumps(posecode["measurements"], ensure_ascii=False, sort_keys=True),
                    ),
                )
                for atom in posecode["observed_atoms"]:
                    connection.execute(
                        """INSERT INTO semantic_atoms(
                        semantic_unit_id,pose_id,scope,predicate,subject,relation,object,axis,value,
                        measure,measure_unit,bucket,polarity,evidence_state,provenance_json
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            unit_id, pose_id, "member", atom["predicate"], atom.get("subject"),
                            atom.get("relation"), atom.get("object"), atom.get("axis"), atom.get("value"),
                            atom.get("measure"), atom.get("measure_unit"), atom.get("bucket"), atom["polarity"],
                            atom["evidence_state"], json.dumps(atom["provenance"], ensure_ascii=False, sort_keys=True),
                        ),
                    )
            documents = [
                ("ko", "posecode_render", proposal["semantic"]["caption_ko"], "posecode_renderer"),
                ("en", "posecode_render", proposal["semantic"]["caption_en"], "posecode_renderer"),
            ]
            for index, alias in enumerate(proposal["semantic"].get("aliases_en", []), 1):
                documents.append(("en", "source_context", alias, proposal["source_clip_id"]))
            for index, (language, document_type, text, ref) in enumerate(documents, 1):
                document_id = f"{unit_id}:{document_type}:{language}:{index}"
                text_hash = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
                document_status = proposal["workflow_status"]
                if document_type == "source_context" and document_status != "rejected":
                    # Raw source names aid candidate recall but are not observed
                    # pose facts and must never inherit observed-tag approval.
                    document_status = "generated"
                provenance = {
                    "kind": document_type,
                    "ref": ref,
                    "version": 1,
                    "review_status": document_status,
                }
                values = (
                    document_id, unit_id, language, document_type, text, text_hash,
                    document_status, json.dumps(provenance, ensure_ascii=False, sort_keys=True),
                )
                connection.execute("INSERT INTO text_documents VALUES(?,?,?,?,?,?,?,?)", values)
                connection.execute("INSERT INTO text_documents_fts VALUES(?,?,?,?)", (document_id, unit_id, language, text))
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"review index integrity check failed: {integrity}")
        connection.close()
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_catalog(paths: CatalogPaths) -> dict[str, Any]:
    header, members = read_inventory(paths.inventory)
    actual_bvh = {path.stem for path in paths.bvh_dir.iterdir() if path.is_file() and path.suffix.lower() == ".bvh"}
    inventory_ids = {row["pose_id"] for row in members}
    if actual_bvh != inventory_ids:
        raise ValueError(
            f"inventory/BVH mismatch: missing={sorted(actual_bvh - inventory_ids)[:5]}, "
            f"stale={sorted(inventory_ids - actual_bvh)[:5]}"
        )

    cmu_records: dict[tuple[str, str], dict[str, Any]] = {}
    snapshot_ref = None
    captured_at = paths.cmu_catalog_captured_at
    if paths.cmu_catalog_html:
        cmu_records = parse_cmu_catalog(paths.cmu_catalog_html)
        snapshot_ref = _sha256_file(paths.cmu_catalog_html)
        captured_at = captured_at or date.today().isoformat()

    source_clips, pose_to_source = build_source_clips(
        members,
        raw_dir=paths.raw_dir,
        cmu_records=cmu_records,
        cmu_snapshot_ref=snapshot_ref,
        cmu_captured_at=captured_at,
    )
    exclusion_policy = apply_source_exclusions(source_clips, paths.exclusions_path)
    lineage_path = paths.output_dir / "pose_lineage.v1.jsonl"
    registry_path = paths.output_dir / "library_numbers.v1.json"
    lineage, library_number_registry = build_pose_lineage(
        members,
        source_clips,
        pose_to_source,
        bvh_dir=paths.bvh_dir,
        raw_dir=paths.raw_dir,
        existing_path=lineage_path,
        registry_path=registry_path,
    )
    proposals_path = paths.output_dir / "proposals.v1.jsonl"
    proposal_history, current, proposal_stats = build_proposals(
        members,
        source_clips,
        pose_to_source,
        bvh_dir=paths.bvh_dir,
        existing_path=proposals_path,
    )

    _atomic_write_jsonl(paths.output_dir / "source_clips.v1.jsonl", source_clips)
    _atomic_write_jsonl(lineage_path, lineage)
    _atomic_write_json(registry_path, library_number_registry)
    _atomic_write_jsonl(proposals_path, proposal_history)
    decisions_path = paths.output_dir / "decisions.v1.jsonl"
    if not decisions_path.exists():
        _atomic_write_jsonl(decisions_path, [])
    write_review_queues(paths.output_dir, current, source_clips, lineage)
    review_db = paths.output_dir / "tagging_review.v1.db"
    build_review_index(review_db, inventory_header=header, source_clips=source_clips, lineage=lineage, current=current)

    cmu_sources = [row for row in source_clips if row["provider"] == "cmu_graphics_lab"]
    local_sources = [row for row in source_clips if row["collection"]["id"] == "local_action_raw"]
    priority_blockers = sum(
        proposal_stats["review_priorities"].get(priority, 0) for priority in ("P0", "P1")
    )
    production_blockers = [
        "licenses_and_product_bvh_export_unresolved",
        "pinned_dense_embedding_not_built",
    ]
    if priority_blockers:
        production_blockers.insert(0, "priority_review_items_remaining")
    summary = {
        "artifact_type": "semantic_tagging_batch_summary",
        "schema_version": 1,
        "production_semantic_ready": False,
        "production_blockers": production_blockers,
        "input": {
            "inventory": _repo_path(paths.inventory),
            "bvh_dir": _repo_path(paths.bvh_dir),
            "pose_library_version": header["pose_library_version"],
            "pose_members": len(members),
            "semantic_units": len(current),
        },
        "provenance": {
            "source_clips": len(source_clips),
            "cmu_source_clips": len(cmu_sources),
            "cmu_catalog_matches": sum(row["catalog_evidence"]["status"] == "verified" for row in cmu_sources),
            "cmu_titles_recovered": sum(bool(row["original"].get("title")) for row in cmu_sources),
            "local_raw_source_clips": len(local_sources),
            "verified_local_frame_lineage": sum(row["verification"]["file_lineage_status"] == "verified" for row in lineage),
            "catalog_snapshot_ref": snapshot_ref,
            "excluded_source_clips": sum(_source_is_excluded(row) for row in source_clips),
            "exclusion_policy_id": exclusion_policy.get("policy_id") if exclusion_policy else None,
        },
        "tagging": proposal_stats,
        "outputs": {
            "source_clips": _repo_path(paths.output_dir / "source_clips.v1.jsonl"),
            "pose_lineage": _repo_path(lineage_path),
            "library_numbers": _repo_path(registry_path),
            "proposals": _repo_path(proposals_path),
            "decisions": _repo_path(decisions_path),
            "review_queue": _repo_path(paths.output_dir / "review_queue.csv"),
            "missing_action_names": _repo_path(paths.output_dir / "missing_action_names.csv"),
            "excluded_source_clips": _repo_path(paths.output_dir / "excluded_source_clips.csv"),
            "provenance_review_queue": _repo_path(paths.output_dir / "provenance_review_queue.csv"),
            "review_index": _repo_path(review_db),
        },
        "incremental_policy": {
            "library_no": "preserve_existing_append_next",
            "proposal_history": "append_only_by_run_key",
            "new_bvh_requires_inventory_regeneration": True,
        },
    }
    _atomic_write_json(paths.output_dir / "tagging-summary.v1.json", summary)
    return summary
