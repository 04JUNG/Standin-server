"""BVH exporter whitespace compatibility regressions."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.bvh import parse_bvh, write_single_frame_bvh  # noqa: E402
from src.semantic_catalog import build_pose_lineage, build_source_clips  # noqa: E402
from scripts.init_bvh_tag_inventory import _filename_evidence  # noqa: E402
from scripts.mirror_bvh import _counterpart_name  # noqa: E402


def test_parse_bvh_accepts_closing_brace_joined_to_motion_header() -> None:
    payload = """HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation
}MOTION
Frames: 1
Frame Time: 0.033333
0 0 0 0 0 0
"""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "joined-motion-header.bvh"
        output = Path(directory) / "rewritten.bvh"
        path.write_text(payload, encoding="utf-8")
        joints, frames = parse_bvh(str(path))
        write_single_frame_bvh(str(path), frames[0], str(output))
        rewritten_joints, rewritten_frames = parse_bvh(str(output))

    assert len(joints) == 1
    assert joints[0][0] == "Hips"
    assert frames.shape == (1, 6)
    assert len(rewritten_joints) == 1
    assert rewritten_frames.shape == (1, 6)


def test_mirror_counterpart_supports_side_suffix_rigs() -> None:
    assert _counterpart_name("index_01_l") == "index_01_r"
    assert _counterpart_name("ball_leaf_r") == "ball_leaf_l"
    assert _counterpart_name("LeftForeArm") == "RightForeArm"


def test_intake_filename_parser_preserves_clip_sample_and_frame() -> None:
    evidence = _filename_evidence("UAL2__NinjaJump_Start__p03_f0042_mirror")
    assert evidence["pattern"] == "intake_clip_sample_frame"
    assert evidence["provider_hint"] == "quaternius"
    assert evidence["native_clip_id_hint"] == "NinjaJump_Start"
    assert evidence["selected_frame_index_hint"] == 42
    assert evidence["sample_ordinal_hint"] == 3
    assert evidence["selection_kind_hint"] == "p"


def test_intake_manifest_drives_source_and_lineage_without_inventing_tags() -> None:
    pose_id = "UAL1__Dance_Loop__p02_f0016"
    source = {
        "source_clip_id": "ual1:Dance_Loop",
        "provider": "quaternius",
        "collection_id": "universal_animation_library_1_standard",
        "native_clip_id": "Dance_Loop",
        "original_title": "Dance_Loop",
        "original_filename": "UAL1_Standard.fbx",
        "original_path": "data/ual1-keyposes/bvh/Dance_Loop__p02_f0016.bvh",
        "source_sha256": None,
        "source_url": None,
        "license_id": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "product_bvh_export": "yes",
        "attribution_required": False,
    }
    intake = {
        "record_type": "pose_intake_member",
        "batch_id": "test-intake",
        "source": source,
        "extraction": {
            "selected_frame_index": 16,
            "frame_index_base": "one",
            "sample_ordinal": 2,
            "selection_kind": "p",
        },
        "derivation": {"operations": [], "parent_pose_id": None},
    }
    member = {
        "pose_id": pose_id,
        "bvh": {"sha256": "sha256:test"},
        "filename_evidence": _filename_evidence(pose_id),
        "intake_evidence": intake,
        "provenance_refs": {"source_clip_id": source["source_clip_id"]},
        "grouping": {
            "variant": {"kind": "original", "mirror_of": None},
        },
    }
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source_clips, pose_to_source = build_source_clips(
            [member],
            raw_dir=root,
            cmu_records={},
            cmu_snapshot_ref=None,
            cmu_captured_at=None,
        )
        lineage, _ = build_pose_lineage(
            [member],
            source_clips,
            pose_to_source,
            bvh_dir=root,
            raw_dir=root,
            existing_path=root / "pose_lineage.v1.jsonl",
            registry_path=root / "library_numbers.v1.json",
        )

    assert source_clips[0]["provider"] == "quaternius"
    assert source_clips[0]["license_ref"]["id"] == "CC0-1.0"
    assert source_clips[0]["original"]["title"] == "Dance_Loop"
    assert lineage[0]["source_clip_id"] == "ual1:Dance_Loop"
    assert lineage[0]["extraction"]["selected_frame_index"] == 16
    assert lineage[0]["verification"]["file_lineage_status"] == "intake_manifest_verified"


if __name__ == "__main__":
    test_parse_bvh_accepts_closing_brace_joined_to_motion_header()
    test_mirror_counterpart_supports_side_suffix_rigs()
    test_intake_filename_parser_preserves_clip_sample_and_frame()
    test_intake_manifest_drives_source_and_lineage_without_inventing_tags()
    print("4/4 passed")
