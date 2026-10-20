"""Build the candidate search index from the current, reviewable BVH revisions."""
from pathlib import Path

from src.bvh import load_coco17
from src.library import build_entries_from_pose
from src.repo import build_db

from .storage import contained_path


def build_candidates(poses: list[dict], batch_dir: Path, batch: str) -> int:
    entries = []
    for pose in poses:
        bvh = contained_path(batch_dir, pose["bvh"])
        kp, scores = load_coco17(str(bvh))
        action = {"FW": "walking", "FR": "running"}.get(pose["movement"], "other")
        generated = build_entries_from_pose(pose["pose_id"], kp,
            {"shot": "full_half", "action": action,
             "relationship": pose.get("relationship", "solo")}, str(bvh), scores)
        for entry in generated:
            entry.meta = {"source": pose["source"], "license": pose["license"], "author": pose["author"],
                          "source_url": pose["source_url"], "source_frame_0based": pose["source_frame_0based"],
                          "pose_family_id": pose.get("pose_family_id", pose["pose_id"].removesuffix("_mirror")), "batch_id": batch, "review_status": "pending"}
            for field in ("set_id", "set_role"):
                if pose.get(field):
                    entry.meta[field] = pose[field]
            if pose.get("composition_variant"):
                entry.meta["composition_variant"] = pose["composition_variant"]
            if "hand_augmentation" in pose:
                entry.meta["hand_augmentation"] = pose["hand_augmentation"]
            for field in ('category', 'category_label', 'style', 'scenario', 'prop_guides', 'support'):
                if field in pose:
                    entry.meta[field] = pose[field]
        entries.extend(generated)
    return build_db(entries, str(batch_dir / "candidates.db"))
