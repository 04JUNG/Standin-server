"""Scope optional contact repair cache invalidation to the routines it executes."""

import hashlib
from pathlib import Path


def revision(recipe):
    if not recipe.contact_profile:
        return ""
    names = {
        "scene_contact_revision.py",
        "scene_contact_profiles.py",
        "scene_contact_pose.py",
        "scene_contact_ik.py",
        "scene_contact_frames.py",
        "scene_contact_guides.py",
        "scene_hand_body_review.py",
        "hand_contact.py",
    }
    kinds = {s.split(":")[0] for s in recipe.prop_guides.split(";") if s}
    if "lantern" in kinds:
        names.add("scene_contact_hanging.py")
    if kinds & {"pistol"}:
        names.add("scene_contact_firearm.py")
    if kinds & {"spoon", "knife", "comb", "key", "bow"}:
        names.add("scene_contact_tools.py")
    if kinds & {"towel", "scarf", "handkerchief", "rope"}:
        names.add("scene_contact_fabric.py")
    if kinds & {
        "phone",
        "book",
        "letter",
        "photo",
        "tray",
        "plate",
        "cake",
        "ringbox",
        "boulder",
        "orb",
        "gift",
        "pillow",
        "laundry",
        "teddy",
        "crate",
        "bag",
        "crown",
        "bouquet",
    }:
        names.add("scene_contact_carry.py")
    if kinds & {
        "gem",
        "goblet",
        "crown",
        "bouquet",
        "pickaxe",
        "key",
    } or recipe.contact_profile in {"offered-blade", "bow-inspection"}:
        names.add("scene_contact_special.py")
    root = Path(__file__).parent
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes().replace(b"\r\n", b"\n") for name in sorted(names)
        )
    ).hexdigest()
