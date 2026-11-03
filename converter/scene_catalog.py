"""Content-bound, versioned scene recipes. Safe to import without Blender.

The catalog holds procedural placement guides, not production object assets.
Only visually reviewed recipes are used by default. Recovery jobs can select
pending recipes explicitly without granting them production approval.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re

VERSION = "scene-guides-v1"
CATALOG = Path(__file__).with_name("scene_library.v1.json")
GUIDE_KINDS = frozenset(
    "sword greatsword dagger spear staff broom mop axe hammer pickaxe wand torch umbrella cup goblet potion jar watering_can phone book letter photo scroll shield tray plate lunchbox cake gift ringbox crate chest basket bag pouch pillow laundry teddy orb gem boulder crown pen spoon brush comb knife key flower bouquet towel scarf handkerchief rope pistol rifle crossbow bow barrier dryer lantern".split()
)
AXES = frozenset("up down forward body_up body_down body_forward aim grip_line".split())


def validate_spec(value: str) -> str:
    if not isinstance(value, str) or len(value) > 500:
        raise ValueError("invalid prop guide specification")
    for item in value.split(";") if value else []:
        parts = item.split(":")
        if (
            len(parts) not in (2, 3)
            or parts[0] not in GUIDE_KINDS
            or parts[1] not in ("L", "R", "B")
            or len(parts) == 3
            and parts[2] not in AXES
        ):
            raise ValueError("unknown prop kind, grip or orientation: " + item)
    return value


@dataclass(frozen=True)
class SceneRecipe:
    pose_id: str
    source_bvh_sha256: str
    prop_guides: str
    support: str
    review_status: str
    quality: str = "primitive-placement-guide"
    grip_correction: str = ""
    hand_defaults: tuple[tuple[str, str], ...] = ()
    bow_state: str = "drawn"
    thumb_correction_sides: tuple[str, ...] = ()
    thumb_correction_overrides: tuple[tuple[str, tuple[float, float]], ...] = ()
    contact_profile: str = ""

    def mirrored(self):
        parts = []
        for item in self.prop_guides.split(";") if self.prop_guides else []:
            fields = item.split(":")
            fields[1] = {"L": "R", "R": "L", "B": "B"}[fields[1]]
            parts.append(":".join(fields))
        return replace(
            self,
            prop_guides=";".join(parts),
            hand_defaults=tuple(
                ({"left": "right", "right": "left"}[side], preset)
                for side, preset in self.hand_defaults
            ),
            thumb_correction_sides=tuple(
                {"left": "right", "right": "left"}[side]
                for side in self.thumb_correction_sides
            ),
            thumb_correction_overrides=tuple(
                ({"left": "right", "right": "left"}[side], angles)
                for side, angles in self.thumb_correction_overrides
            ),
        )


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, SceneRecipe]:
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("version") != VERSION:
        raise ValueError("unsupported scene catalog version")
    result = {}
    for row in payload["scenes"]:
        digest = row["source_bvh_sha256"]
        if not re.fullmatch("[a-f0-9]{64}", digest):
            raise ValueError("scene source must be a lowercase SHA256")
        recipe = SceneRecipe(
            row["pose_id"],
            digest,
            validate_spec(row["prop_guides"]),
            row["support"],
            row["review_status"],
            grip_correction=row.get("grip_correction", ""),
            hand_defaults=tuple(sorted(row.get("hand_defaults", {}).items())),
            bow_state=row.get("bow_state", "drawn"),
            contact_profile=row.get("contact_profile", ""),
            thumb_correction_sides=tuple(row.get("thumb_correction_sides", [])),
            thumb_correction_overrides=tuple(
                (side, tuple(angles))
                for side, angles in sorted(
                    row.get("thumb_correction_overrides", {}).items()
                )
            ),
        )
        if recipe.support not in ("", "seat") or recipe.review_status not in (
            "pending",
            "accepted",
            "excluded",
        ):
            raise ValueError("invalid scene support or review status")
        if recipe.grip_correction not in ("", "two_hand_shaft", "rifle_support"):
            raise ValueError("unknown scene grip correction")
        if recipe.contact_profile:
            from converter.scene_contact_profiles import get_profile

            get_profile(recipe.contact_profile)
        if recipe.bow_state not in ("drawn", "undrawn"):
            raise ValueError("unknown bow state")
        if len(set(recipe.thumb_correction_sides)) != len(
            recipe.thumb_correction_sides
        ) or any(
            side not in {"left", "right"} for side in recipe.thumb_correction_sides
        ):
            raise ValueError("invalid thumb correction sides")
        if any(
            side not in recipe.thumb_correction_sides
            or len(angles) != 2
            or any(
                type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 40
                for x in angles
            )
            for side, angles in recipe.thumb_correction_overrides
        ):
            raise ValueError("invalid bounded thumb correction angles")
        if recipe.hand_defaults:
            from converter.hand_library import load_library

            library = load_library()
            if any(
                side not in {"left", "right"} or preset not in library.presets
                for side, preset in recipe.hand_defaults
            ):
                raise ValueError("unknown scene default hand preset")
        if digest in result:
            raise ValueError("ambiguous scene recipes for one BVH")
        result[digest] = recipe
    return result


def lookup(digest: str, *, mirror=False, allow_pending=False) -> SceneRecipe | None:
    recipe = load_catalog().get(digest)
    allowed = {"accepted", "pending"} if allow_pending else {"accepted"}
    if recipe is None or recipe.review_status not in allowed:
        return None
    return recipe.mirrored() if mirror else recipe


def revision() -> str:
    root = Path(__file__).parent
    names = (
        "scene_catalog.py",
        "scene_props.py",
        "scene_guides.py",
        "scene_placement.py",
        "scene_alignment.py",
        "scene_firearms.py",
        "scene_primitives.py",
        "scene_clearance.py",
        "scene_containers.py",
        "scene_surfaces.py",
        "scene_phone.py",
        "scene_support.py",
        "scene_hand_stage.py",
        "scene_thumb_adjustment.py",
    )
    return hashlib.sha256(
        b"".join((root / name).read_bytes().replace(b"\r\n", b"\n") for name in names)
    ).hexdigest()


def recipe_token(source_sha: str, *, allow_pending=False) -> str:
    """Only a changed pose's geometry invalidates its preview cache.

    Review approval does not change geometry. Offline recovery can precompute
    pending scenes under their final geometry key; runtime still selects only
    accepted scenes. Excluding a scene returns the separate body-only key.
    """
    recipe = lookup(source_sha, allow_pending=allow_pending)
    if recipe is None:
        return ""
    geometry = {
        "source_bvh_sha256": source_sha,
        "prop_guides": recipe.prop_guides,
        "support": recipe.support,
        "grip_correction": recipe.grip_correction,
        "hand_defaults": dict(recipe.hand_defaults),
        "bow_state": recipe.bow_state,
        "contact_profile": recipe.contact_profile,
        "thumb_correction_sides": recipe.thumb_correction_sides,
        "thumb_correction_overrides": recipe.thumb_correction_overrides,
        "algorithm_revision": revision(),
    }
    if recipe.contact_profile:
        from converter.scene_contact_revision import revision as contact_revision

        geometry["contact_revision"] = contact_revision(recipe)
    if (
        recipe.hand_defaults
        or recipe.grip_correction == "rifle_support"
        or recipe.thumb_correction_sides
    ):
        # A different default preset must never reuse an old posed mesh.
        root = Path(__file__).parent
        geometry["default_hand_revision"] = hashlib.sha256(
            b"".join(
                (root / name).read_bytes().replace(b"\r\n", b"\n")
                for name in ("hands.py", "hand_geometry.py", "hand_library.v1.json")
            )
        ).hexdigest()
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


def validate_receipt(value, source_sha, *, mirror=False, exact=False):
    """Validate the worker's production recipe selection independently of bpy."""
    recipe = None if exact else lookup(source_sha, mirror=mirror)
    if recipe is None:
        if value is not None and (
            not isinstance(value, dict)
            or value.get("status")
            not in (
                "no_reviewed_recipe",
                "exact_v324_bypass",
            )
        ):
            raise ValueError("worker applied an unregistered or unreviewed scene")
        return
    expected = {
        "version": VERSION,
        "status": "baked",
        "source_bvh_sha256": source_sha,
        "catalog_revision": revision(),
        "recipe_token": recipe_token(source_sha),
        "pose_id": recipe.pose_id,
        "recipe_review_status": "accepted",
        "quality": recipe.quality,
        "mirrored": mirror,
        "body_bones_preserved": not bool(
            recipe.grip_correction or recipe.contact_profile
        ),
        "contact_certified": False,
    }
    if (
        not isinstance(value, dict)
        or any(
            type(value.get(k)) is not type(v) or value.get(k) != v
            for k, v in expected.items()
        )
        or type(value.get("applied_meshes")) is not int
        or value["applied_meshes"] <= 0
        or not isinstance(value.get("meshes"), list)
        or len(value["meshes"]) != value["applied_meshes"]
    ):
        raise ValueError("scene recipe binding or output receipt mismatch")
    defaults = value.get("scene_hand_defaults", {})
    registered_defaults = dict(recipe.hand_defaults)
    contact = value.get("contact_adjustment", {})
    if recipe.contact_profile or contact.get("applied"):

        def contact_bound(number, limit):
            return (
                type(number) in (int, float)
                and math.isfinite(number)
                and 0 <= number <= limit
            )

        if (
            contact.get("applied") is not True
            or contact.get("profile") != recipe.contact_profile
            or contact.get("source_bvh_unchanged") is not True
            or contact.get("visual_approval") is not False
            or not contact_bound(contact.get("torso_head_legs_matrix_max_error"), 1e-5)
            or not contact_bound(contact.get("local_finger_matrix_max_error"), 1e-5)
            or not contact_bound(contact.get("wrist_movement_max"), 0.18)
            or not isinstance(contact.get("hands"), list)
            or any(
                not isinstance(h, dict)
                or not contact_bound(h.get("wrist_bend", h.get("wrist_after_deg")), 42)
                for h in contact.get("hands", [])
            )
        ):
            raise ValueError("invalid bounded contact correction receipt")
    if recipe.grip_correction == "rifle_support":
        registered_defaults.setdefault("right" if mirror else "left", "hold_cup")
    if not isinstance(defaults, dict) or any(
        registered_defaults.get(side) != preset for side, preset in defaults.items()
    ):
        raise ValueError("unregistered scene hand default")
    thumb = value.get("thumb_adjustment", {})
    if recipe.thumb_correction_sides or thumb.get("applied"):
        from converter.scene_thumb_adjustment import PROFILE, ANGLES

        if (
            not isinstance(thumb, dict)
            or thumb.get("profile") != PROFILE
            or type(thumb.get("applied")) is not bool
            or not isinstance(thumb.get("sides"), dict)
            or thumb["applied"] != bool(thumb["sides"])
        ):
            raise ValueError("invalid scene thumb correction receipt")
        for side, adjustment in thumb["sides"].items():
            if (
                side not in recipe.thumb_correction_sides
                or side in defaults
                or adjustment
                != dict(
                    zip(
                        ("spread_deg", "lift_deg"),
                        dict(recipe.thumb_correction_overrides).get(side, ANGLES[side]),
                    )
                )
            ):
                raise ValueError("unregistered thumb adjustment")
        if thumb["applied"]:
            error = thumb.get("body_and_other_digits_matrix_max_error")
            if (
                type(error) not in (float, int)
                or not math.isfinite(error)
                or not 0 <= error <= 1e-5
                or thumb.get("source_bvh_unchanged") is not True
            ):
                raise ValueError("thumb adjustment changed other bones")
    if recipe.grip_correction:
        alignment = value.get("grip_alignment", {})

        def bounded(number, limit):
            return (
                type(number) in (int, float)
                and math.isfinite(number)
                and 0 <= number <= limit
            )

        if (
            not isinstance(alignment, dict)
            or alignment.get("applied") is not True
            or alignment.get("policy") != recipe.grip_correction
            or not bounded(alignment.get("torso_head_legs_matrix_max_error"), 1e-6)
            or not bounded(alignment.get("local_finger_matrix_max_error"), 1e-5)
            or not isinstance(alignment.get("hands"), dict)
            or set(alignment.get("hands", {})) != {"Left", "Right"}
            or any(
                not isinstance(hand, dict)
                or not bounded(hand.get("wrist_after_deg"), 42)
                or not bounded(hand.get("joint_length_max_error"), 1e-5)
                for hand in alignment.get("hands", {}).values()
            )
        ):
            raise ValueError("scene grip correction receipt is invalid")
