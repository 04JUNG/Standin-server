"""Content binding and review gates protect production from stale recipes."""

from dataclasses import replace
import json

import pytest

from converter import scene_catalog as catalog
from converter.scene_props import bake_scene_props


def test_retained_catalog_has_no_ambiguous_or_unsupported_recipe():
    rows = catalog.load_catalog()
    assert len(rows) == 170
    assert sum(bool(r.prop_guides) for r in rows.values()) == 150
    assert all(r.quality == "primitive-placement-guide" for r in rows.values())


@pytest.mark.parametrize(
    "spec",
    [
        "sword:R:body_up",
        "cup:L",
        "bow:L;arrow:R",
        "cup:Q",
        "cup:R:else",
        "/tmp/model:R",
        "sword:R:aim:extra",
    ],
)
def test_scene_spec_requires_registered_kinds_grips_and_axes(spec):
    valid = spec in ("sword:R:body_up", "cup:L")
    if valid:
        assert catalog.validate_spec(spec) == spec
    else:
        with pytest.raises(ValueError):
            catalog.validate_spec(spec)


def test_pending_recipe_is_available_only_to_explicit_recovery_jobs(monkeypatch):
    digest = "a" * 64
    recipe = catalog.SceneRecipe("pending-example", digest, "cup:R", "", "pending")
    monkeypatch.setattr(catalog, "load_catalog", lambda: {digest: recipe})
    assert catalog.lookup(digest) is None
    assert catalog.lookup(digest, allow_pending=True) == recipe
    assert catalog.lookup("0" * 64, allow_pending=True) is None


def test_no_recipe_and_exact_bypass_never_read_or_modify_output(tmp_path):
    p = tmp_path / "untouched.fbx"
    p.write_bytes(b"unchanged")
    assert bake_scene_props(p, "0" * 64)["status"] == "no_reviewed_recipe"
    assert (
        bake_scene_props("missing.fbx", "bad", force_exact_v324=True)["status"]
        == "exact_v324_bypass"
    )
    assert p.read_bytes() == b"unchanged"


def test_mirroring_swaps_independent_attachments_and_is_involutive():
    r = catalog.SceneRecipe(
        "example", "a" * 64, "sword:R:body_up;shield:L;spear:B", "seat", "accepted"
    )
    assert r.mirrored().prop_guides == "sword:L:body_up;shield:R;spear:B"
    assert r.mirrored().mirrored() == r


def test_thumb_correction_mirrors_and_is_part_of_the_geometry_key(monkeypatch):
    source, recipe = next(iter(catalog.load_catalog().items()))
    recipe = replace(recipe, thumb_correction_sides=(), thumb_correction_overrides=())
    monkeypatch.setattr(catalog, "load_catalog", lambda: {source: recipe})
    before = catalog.recipe_token(source, allow_pending=True)
    adjusted = replace(recipe, thumb_correction_sides=("left",))
    monkeypatch.setattr(catalog, "load_catalog", lambda: {source: adjusted})
    assert catalog.recipe_token(source, allow_pending=True) != before
    assert adjusted.mirrored().thumb_correction_sides == ("right",)
    assert adjusted.mirrored().mirrored() == adjusted


def test_thumb_adjustment_never_opens_file_when_user_or_scene_preset_wins():
    from converter.scene_thumb_adjustment import bake

    assert bake("missing.fbx", ("left",), selected={"left": "open"})["applied"] is False
    assert (
        bake("missing.fbx", ("right",), defaults={"right": "hold_phone"})["applied"]
        is False
    )


def test_excluded_recipes_are_never_reactivated_by_pending_option(monkeypatch):
    r = catalog.SceneRecipe("example", "a" * 64, "cup:R", "", "excluded")
    monkeypatch.setattr(catalog, "load_catalog", lambda: {"a" * 64: r})
    assert catalog.lookup("a" * 64, allow_pending=True) is None


def test_accepted_recipe_applies_only_to_the_exact_bvh_digest(monkeypatch):
    r = catalog.SceneRecipe("example", "a" * 64, "cup:R", "", "accepted")
    monkeypatch.setattr(catalog, "load_catalog", lambda: {"a" * 64: r})
    assert catalog.lookup("a" * 64) == r
    assert catalog.lookup("b" * 64) is None


@pytest.mark.parametrize("filename", ["scene_guides.py", "scene_placement.py"])
def test_revised_scene_geometry_invalidates_model_cache(monkeypatch, filename):
    from pathlib import Path
    from converter.preview_model import model_revision, model_key

    source = next(iter(catalog.load_catalog()))
    before = model_key(source, "b" * 64, include_pending_scene=True)
    untouched = model_key("a" * 64, "b" * 64)
    original = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda p: original(p) + (b"\n" if p.name == filename else b""),
    )
    assert model_key(source, "b" * 64, include_pending_scene=True) != before
    assert model_key("a" * 64, "b" * 64) == untouched


def test_scene_approval_reuses_reviewed_geometry_key(monkeypatch):
    source, pending = next(iter(catalog.load_catalog().items()))
    from converter.preview_model import model_key

    preview = model_key(source, "b" * 64, include_pending_scene=True)
    monkeypatch.setattr(
        catalog,
        "load_catalog",
        lambda: {source: replace(pending, review_status="accepted")},
    )
    assert model_key(source, "b" * 64) == preview


def test_grip_correction_is_part_of_geometry_identity(monkeypatch):
    source, recipe = next(iter(catalog.load_catalog().items()))
    monkeypatch.setattr(catalog, "load_catalog", lambda: {source: recipe})
    before = catalog.recipe_token(source, allow_pending=True)
    monkeypatch.setattr(
        catalog,
        "load_catalog",
        lambda: {source: replace(recipe, grip_correction="two_hand_shaft")},
    )
    assert catalog.recipe_token(source, allow_pending=True) != before


def test_correction_receipt_cannot_claim_unchanged_body_or_unbounded_wrist(monkeypatch):
    digest = "a" * 64
    recipe = catalog.SceneRecipe(
        "test", digest, "sword:B", "", "accepted", grip_correction="two_hand_shaft"
    )
    monkeypatch.setattr(catalog, "load_catalog", lambda: {digest: recipe})
    value = dict(
        version=catalog.VERSION,
        status="baked",
        source_bvh_sha256=digest,
        catalog_revision=catalog.revision(),
        recipe_token=catalog.recipe_token(digest),
        pose_id="test",
        recipe_review_status="accepted",
        quality=recipe.quality,
        mirrored=False,
        body_bones_preserved=False,
        contact_certified=False,
        applied_meshes=1,
        meshes=[{}],
        grip_alignment=dict(
            applied=True,
            policy="two_hand_shaft",
            torso_head_legs_matrix_max_error=0,
            local_finger_matrix_max_error=0,
            hands={
                s: dict(wrist_after_deg=38, joint_length_max_error=0)
                for s in ("Left", "Right")
            },
        ),
    )
    catalog.validate_receipt(value, digest)
    with pytest.raises(ValueError):
        catalog.validate_receipt({**value, "body_bones_preserved": True}, digest)
    value["grip_alignment"]["hands"]["Left"]["wrist_after_deg"] = 65
    with pytest.raises(ValueError, match="grip correction"):
        catalog.validate_receipt(value, digest)
    for invalid in (float("nan"), float("inf"), None, "38", True, -1):
        value["grip_alignment"]["hands"]["Left"]["wrist_after_deg"] = invalid
        with pytest.raises(ValueError, match="grip correction"):
            catalog.validate_receipt(value, digest)


def test_scene_failure_does_not_leave_partial_hand_or_arm_edits(tmp_path, monkeypatch):
    from converter import scene_props

    source = "a" * 64
    recipe = catalog.SceneRecipe("test", source, "sword:B", "", "accepted")
    monkeypatch.setattr(scene_props, "lookup", lambda *args, **kwargs: recipe)
    path = tmp_path / "original.fbx"
    path.write_bytes(b"original body and fingers")

    def fail(staged, *args, **kwargs):
        staged.write_bytes(b"partially corrected")
        raise ValueError("unsafe placement")

    monkeypatch.setattr(scene_props, "_bake_scene_props", fail)
    with pytest.raises(ValueError, match="unsafe placement"):
        scene_props.bake_scene_props(path, source)
    assert path.read_bytes() == b"original body and fingers"
    assert list(tmp_path.iterdir()) == [path]


def test_registered_hand_defaults_mirror_and_change_geometry_identity(monkeypatch):
    source, original = next(iter(catalog.load_catalog().items()))
    recipe = replace(original, hand_defaults=(("right", "hold_phone"),))
    assert recipe.mirrored().hand_defaults == (("left", "hold_phone"),)
    assert recipe.mirrored().mirrored() == recipe
    before = catalog.recipe_token(source, allow_pending=True)
    monkeypatch.setattr(catalog, "load_catalog", lambda: {source: recipe})
    assert catalog.recipe_token(source, allow_pending=True) != before
