from dataclasses import replace
import pytest
from converter import scene_catalog as catalog
from converter.scene_contact_profiles import get_profile


def test_contact_profile_is_bound_to_geometry_and_mirrors_without_changing_policy(
    monkeypatch,
):
    digest = "a" * 64
    recipe = catalog.SceneRecipe("repair", digest, "phone:R", "", "pending")
    monkeypatch.setattr(catalog, "load_catalog", lambda: {digest: recipe})
    original = catalog.recipe_token(digest, allow_pending=True)
    recipe = replace(recipe, contact_profile="surface")
    assert catalog.recipe_token(digest, allow_pending=True) != original
    assert recipe.mirrored().contact_profile == "surface"
    assert recipe.mirrored().prop_guides == "phone:L"
    assert recipe.mirrored().mirrored() == recipe


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown scene contact profile"):
        get_profile("unbounded-automatic-approval")


def test_family_change_does_not_invalidate_unrelated_scenes(monkeypatch):
    from pathlib import Path

    examples = {
        "a" * 64: catalog.SceneRecipe("body", "a" * 64, "sword:R", "", "pending"),
        "b" * 64: catalog.SceneRecipe(
            "cloth", "b" * 64, "towel:B", "", "pending", contact_profile="surface"
        ),
        "c" * 64: catalog.SceneRecipe(
            "phone", "c" * 64, "phone:R", "", "pending", contact_profile="surface"
        ),
    }
    monkeypatch.setattr(catalog, "load_catalog", lambda: examples)
    before = {k: catalog.recipe_token(k, allow_pending=True) for k in examples}
    original = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda p: original(p) + (b"\n" if p.name == "scene_contact_fabric.py" else b""),
    )
    after = {k: catalog.recipe_token(k, allow_pending=True) for k in examples}
    assert before["a" * 64] == after["a" * 64]
    assert before["c" * 64] == after["c" * 64]
    assert before["b" * 64] != after["b" * 64]


def test_contact_receipt_must_prove_bounds_and_nonarm_preservation(monkeypatch):
    digest = "a" * 64
    recipe = catalog.SceneRecipe(
        "repair", digest, "phone:R", "", "accepted", contact_profile="surface"
    )
    monkeypatch.setattr(catalog, "load_catalog", lambda: {digest: recipe})
    receipt = dict(
        version=catalog.VERSION,
        status="baked",
        source_bvh_sha256=digest,
        catalog_revision=catalog.revision(),
        recipe_token=catalog.recipe_token(digest),
        pose_id="repair",
        recipe_review_status="accepted",
        quality=recipe.quality,
        mirrored=False,
        body_bones_preserved=False,
        contact_certified=False,
        applied_meshes=1,
        meshes=[{}],
        contact_adjustment=dict(
            applied=True,
            profile="surface",
            source_bvh_unchanged=True,
            visual_approval=False,
            wrist_movement_max=0.06,
            torso_head_legs_matrix_max_error=0,
            local_finger_matrix_max_error=0,
            hands=[dict(wrist_bend=35, displacement=0.06)],
        ),
    )
    catalog.validate_receipt(receipt, digest)
    for field, value in (
        ("wrist_movement_max", 0.25),
        ("torso_head_legs_matrix_max_error", 0.01),
        ("local_finger_matrix_max_error", float("nan")),
        ("visual_approval", True),
    ):
        bad = {
            **receipt,
            "contact_adjustment": {**receipt["contact_adjustment"], field: value},
        }
        with pytest.raises(ValueError):
            catalog.validate_receipt(bad, digest)
