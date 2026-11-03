"""Bake retained scene guides into the same skinned FBX used by all previews.

Rigid one-bone skinning lets existing anatomical cropping, camera rotation,
FBX validation and GLB baking handle props through their normal mesh paths.
The body solver and original BVH stay unchanged. Explicit scene policies may
correct arm alignment or default finger grips; receipts identify those changes.
"""

from __future__ import annotations

from pathlib import Path
from converter.scene_catalog import VERSION, lookup, revision, recipe_token


def bake_scene_props(path, source_sha, **options):
    """Commit the entire scene stage atomically, including optional hand/arm work."""
    if (
        options.get("force_exact_v324")
        or lookup(
            source_sha,
            mirror=options.get("mirror", False),
            allow_pending=options.get("allow_pending", False),
        )
        is None
    ):
        return _bake_scene_props(path, source_sha, **options)
    import shutil
    from tempfile import TemporaryDirectory

    path = Path(path)
    with TemporaryDirectory(prefix="scene-", dir=path.parent) as folder:
        staged = Path(folder) / path.name
        shutil.copy2(path, staged)
        result = _bake_scene_props(staged, source_sha, **options)
        staged.replace(path)
        return result


def _bake_scene_props(
    path,
    source_sha,
    *,
    mirror=False,
    allow_pending=False,
    force_exact_v324=False,
    selected_hands=None,
    prepared_hand_defaults=None,
):
    receipt = {"version": VERSION, "applied_meshes": 0, "body_bones_preserved": True}
    if force_exact_v324:
        return {**receipt, "status": "exact_v324_bypass"}
    recipe = lookup(source_sha, mirror=mirror, allow_pending=allow_pending)
    if recipe is None:
        return {**receipt, "status": "no_reviewed_recipe"}
    import bpy
    from converter import retarget as rt
    from converter.scene_guides import add_guides
    from converter.scene_alignment import bake_alignment

    from converter.scene_hand_stage import default_hands

    defaults = default_hands(
        source_sha, selected_hands, mirror=mirror, allow_pending=allow_pending
    )
    if (prepared_hand_defaults or {}) != defaults:
        raise ValueError(
            "scene hand defaults must be baked once before source-hand rest promotion"
        )
    from converter.scene_thumb_adjustment import bake as bake_thumb

    contact_adjustment = {"applied": False, "profile": "none"}
    if recipe.contact_profile:
        from converter.scene_contact_pose import bake as bake_contact

        contact_adjustment = bake_contact(
            path,
            recipe.contact_profile,
            recipe.prop_guides,
            mirror=mirror,
            selected_hands=selected_hands,
        )

    thumb_adjustment = bake_thumb(
        path,
        recipe.thumb_correction_sides,
        selected=selected_hands,
        defaults=defaults,
        overrides=dict(recipe.thumb_correction_overrides),
    )
    alignment = bake_alignment(
        path,
        recipe.grip_correction,
        mirror=mirror,
        selected_hands=selected_hands,
        prepared_hand_defaults=defaults,
    )

    rt.reset_scene()
    arm, _ = rt.import_character(str(path))
    before = {b.name: b.matrix.copy() for b in arm.pose.bones}
    prefixes = [
        b.name[: -len("LeftHand")]
        for b in arm.data.bones
        if b.name.endswith("LeftHand")
    ]
    if len(prefixes) != 1:
        raise ValueError("scene guides require one identifiable humanoid hand rig")
    prefix = prefixes[0]
    if prefix != "mixamorig:":
        raise ValueError("scene guide adapter requires a Mixamo target rig")
    created = []
    placements = []
    specs = recipe.prop_guides.split(";") if recipe.prop_guides else []
    for spec in specs + ([""] if recipe.support else []):
        original = set(bpy.data.objects)
        support = recipe.support if not spec else ""
        options = {"bow_state": recipe.bow_state} if recipe.bow_state != "drawn" else {}
        if recipe.contact_profile:
            from converter.scene_contact_guides import add_guides as repaired_guides

            placements.append(
                repaired_guides(
                    arm,
                    spec,
                    support,
                    profile=recipe.contact_profile,
                    mirror=mirror,
                    **options,
                )
            )
        else:
            placements.append(add_guides(arm, spec, support, mirror=mirror, **options))
        bpy.context.view_layer.update()
        added = [o for o in bpy.data.objects if o not in original and o.type == "MESH"]
        grip = spec.split(":")[1] if spec else None
        anchor = prefix + (
            {
                "L": "LeftHand",
                "R": "RightHand",
                "B": "LeftHand" if mirror else "RightHand",
            }[grip]
            if grip
            else "Hips"
        )
        if anchor not in arm.data.bones:
            raise ValueError("scene attachment bone missing: " + anchor)
        for obj in added:
            obj.name = "StandinScene__" + obj.name.replace(" ", "_")
            matrix = obj.matrix_world.copy()
            obj.parent = arm
            obj.matrix_world = matrix
            group = obj.vertex_groups.new(name=anchor)
            group.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")
            modifier = obj.modifiers.new(name="Scene rigid binding", type="ARMATURE")
            modifier.object = arm
            obj["standin_scene_guide"] = True
            created.append(
                {
                    "name": obj.name,
                    "anchor": anchor,
                    "vertices": len(obj.data.vertices),
                    "specification": spec or support,
                }
            )
    bpy.context.view_layer.update()
    error = max(
        abs(before[b.name][r][c] - b.matrix[r][c])
        for b in arm.pose.bones
        for r in range(4)
        for c in range(4)
    )
    if error > 1e-6:
        raise ValueError("scene baking changed body or finger transforms")
    if not created and (recipe.prop_guides or recipe.support):
        raise ValueError("nonempty scene recipe produced no meshes")
    if created:
        temporary = Path(path).with_name(Path(path).stem + ".scene.fbx")
        rt.export_fbx(str(temporary), embed_textures=False)
        temporary.replace(path)
    return {
        **receipt,
        "body_bones_preserved": not (
            alignment["applied"] or contact_adjustment["applied"]
        ),
        "contact_adjustment": contact_adjustment,
        "grip_alignment": alignment,
        "scene_hand_defaults": defaults,
        "thumb_adjustment": thumb_adjustment,
        "status": "baked",
        "source_bvh_sha256": source_sha,
        "catalog_revision": revision(),
        "recipe_token": recipe_token(source_sha, allow_pending=allow_pending),
        "pose_id": recipe.pose_id,
        "recipe_review_status": recipe.review_status,
        "quality": recipe.quality,
        "mirrored": mirror,
        "applied_meshes": len(created),
        "meshes": created,
        "placement_diagnostics": placements,
        "contact_certified": False,
        "body_matrix_max_error": error,
    }
