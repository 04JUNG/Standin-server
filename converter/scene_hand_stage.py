"""Resolve hand precedence before baking source fingers into the rest skeleton.

Selected and scene-default sides skip source transfer and receive exactly one
preset bake. Never apply a default on top of an already baked source hand.
"""

from converter.scene_catalog import lookup


def default_hands(
    source_sha, selected=None, *, mirror=False, allow_pending=False, exact=False
):
    recipe = (
        None
        if exact
        else lookup(source_sha, mirror=mirror, allow_pending=allow_pending)
    )
    if recipe is None:
        return {}
    defaults = dict(recipe.hand_defaults)
    if recipe.grip_correction == "rifle_support":
        defaults.setdefault("right" if mirror else "left", "hold_cup")
    return {
        side: preset
        for side, preset in defaults.items()
        if not (selected or {}).get(side)
    }


def apply_hand_stage(
    path,
    bvh_path,
    source_sha,
    *,
    selected=None,
    frame=0,
    mirror=False,
    allow_pending=False,
    exact=False,
):
    from converter.source_fingers import bake_source_fingers
    from converter.hands import bake_hands

    defaults = default_hands(
        source_sha, selected, mirror=mirror, allow_pending=allow_pending, exact=exact
    )
    effective = {**defaults, **{s: p for s, p in (selected or {}).items() if p}}
    source = bake_source_fingers(
        path,
        bvh_path,
        frame=frame,
        mirror=mirror,
        hands=effective,
        force_exact_v324=exact,
    )
    # Disjoint sides keep the public user-selection receipt unchanged.
    user_receipt = bake_hands(path, selected) if selected else None
    default_receipt = bake_hands(path, defaults) if defaults else None
    return dict(
        source_fingers=source,
        hands=user_receipt,
        scene_hand_defaults=defaults,
        scene_hand_bake=default_receipt,
    )


def convert_scene_pose(source_sha, *, allow_pending=False, **options):
    if options.get("output_mode", "rigged_rest") != "rigged_rest":
        raise ValueError("scene finger baking requires rigged_rest")
    from converter.source_fingers import convert_source_pose, SourcePoseReport

    defaults = default_hands(
        source_sha,
        mirror=options.get("mirror", False),
        allow_pending=allow_pending,
        exact=options.get("force_exact_v324", False),
    )
    if not defaults:
        return convert_source_pose(**options)
    from converter.convert import convert

    body = convert(**options)
    if not body.ok:
        return body
    stage = apply_hand_stage(
        options["out_path"],
        options["bvh_path"],
        source_sha,
        frame=options.get("frame", 0),
        mirror=options.get("mirror", False),
        allow_pending=allow_pending,
        exact=options.get("force_exact_v324", False),
    )

    class ScenePoseReport(SourcePoseReport):
        def as_dict(self):
            return {
                **super().as_dict(),
                "scene_hand_defaults": stage["scene_hand_defaults"],
                "scene_hand_bake": stage["scene_hand_bake"],
            }

    return ScenePoseReport(body, stage["source_fingers"])
