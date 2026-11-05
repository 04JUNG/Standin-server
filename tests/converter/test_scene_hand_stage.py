from converter.scene_catalog import SceneRecipe
from converter import scene_hand_stage as stage


def test_defaults_skip_source_transfer_and_never_double_bake_selected_side(monkeypatch):
    recipe = SceneRecipe(
        "phone",
        "a" * 64,
        "phone:R",
        "",
        "accepted",
        hand_defaults=(("right", "hold_phone"), ("left", "hold_cup")),
    )
    monkeypatch.setattr(stage, "lookup", lambda *args, **kwargs: recipe)
    calls = []
    from converter import source_fingers, hands

    def source(path, bvh, **options):
        calls.append(("source", options["hands"]))
        return dict(status="preset_override")

    def bake(path, selected):
        calls.append(("preset", dict(selected)))
        return dict(selected)

    monkeypatch.setattr(source_fingers, "bake_source_fingers", source)
    monkeypatch.setattr(hands, "bake_hands", bake)
    result = stage.apply_hand_stage(
        "result.fbx", "original.bvh", "a" * 64, selected={"right": "point"}
    )
    assert calls == [
        ("source", {"left": "hold_cup", "right": "point"}),
        ("preset", {"right": "point"}),
        ("preset", {"left": "hold_cup"}),
    ]
    assert result["hands"] == {"right": "point"}
    assert result["scene_hand_defaults"] == {"left": "hold_cup"}


def test_rifle_default_mirrors_and_explicit_choice_wins(monkeypatch):
    recipe = SceneRecipe(
        "rifle", "a" * 64, "rifle:B", "", "accepted", grip_correction="rifle_support"
    )
    monkeypatch.setattr(stage, "lookup", lambda *args, **kwargs: recipe)
    assert stage.default_hands("a" * 64) == {"left": "hold_cup"}
    assert stage.default_hands("a" * 64, mirror=True) == {"right": "hold_cup"}
    assert stage.default_hands("a" * 64, {"left": "fist"}) == {}
    assert stage.default_hands("a" * 64, exact=True) == {}


def test_ordinary_body_build_uses_unchanged_source_wrapper(monkeypatch):
    from converter import source_fingers

    monkeypatch.setattr(stage, "lookup", lambda *args, **kwargs: None)
    sentinel = object()
    received = []
    monkeypatch.setattr(
        source_fingers,
        "convert_source_pose",
        lambda **options: (received.append(options), sentinel)[1],
    )
    options = dict(
        bvh_path="source.bvh", out_path="final.fbx", character_fbx="body.fbx"
    )
    assert stage.convert_scene_pose("a" * 64, **options) is sentinel
    assert received == [options]
