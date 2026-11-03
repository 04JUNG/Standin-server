import hashlib
import sys
from types import SimpleNamespace

import pytest

from converter import source_fingers as fingers
from converter.preview_model import model_key, model_revision


@pytest.mark.parametrize("prefix", ["", "mixamorig:"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_resolves_complete_mixamo_chains_without_leaf_markers(prefix, side):
    label = side.title()
    names = [prefix + label + "Hand"] + [
        f"{prefix}{label}Hand{finger.title()}{n}"
        for finger in fingers.FINGERS for n in range(1, 5)]
    wrist, chain = fingers.bone_names(names, side)
    assert wrist == prefix + label + "Hand"
    assert len(chain) == 15
    assert all(not n.endswith("4") for n in chain.values())


def test_unreal_fingers_with_mixamo_body_and_opposite_hand_names():
    names = ["LeftHand", "RightHand"] + [
        f"{finger}_{joint:02d}_{side}" for finger in fingers.FINGERS
        for joint in range(1, 4) for side in ("l", "r")]
    wrist, chain = fingers.bone_names(names, "right")
    assert wrist == "RightHand"
    assert len(chain) == 15
    assert chain["index", 1] == "index_01_r"


def test_style100_wrist_alias_keeps_its_complete_finger_chain():
    names = ["LeftWrist"] + [f"LeftHand{f.title()}{n}"
                              for f in fingers.FINGERS for n in range(1, 4)]
    wrist, chain = fingers.bone_names(names, "left")
    assert wrist == "LeftWrist" and len(chain) == 15


def test_partial_cmu_data_is_not_misrepresented_as_a_complete_hand():
    wrist, chain = fingers.bone_names(
        ["LeftHand", "LeftFingerBase", "LeftHandIndex1", "LThumb"], "left")
    assert wrist == "LeftHand"
    assert chain == {("index", 1): "LeftHandIndex1"}


def test_source_stage_can_be_skipped_without_touching_the_fbx(tmp_path, monkeypatch):
    # Bypass paths must not import, re-export or even read the artifact/source.
    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mathutils", SimpleNamespace(Matrix=None))
    monkeypatch.setitem(sys.modules, "converter.retarget", SimpleNamespace())
    artifact = tmp_path / "out.fbx"
    artifact.write_bytes(b"unchanged exact output")
    before = hashlib.sha256(artifact.read_bytes()).hexdigest()
    for options, status in [({"force_exact_v324": True}, "exact_v324_bypass"),
        ({"hands": {"left": "fist", "right": "open"}}, "preset_override")]:
        receipt = fingers.bake_source_fingers(artifact, "missing.bvh", **options)
        assert receipt["status"] == status
        assert receipt["applied_bones"] == 0
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == before


def test_batch_conversion_uses_same_source_stage_and_preserves_body_report(monkeypatch):
    body = SimpleNamespace(ok=True, mapped_bones=22,
        as_dict=lambda: {"ok": True, "mapped_bones": 22, "airborne_plantar": {"safe": True}})
    calls = []
    monkeypatch.setitem(sys.modules, "converter.convert", SimpleNamespace(
        convert=lambda **kw: calls.append(("body", kw)) or body))
    monkeypatch.setattr(fingers, "bake_source_fingers", lambda *a, **kw:
        calls.append(("fingers", a, kw)) or {"applied_bones": 30})
    report = fingers.convert_source_pose(bvh_path="source.bvh", character_fbx="body.fbx",
        out_path="out.fbx", mirror=True, frame=2)
    assert [c[0] for c in calls] == ["body", "fingers"]
    assert calls[1][1] == ("out.fbx", "source.bvh")
    assert calls[1][2]["mirror"] is True and calls[1][2]["frame"] == 2
    assert report.ok and report.mapped_bones == 22
    assert report.as_dict() == {**body.as_dict(), "source_fingers": {"applied_bones": 30}}


def test_failed_body_conversion_never_bakes_hands(monkeypatch):
    failed = SimpleNamespace(ok=False)
    monkeypatch.setitem(sys.modules, "converter.convert", SimpleNamespace(convert=lambda **kw: failed))
    monkeypatch.setattr(fingers, "bake_source_fingers", lambda *a, **kw: pytest.fail("body failed"))
    assert fingers.convert_source_pose(out_path="missing.fbx") is failed


def test_batch_rejects_modes_where_the_rig_cannot_be_baked():
    with pytest.raises(ValueError, match="rigged_rest"):
        fingers.convert_source_pose(output_mode="static_mesh")


def test_old_precomputed_models_are_invalidated_when_finger_stage_changes(monkeypatch):
    from pathlib import Path
    original = Path.read_bytes
    before = model_revision(), model_key("a" * 64, "b" * 64)
    monkeypatch.setattr(Path, "read_bytes", lambda p:
        original(p) + (b"\n# changed finger transport" if p.name == "source_fingers.py" else b""))
    assert model_revision() != before[0]
    assert model_key("a" * 64, "b" * 64) != before[1]
