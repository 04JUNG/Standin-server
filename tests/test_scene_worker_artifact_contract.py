"""Real Engine build result contract; model/Blender work is a synthetic boundary."""

import base64
import json
import time
from types import SimpleNamespace

import pytest
from src.scene_refinement import layout
from src.scene_refinement.engine import Engine, MAGIC, digest


def test_plan_does_not_require_a_camera_before_final_artifact_exists(tmp_path, monkeypatch):
    import io
    from PIL import Image
    from src.scene_refinement import evidence, validation, plan

    image = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(image, format="PNG")
    raw = image.getvalue()
    library = tmp_path / "library" / "bvh"
    library.mkdir(parents=True)
    (library / "original.bvh").write_text("registered fixture")
    engine = object.__new__(Engine)
    engine.settings = SimpleNamespace(root=tmp_path / "runs", catalog=tmp_path / "catalog",
        library=library.parent, character=tmp_path / "rig", head_axes=tmp_path / "axes",
        character_sha256="a" * 64)
    engine.pending = {}
    engine.reader = engine.visual = None
    engine.stage = lambda *args: None
    monkeypatch.setattr(evidence, "run", lambda *args: None)
    monkeypatch.setattr(validation, "validate_evidence", lambda *args: None)
    selected = {"people": [{"person_index": 4, "pose_id": "original"}]}
    monkeypatch.setattr(plan, "run", lambda ctx: (ctx.work / "scene/selection.json").write_text(json.dumps(selected)))
    fbx = MAGIC + b"reviewed fixture"
    result = engine._prepare({"image_base64": base64.b64encode(raw).decode(), "image_sha256": digest(raw),
        "persons": [{"person_index": 4, "pose_id": "original", "output_scope": "full",
            "observation": {"box": [0, 0, 32, 32], "keypoints": [[5, 5]] * 17, "scores": [1] * 17},
            "fbx_base64": base64.b64encode(fbx).decode(), "fbx_sha256": digest(fbx)}]})
    assert result["people"] == selected["people"] and "camera" not in result
    assert result["plan_id"] in engine.pending


@pytest.mark.parametrize("artifact_only", [True, False])
def test_build_keeps_artifact_first_and_paired_preview_contracts(
    tmp_path, monkeypatch, artifact_only
):
    work = tmp_path / "work"
    folder = work / "scene"
    folder.mkdir(parents=True)
    (work / "inputs.json").write_text(json.dumps([{"cut_id": "scene"}]))
    ctx = SimpleNamespace(work=work)
    people = [{"person_index": 4, "pose_id": "original"}]
    engine = object.__new__(Engine)
    engine.settings = SimpleNamespace(ttl=1800, character_sha256="a" * 64)
    engine.pending = {
        "plan": (time.monotonic(), ctx, {"people": people}, {"image_sha256": "b" * 64})
    }
    engine.retired = {}
    fbx = MAGIC + b"contract-only-fixture"
    png = b"\x89PNG\r\n\x1a\n"
    from converter.rough_layout import camera_for
    from converter.scene_camera import framing_recipe
    recipe = framing_recipe(camera_for(1000, 768))
    monkeypatch.setattr(layout, "run", lambda ctx: None)

    def stage(ctx, *stages):
        if "scenes" not in stages:
            return
        assert (folder / "artifact-only").exists() == artifact_only
        (folder / "scene.fbx").write_bytes(fbx)
        (folder / "scene-result.json").write_text(json.dumps({"results": {"final": {"verification": {"framing": recipe}}}}))
        (folder / "layout.json").write_text(
            json.dumps(
                {
                    "final_overlap_pairs": [],
                    "depth_order": [],
                    "people": [{"scale": 1}],
                }
            )
        )
        if not artifact_only:
            (folder / "final.png").write_bytes(png)

    engine.stage = stage
    result = engine.build(
        {
            "plan_id": "plan",
            "artifact_only": artifact_only,
            "persons": [
                {
                    "person_index": 4,
                    "pose_id": "original",
                    "character_sha256": "a" * 64,
                    "fbx_base64": base64.b64encode(fbx).decode(),
                    "fbx_sha256": digest(fbx),
                }
            ],
        }
    )
    assert result["version"] == "scene-refinement-v4"
    assert result["person_count"] == 1 and result["people"] == people
    assert result["artifact_sha256"] == digest(base64.b64decode(result["fbx_base64"]))
    if artifact_only:
        assert result["artifact_contract"] == "verified-scene-fbx-v1"
        assert result["preview_base64"] == ""
        assert result["preview_sha256"] == digest(b"")
    else:
        assert "artifact_contract" not in result
        assert base64.b64decode(result["preview_base64"]) == png
    assert result["camera"]["artifact_sha256"] == result["artifact_sha256"]
    assert result["camera"]["recipe"] == recipe
    assert not engine.pending
