"""Real Engine build result contract; model/Blender work is a synthetic boundary."""

import base64
import json
import time
from types import SimpleNamespace

import pytest
from src.scene_refinement import layout
from src.scene_refinement.engine import Engine, MAGIC, digest


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
    monkeypatch.setattr(layout, "run", lambda ctx: None)

    def stage(ctx, *stages):
        if "scenes" not in stages:
            return
        assert (folder / "artifact-only").exists() == artifact_only
        (folder / "scene.fbx").write_bytes(fbx)
        (folder / "scene-result.json").write_text("{}")
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
    assert not engine.pending
