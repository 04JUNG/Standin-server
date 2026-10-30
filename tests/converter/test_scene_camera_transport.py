import base64
import hashlib
import json
from pathlib import Path

from converter.rough_layout import camera_for
from converter.scene_camera import framing_recipe
from converter.scene_camera_transport import camera_binding, camera_header
from converter_api.scene_runner import SceneResult
from tests.converter.test_scene_contract import FBX, FakeRunner, client, manifest


def test_pinned_wire_fixture_is_generated_from_actual_camera_math():
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/scene-camera-artifact-v1.json").read_text())
    sha = hashlib.sha256(FBX).hexdigest()
    recipe = framing_recipe(camera_for(1000, 768))
    assert fixture == camera_binding(recipe, sha)
    assert json.loads(base64.b64decode(camera_header(recipe, sha))) == fixture


def test_binary_composition_transports_the_verified_camera_bound_to_fbx_not_zip():
    class Runner(FakeRunner):
        def compose_scene(self, **kwargs):
            return SceneResult(FBX, {"framing": framing_recipe(camera_for(1000, 768))})
    response = client(Runner()).post("/compose-scene",
        data={"manifest": json.dumps(manifest()), "response_format": "fbx"},
        files=[("fbx", ("person.fbx", FBX))] * 2)
    assert response.status_code == 200 and response.content == FBX
    camera = json.loads(base64.b64decode(response.headers["x-standin-scene-camera"]))
    assert camera["artifact_sha256"] == hashlib.sha256(response.content).hexdigest()
    assert camera["recipe"]["source_dimensions"] == [1000, 768]
    assert camera["recipe"]["automatic_import_framing_verified"] is False
