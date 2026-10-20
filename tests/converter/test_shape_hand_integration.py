"""Combined multipart/job boundary; real mesh QA remains a Blender probe."""
from dataclasses import replace
import hashlib
import json

import pytest

from converter.hand_library import load_library
from converter.shape_spec import resolve_shape
from converter.worker import load_job
from tests.converter.test_api_contract import _client
from tests.converter.test_framing_contract import FramedRunner, post
from tests.converter.test_shape_bake_contract import envelope, write_registry
from tests.converter.test_worker_contract import _job


class ShapeHandRunner(FramedRunner):
    def convert(self, **kwargs):
        result = super().convert(**kwargs)
        recipe, hands = kwargs["shape_recipe"], kwargs["hands"]
        return replace(result, report={**result.report,
            "shape_bake": {"shape_sha256": recipe["shape_sha256"],
                           "rules": recipe["rules"], "controls": recipe["values"]},
            "hands": {**hands, "library_sha256": load_library().sha256,
                      "body_bones_preserved": True},
        })


def test_framed_request_preserves_shape_source_and_both_hand_choices(tmp_path, monkeypatch):
    runner = ShapeHandRunner()
    client = _client(tmp_path, runner=runner)
    digest = hashlib.sha256(b"character").hexdigest()
    registry = write_registry(tmp_path, monkeypatch, asset_sha=digest)
    source = next(iter(registry["sources"].values()))
    expected = {"rules": "standin-body-9.9.9", "blendSha256": source["blend_sha256"],
                "controllerSha256": source["controller_sha256"]}
    shape = envelope({"muscle": 1}, asset_sha=digest)
    response = post(client, shape=json.dumps(shape), expected_shape_source=json.dumps(expected),
                    hand_left="fist", hand_right="relaxed")
    assert response.status_code == 200, response.text
    call = runner.calls[0]
    assert call["hands"] == {"left": "fist", "right": "relaxed"}
    assert call["shape_recipe"]["shape_sha256"] == shape["shapeSha256"]
    assert call["shape_recipe"]["controller_sha256"] == expected["controllerSha256"]
    assert response.json()["shape_sha256"] == shape["shapeSha256"]
    assert response.json()["hands"]["left"] == "fist"


@pytest.mark.parametrize("bad_field", ["shape", "hand_left"])
def test_invalid_combined_request_never_starts_worker(tmp_path, monkeypatch, bad_field):
    runner = ShapeHandRunner()
    client = _client(tmp_path, runner=runner)
    digest = hashlib.sha256(b"character").hexdigest()
    write_registry(tmp_path, monkeypatch, asset_sha=digest)
    fields = {"shape": json.dumps(envelope({"muscle": 1}, asset_sha=digest)),
              "hand_left": "fist", "hand_right": "relaxed"}
    fields[bad_field] = "invalid"
    assert post(client, **fields).status_code == 400
    assert runner.calls == []


def test_worker_accepts_shape_and_hands_together(tmp_path, monkeypatch):
    path, job = _job(tmp_path)
    write_registry(tmp_path, monkeypatch, asset_sha=job["character_sha256"])
    recipe = resolve_shape("standin-master-v2", job["character_sha256"])
    job.update(shape_recipe=recipe, hands={"left": "fist", "right": "relaxed"})
    path.write_text(json.dumps(job))
    normalized = load_job(path)
    assert normalized["shape_recipe"] == recipe
    assert normalized["hands"] == job["hands"]


def test_worker_preserves_shape_hands_and_skin_screen_together(tmp_path, monkeypatch):
    path, job = _job(tmp_path)
    write_registry(tmp_path, monkeypatch, asset_sha=job['character_sha256'])
    recipe = resolve_shape('standin-master-v2', job['character_sha256'])
    job.update(shape_recipe=recipe, hands={'left': 'fist', 'right': 'relaxed'},
               hand_contact_screen=True)
    path.write_text(json.dumps(job))
    normalized = load_job(path)
    assert normalized['shape_recipe'] == recipe
    assert normalized['hands'] == job['hands']
    assert normalized['hand_contact_screen'] is True
