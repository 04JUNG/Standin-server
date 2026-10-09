"""A review model must describe the exact same source, framing and exported FBX."""
from dataclasses import replace
import hashlib
import json
import struct

import pytest

from converter.framed_model import MODEL_VERSION, metadata, revision
from converter.worker import JobValidationError, load_job
from tests.converter.test_worker_contract import _job
from tests.converter.test_framing_contract import FramedRunner, post
from tests.converter.test_api_contract import _client


class ModelRunner(FramedRunner):
    bad_field = None

    def convert(self, **kwargs):
        result = super().convert(**kwargs)
        assert kwargs["preview_model"] is True and kwargs["preview_view"] is None
        meta = {
            "version": MODEL_VERSION, "revision": revision(), "scope": kwargs["output_scope"],
            "camera_rotation": kwargs.get("camera_rotation"),
            "base_fbx_sha256": result.artifact_sha256,
            "source_bvh_sha256": result.source_bvh_sha256,
            "character_id": kwargs["character_id"], "character_sha256": kwargs["character_sha256"],
        }
        if self.bad_field:
            meta[self.bad_field] = "bad"
        encoded = json.dumps({"asset": {"extras": meta}}).encode()
        encoded += b" " * (-len(encoded) % 4)
        model = (struct.pack("<5I", 0x46546C67, 2, 32 + len(encoded), len(encoded), 0x4E4F534A)
                 + encoded + struct.pack("<II", 4, 0x004E4942) + b"\0" * 4)
        return replace(result, preview=b"", preview_model=model)


@pytest.mark.parametrize("scope", ["full", "half", "bust", "head"])
def test_model_pair_uses_no_png(tmp_path, scope):
    response = post(_client(tmp_path, runner=ModelRunner()), preview_format="model", output_scope=scope)
    assert response.status_code == 200, response.text
    value = response.json()
    import base64
    data = base64.b64decode(value["preview_base64"])
    assert metadata(data)["scope"] == scope
    assert value["preview_sha256"] == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("field", ["version", "revision", "scope", "camera_rotation", "base_fbx_sha256",
                                  "source_bvh_sha256", "character_id", "character_sha256"])
def test_reject_mismatched_model(tmp_path, field):
    runner = ModelRunner()
    runner.bad_field = field
    assert post(_client(tmp_path, runner=runner), preview_format="model").status_code == 500


def test_model_camera_job_needs_no_png_but_preserves_exact_mode_guard(tmp_path):
    path, value = _job(tmp_path)
    value.update(preview_model=True, camera_rotation=[[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    path.write_text(json.dumps(value))
    assert load_job(path)["preview_model"] is True
    value["force_exact_v324"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(JobValidationError):
        load_job(path)


def test_capability_pins_model_exporter_and_character(tmp_path):
    health = _client(tmp_path).get("/healthz").json()
    assert health["preview_model_revision"] == revision()
    assert health["character_hashes"]["standin-master-v2"] == hashlib.sha256(b"character").hexdigest()


def test_container_includes_review_exporter_and_smokes_real_http_pair():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    assert "COPY converter/framed_model.py /app/converter/framed_model.py" in (root / "Dockerfile.converter").read_text(encoding="utf-8")
    workflow = (root / ".github/workflows/converter-ci.yml").read_text(encoding="utf-8")
    assert "--form preview_format=model" in workflow
    assert "--review-model" in workflow
