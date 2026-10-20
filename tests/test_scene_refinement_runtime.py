"""Save-path boundaries, without mocking pose accuracy or running a paid model."""

import base64, hashlib, json
from pathlib import Path
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from src.scene_refinement.api import create_app
from src.scene_refinement.engine import decode, fbx, MAGIC
from src.scene_refinement.validation import validate_evidence


def test_private_api_requires_operator_token_and_valid_stage():
    class Engine:
        def prepare(self, request):
            return {"fresh": request["image"]}

        def build(self, request):
            raise ValueError("rejected")

    with pytest.raises(ValueError):
        create_app(Engine(), "short")
    client = TestClient(create_app(Engine(), "x" * 32))
    headers = {"Authorization": "Bearer " + "x" * 32}
    assert client.post("/scene/plan", json={"image": "raw"}).status_code == 401
    assert client.post(
        "/scene/plan", headers=headers, json={"image": "raw"}
    ).json() == {"fresh": "raw"}
    assert client.post("/scene/build", headers=headers, json={}).status_code == 422
    assert client.post("/scene/path", headers=headers, json={}).status_code == 404


def test_artifact_gate_rejects_size_type_and_substitution():
    data = MAGIC + b"fbx"
    encoded = base64.b64encode(data).decode()
    sha = hashlib.sha256(data).hexdigest()
    assert fbx(encoded, sha) == data
    for value, limit in [("bad!", 32), (encoded, 2), ("", 32)]:
        with pytest.raises((ValueError, TypeError)):
            decode(value, limit)
    with pytest.raises(ValueError):
        fbx(encoded, "0" * 64)


def test_unknown_people_nonfinite_boxes_and_relations_are_not_authority(tmp_path):
    (tmp_path / "inputs.json").write_text(json.dumps([{"cut_id": "cut"}]))
    folder = tmp_path / "cut"
    folder.mkdir()
    surface = {
        "person_index": 7,
        "torso_surface": "front",
        "torso_heading": "front",
        "head_heading": "front",
        "head_bbox": [1, 2, 20, 30],
    }

    def write(name, value):
        (folder / name).write_text(json.dumps(value))

    write("surfaces-review.json", {"analysis": {"people": [surface]}})
    write("roles-review.json", {"analysis": {"relations": []}})
    write("person_7-analysis.json", {"analysis": {"person_index": 7}})
    write("localized-hands.json", {"hands": []})
    ctx = SimpleNamespace(work=tmp_path)
    validate_evidence(ctx, [7])
    write("roles-review.json", {"analysis": {"relations": [{"actor": 7, "target": 8}]}})
    with pytest.raises(ValueError):
        validate_evidence(ctx, [7])
    write("roles-review.json", {"analysis": {"relations": []}})
    surface["head_bbox"][0] = float("nan")
    write("surfaces-review.json", {"analysis": {"people": [surface]}})
    with pytest.raises(ValueError):
        validate_evidence(ctx, [7])


def test_completed_or_failed_builds_release_preparation_slots(tmp_path):
    from src.scene_refinement.engine import Engine

    engine = object.__new__(Engine)
    engine.settings = SimpleNamespace(ttl=60)
    engine.retired = {}
    ctx = SimpleNamespace(work=tmp_path / "work")
    ctx.work.mkdir()
    engine.pending = {"plan": (0, ctx, None, None)}
    engine._build = lambda request: {"same_file": True}
    assert engine.build({"plan_id": "plan"}) == {"same_file": True}
    assert not engine.pending and tmp_path in engine.retired
    engine.pending = {"failure": (0, ctx, None, None)}

    def failed(request):
        raise ValueError("reject")

    engine._build = failed
    with pytest.raises(ValueError):
        engine.build({"plan_id": "failure"})
    assert not engine.pending


def test_fresh_reader_has_one_bounded_transient_retry_and_no_quota_retry(monkeypatch):
    from src.scene_refinement.evidence import Reader
    from src.scene_refinement import evidence

    monkeypatch.setattr(evidence.time, "sleep", lambda seconds: None)

    class UpstreamError(Exception):
        def __init__(self, code):
            self.code = code

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise UpstreamError(504)
        return SimpleNamespace(
            text='```json\n[{"person_index":7,"head_bbox":{"left":1,"top":2,"right":3,"bottom":4}}]\n```'
        )

    reader = object.__new__(Reader)
    reader.provider = "gemini"
    reader.model_name = "configured"
    reader.adapter = SimpleNamespace(
        _client=SimpleNamespace(models=SimpleNamespace(generate_content=generate)),
        _to_part=lambda path: "image",
    )
    result = reader.ask("image", "prompt")
    assert result["analysis"]["people"][0]["head_bbox"] == [1, 2, 3, 4]
    assert len(calls) == 2 and calls[0]["config"].http_options.timeout <= 60000
    calls.clear()

    def quota(**kwargs):
        calls.append(kwargs)
        raise UpstreamError(429)

    reader.adapter._client.models.generate_content = quota
    with pytest.raises(UpstreamError):
        reader.ask("image", "prompt")
    assert len(calls) == 1


def test_incomplete_separation_or_reversed_depth_cannot_be_released():
    from src.scene_refinement.validation import validate_layout

    def report(pixels=0, reversed=False, scale=1.0):
        return {
            "final_overlap_pairs": [{"pixels": pixels}],
            "depth_order": [{"reversed": reversed}],
            "people": [{"scale": scale}],
        }

    validate_layout(report())
    for value in (report(pixels=1), report(reversed=True), report(scale=0.5)):
        with pytest.raises(ValueError):
            validate_layout(value)
