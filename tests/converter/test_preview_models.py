import hashlib
import json

import pytest

from converter.preview_model import MODEL_VERSION, model_key, model_revision
from converter_api.preview_models import PreviewModelStore


def publish(root, source="a" * 64, character="b" * 64, data=b"glTFtest"):
    key = model_key(source, character)
    path = root / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".glb").write_bytes(data)
    manifest = {"version": MODEL_VERSION, "revision": model_revision(),
                "source_bvh_sha256": source, "character_sha256": character,
                "coordinates": "Y-up-hips-origin", "scope": "full",
                "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    path.with_suffix(".json").write_text(json.dumps(manifest))
    return path


def test_cache_reuses_verified_bytes_but_not_another_character(tmp_path):
    path = publish(tmp_path)
    store = PreviewModelStore(str(tmp_path))
    assert store.get("a"*64, "b"*64)[0] == b"glTFtest"
    path.with_suffix(".glb").unlink()
    assert store.get("a"*64, "b"*64)[0] == b"glTFtest"
    with pytest.raises(FileNotFoundError):
        store.get("a"*64, "c"*64)


@pytest.mark.parametrize("kind", ["corrupt", "stale", "source", "character"])
def test_bad_assets_fail_closed(tmp_path, kind):
    path = publish(tmp_path)
    if kind == "corrupt":
        path.with_suffix(".glb").write_bytes(b"glTFbad!")
    else:
        manifest = json.loads(path.with_suffix(".json").read_text())
        field = {"stale": "revision", "source": "source_bvh_sha256", "character": "character_sha256"}[kind]
        manifest[field] = "c" * 64
        path.with_suffix(".json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        PreviewModelStore(str(tmp_path)).get("a"*64, "b"*64)


def test_cache_is_bounded_and_never_bakes_missing_models(tmp_path):
    first = publish(tmp_path)
    publish(tmp_path, "c"*64)
    store = PreviewModelStore(str(tmp_path), max_bytes=8)
    store.get("a"*64, "b"*64)
    store.get("c"*64, "b"*64)
    assert len(store._cache) == 1
    first.with_suffix(".glb").unlink()
    with pytest.raises(FileNotFoundError):
        store.get("a"*64, "b"*64)
    with pytest.raises(ValueError):
        store.get("../escape", "b"*64)


def test_endpoint_does_not_start_blender(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from converter_api.app import create_app

    publish(tmp_path)
    monkeypatch.setenv("POSE_PREVIEW_URI", str(tmp_path))
    registry = SimpleNamespace(metadata=lambda _: SimpleNamespace(sha256="b"*64))
    # An inert object is intentional: no health, render or conversion method exists.
    app = create_app(registry=registry, runner=object())
    with TestClient(app) as client:
        response = client.get("/pose-preview/" + "a"*64)
        assert response.status_code == 200
        assert response.content == b"glTFtest"
        assert response.headers["content-type"] == "model/gltf-binary"
        assert client.get("/pose-preview/" + "c"*64).status_code == 404
