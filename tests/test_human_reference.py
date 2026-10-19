import io
import json
from pathlib import Path
import pytest
from PIL import Image
from src.human_reference import (
    METHOD,
    reference_identity,
    read_template,
    template_manifest,
)
from src.pose_posters import read_poster, digest, headers


def test_reference_poster_binds_template_source_scope_method_and_png(tmp_path):
    sha = "a" * 64
    scope = "half"
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), "gray").save(buffer, format="PNG")
    png = buffer.getvalue()
    meta = {
        **reference_identity(sha, scope),
        "image_sha256": digest(png),
        "bytes": len(png),
    }
    path = tmp_path / (sha + "__" + scope + ".json")
    path.write_text(json.dumps(meta))
    (tmp_path / (sha + "__" + scope + ".png")).write_bytes(png)
    data, actual = read_poster(tmp_path, sha, scope)
    assert (
        data == png
        and actual["kind"] == "default-body"
        and actual["pose_approximate"] is True
    )
    assert headers(actual)["X-Standin-Preview-Method"] == METHOD
    for field in [
        "source_bvh_sha256",
        "scope",
        "template_sha256",
        "renderer_revision",
        "character_sha256",
    ]:
        path.write_text(json.dumps({**meta, field: "changed"}))
        with pytest.raises(ValueError):
            read_poster(tmp_path, sha, scope)


def test_template_read_requires_pinned_compressed_asset_and_scope(
    tmp_path, monkeypatch
):
    import src.human_reference as h

    packed = b"\x1f\x8btest"
    (tmp_path / "full.json.gz").write_bytes(packed)
    manifest = template_manifest()
    manifest["scopes"]["full"].update(bytes=len(packed), sha256=digest(packed))
    monkeypatch.setattr(h, "template_manifest", lambda: manifest)
    data, response_headers = read_template(tmp_path, "full")
    assert data == packed and response_headers["X-Standin-Preview-Only"] == "true"
    assert (
        response_headers["Content-Type"] == "application/gzip"
        and "Content-Encoding" not in response_headers
    )
    (tmp_path / "full.json.gz").write_bytes(b"changed")
    with pytest.raises(ValueError):
        read_template(tmp_path, "full")
    with pytest.raises(ValueError):
        read_template(tmp_path, "../full")


def test_template_api_missing_assets_is_explicit_and_never_runs_converter(
    tmp_path, monkeypatch
):
    from fastapi.testclient import TestClient
    import api.app as app

    monkeypatch.setattr(app.CFG, "data_dir", str(tmp_path))
    response = TestClient(app.app).get("/pose-poster-template/full")
    assert response.status_code == 404
