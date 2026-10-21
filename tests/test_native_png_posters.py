"""The existing app/BFF PNG protocol remains unchanged for native references."""
import hashlib
import io
import json

from PIL import Image
import pytest

from src import native_png_posters as native


def asset(root, scope="full", source=b"public source BVH"):
    sha = hashlib.sha256(source).hexdigest()
    directory = root / "thumbs/scope8"
    directory.mkdir(parents=True, exist_ok=True)
    pixels = Image.new("RGB", (512, 512), (60, 65, 75))
    pixels.paste((200, 205, 215), (200, 32, 310, 480))
    buf = io.BytesIO()
    pixels.save(buf, format="JPEG", quality=90)
    jpeg = buf.getvalue()
    meta = dict(schema_version=2, render_method=native.METHOD, source_bvh_sha256=sha,
        character_id="standin-master-v2", character_sha256=native.CHARACTER_SHA,
        renderer_revision=sorted(native.REVISIONS)[0], scope=scope, view="yaw_000",
        azimuth_degrees=0, native_camera_yaw=180, pitch_degrees=0,
        projection="orthographic", projection_matches_search=False, selected_body=False,
        hands_applied=False, source_frame=0, preview_only=True, anatomy_approval=False,
        width=512, height=512, format="jpeg", quality=90,
        image_sha256=hashlib.sha256(jpeg).hexdigest(), bytes=len(jpeg),
        style="approved-anatomy-detail-dark-5scope-8view-v1",
        solver_manifest_sha256=native.SOLVER_SHA, border_pixels=32,
        framing=dict(version="skin-regions-v2-shoulder-head", scope=scope,
                     vertices_before=100, vertices_after=100 if scope == "full" else 50,
                     bones_preserved=52, cap_faces=0))
    stem = directory / (sha + "__" + scope + "__yaw_000")
    stem.with_suffix(".jpg").write_bytes(jpeg)
    stem.with_suffix(".json").write_text(json.dumps(meta))
    manifest = dict(library_version="lib-20261021-21b1b3d9",
        directional_thumbnails=dict(status="complete", render_method=native.METHOD,
            renderer_revisions=sorted(native.REVISIONS), output_scopes=list(native.SCOPES),
            views=["yaw_000"]))
    (root / "library_manifest.json").write_text(json.dumps(manifest))
    return sha, jpeg, meta, stem


@pytest.mark.parametrize("scope", native.SCOPES)
@pytest.mark.parametrize("revision", sorted(native.REVISIONS))
def test_png_protocol_exact_decoded_pixels_and_real_provenance(tmp_path, scope, revision):
    source, jpeg, meta, stem = asset(tmp_path, scope)
    meta["renderer_revision"] = revision
    metadata = stem.with_suffix(".json")
    metadata.write_text(json.dumps(meta))
    original = metadata.read_bytes()
    png, headers = native.read_native_png(tmp_path, source, scope)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(png)) as image, Image.open(io.BytesIO(jpeg)) as old:
        assert image.size == (512, 512) and image.convert("RGB").tobytes() == old.convert("RGB").tobytes()
    assert headers["X-Standin-Preview-Method"] == "frozen-retarget-v1"
    assert headers["X-Standin-Native-Render-Method"] == native.METHOD
    assert headers["X-Standin-Preview-Revision"] == revision
    assert headers["X-Standin-Artifact-SHA256"] == hashlib.sha256(png).hexdigest()
    assert headers["X-Standin-Native-Artifact-SHA256"] == hashlib.sha256(jpeg).hexdigest()
    assert headers["X-Standin-Source-BVH-SHA256"] == source
    assert headers["X-Standin-Output-Scope"] == scope
    assert headers["Cache-Control"] == "private, no-store"
    assert metadata.read_bytes() == original and stem.with_suffix(".jpg").read_bytes() == jpeg


@pytest.mark.parametrize("key", ["schema_version", "render_method", "source_bvh_sha256",
    "character_id", "character_sha256", "renderer_revision", "scope", "view", "azimuth_degrees",
    "native_camera_yaw", "pitch_degrees", "projection", "projection_matches_search", "selected_body",
    "hands_applied", "source_frame", "preview_only", "anatomy_approval", "width", "height",
    "format", "quality", "style", "solver_manifest_sha256", "image_sha256", "bytes", "border_pixels"])
def test_changed_identity_fails_closed_even_with_cached_png(tmp_path, key):
    source, _, meta, stem = asset(tmp_path)
    native.read_native_png(tmp_path, source)
    stem.with_suffix(".json").write_text(json.dumps({**meta, key: "changed"}))
    with pytest.raises((ValueError, TypeError)):
        native.read_native_png(tmp_path, source)


@pytest.mark.parametrize("key,value", [("source_frame", False), ("hands_applied", 0),
    ("projection_matches_search", 0), ("width", 512.0), ("schema_version", 2.0)])
def test_strict_metadata_types(tmp_path, key, value):
    source, _, meta, stem = asset(tmp_path)
    stem.with_suffix(".json").write_text(json.dumps({**meta, key: value}))
    with pytest.raises(ValueError):
        native.read_native_png(tmp_path, source)


def test_incomplete_missing_corrupt_and_symlink_assets_do_not_fall_back(tmp_path):
    source, _, meta, stem = asset(tmp_path)
    image = stem.with_suffix(".jpg")
    image.write_bytes(b"corrupt")
    with pytest.raises(ValueError): native.read_native_png(tmp_path, source)
    image.unlink()
    with pytest.raises(OSError): native.read_native_png(tmp_path, source)
    image.symlink_to(tmp_path / "private.jpg")
    with pytest.raises(ValueError): native.read_native_png(tmp_path, source)
    with pytest.raises(ValueError): native.read_native_png(tmp_path, "../" + source)
    (tmp_path / "library_manifest.json").write_text(json.dumps({"directional_thumbnails": {"status": "pending"}}))
    with pytest.raises(ValueError): native.read_native_png(tmp_path, source)


def test_head_is_excluded_and_old_bundle_keeps_existing_reader(tmp_path):
    assert native.read_native_png(tmp_path, "a" * 64) is None
    (tmp_path / "library_manifest.json").write_text("{}")
    assert native.read_native_png(tmp_path, "a" * 64) is None
    asset(tmp_path)
    assert native.read_native_png(tmp_path, "a" * 64, "head") is None


def test_original_api_url_png_guards_and_fail_closed(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from fastapi.responses import FileResponse
    import api.app as api
    source_bytes = b"public source BVH"
    source, _, meta, stem = asset(tmp_path, source=source_bytes)
    bvh = tmp_path / "source.bvh"
    bvh.write_bytes(source_bytes)
    monkeypatch.setattr(api, "get_pose_bvh", lambda _: FileResponse(bvh))
    monkeypatch.setattr(api.CFG, "data_dir", str(tmp_path))
    client = TestClient(api.app)
    url = "/pose/pose/poster?source_sha256=" + source
    response = client.get(url)
    assert response.status_code == 200 and response.headers["content-type"] == "image/png"
    assert response.headers["x-standin-native-render-method"] == native.METHOD
    assert response.content == native.read_native_png(tmp_path, source)[0]
    assert client.get(url.replace(source, "a" * 64)).status_code == 409
    stem.with_suffix(".json").write_text(json.dumps({**meta, "renderer_revision": "a" * 64}))
    assert client.get(url).status_code == 404
    assert bvh.read_bytes() == source_bytes


def test_api_native_assets_never_bypass_quarantine(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import api.app as api
    source, *_ = asset(tmp_path)
    monkeypatch.setattr(api.CFG, "data_dir", str(tmp_path))
    monkeypatch.setattr(api, "quarantine_record", lambda *a: {"reason": "known_bad"})
    assert TestClient(api.app).get("/pose/bad/poster?source_sha256=" + source).status_code == 409
