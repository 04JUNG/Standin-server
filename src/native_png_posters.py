"""Adapt verified frozen-native front references to the existing PNG protocol.

No 3D rendering, source mutation, template relabelling or new client contract.
The HTTP method is the existing frozen-retarget family; native renderer identity
and the original JPEG digest remain explicit in additional provenance headers.
"""
from functools import lru_cache
import io
import json
from pathlib import Path
import re

from PIL import Image

METHOD = "frozen-native-scope8-v1"
SCOPES = ("full", "half", "bust", "shoulder_head")
CHARACTER_SHA = "7c648b97a24a3bb4914b6e5d515708c33727979881d92ef916d5726e22301f3d"
SOLVER_SHA = "52eae0b0eb6a7023540fdf023ad4321f24c11d439f3ecf9f390b5b72358b24e3"
REVISIONS = frozenset((
    "14d60ac3973bb86a9cb9b01c3bdfc4e89f233013373b5e6715ef9db40582518f",
    "a7ebf9dafc931a65ad171571c9033d45890693a5b24d976edd5ae3867d2a0f53",
))
MAX_BYTES = 2 * 1024 * 1024


def _json(path, maximum):
    if path.is_symlink() or not 0 < path.stat().st_size <= maximum:
        raise ValueError("invalid native reference metadata")
    result = json.loads(path.read_bytes())
    if not isinstance(result, dict):
        raise ValueError("invalid native reference metadata")
    return result


@lru_cache(maxsize=32)
def _png(jpeg: bytes) -> bytes:
    with Image.open(io.BytesIO(jpeg)) as image:
        if image.format != "JPEG" or image.size != (512, 512):
            raise ValueError("invalid native reference raster")
        image.load()
        output = io.BytesIO()
        image.convert("RGB").save(output, format="PNG")
    data = output.getvalue()
    if len(data) > MAX_BYTES:
        raise ValueError("native PNG is too large")
    return data


def read_native_png(data_dir: Path, source: str, scope="full"):
    """Return PNG/headers, or None only for old bundles or excluded head scope.

    An advertised native bundle fails closed on missing/corrupted assets. The
    source-registration and quarantine checks still belong to the caller.
    """
    if not re.fullmatch(r"[a-f0-9]{64}", source):
        raise ValueError("invalid native reference source")
    if scope not in SCOPES:
        return None
    base = data_dir.resolve()
    root = base / "thumbs" / "scope8"
    manifest_path = base / "library_manifest.json"
    if not manifest_path.exists() and not root.exists() and not root.is_symlink():
        return None
    manifest = _json(manifest_path, 65536)
    declared = manifest.get("directional_thumbnails")
    if declared is None and not root.exists() and not root.is_symlink():
        return None
    if (not isinstance(declared, dict) or declared.get("status") != "complete"
            or declared.get("render_method") != METHOD
            or scope not in declared.get("output_scopes", [])
            or "yaw_000" not in declared.get("views", [])
            or not isinstance(manifest.get("library_version"), str)
            or not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", manifest["library_version"])):
        raise ValueError("unpublished native reference bundle")
    if ((base / "thumbs").is_symlink() or root.is_symlink()
            or not root.resolve().is_relative_to(base)):
        raise ValueError("invalid native reference directory")
    stem = root / (source + "__" + scope + "__yaw_000")
    meta = _json(stem.with_suffix(".json"), 32768)
    expected = dict(schema_version=2, render_method=METHOD, source_bvh_sha256=source,
        character_id="standin-master-v2", character_sha256=CHARACTER_SHA,
        scope=scope, view="yaw_000", azimuth_degrees=0, native_camera_yaw=180,
        pitch_degrees=0, projection="orthographic", projection_matches_search=False,
        selected_body=False, hands_applied=False, source_frame=0, preview_only=True,
        anatomy_approval=False, width=512, height=512, format="jpeg", quality=90,
        style="approved-anatomy-detail-dark-5scope-8view-v1", solver_manifest_sha256=SOLVER_SHA)
    from src.pose_posters import character_sha256, digest, headers
    if (any(type(meta.get(k)) is not type(v) or meta.get(k) != v for k, v in expected.items())
            or character_sha256() != CHARACTER_SHA
            or meta.get("renderer_revision") not in REVISIONS
            or meta["renderer_revision"] not in declared.get("renderer_revisions", [])):
        raise ValueError("native reference lineage changed")
    crop = meta.get("framing")
    if (not isinstance(crop, dict) or crop.get("scope") != scope
            or crop.get("version") != "skin-regions-v2-shoulder-head"
            or type(crop.get("vertices_after")) is not int
            or type(crop.get("vertices_before")) is not int
            or not 0 < crop["vertices_after"] <= crop["vertices_before"]
            or crop.get("bones_preserved") != 52
            or type(meta.get("border_pixels")) is not int or meta["border_pixels"] < 12):
        raise ValueError("native reference crop changed")
    if ((scope == "full" and (crop["vertices_after"] != crop["vertices_before"] or crop.get("cap_faces") != 0))
            or (scope != "full" and crop["vertices_after"] >= crop["vertices_before"])):
        raise ValueError("native reference anatomical scope changed")
    path = stem.with_suffix(".jpg")
    if path.is_symlink() or not 3 <= path.stat().st_size <= MAX_BYTES:
        raise ValueError("invalid native reference image")
    jpeg = path.read_bytes()
    if (type(meta.get("bytes")) is not int or meta["bytes"] != len(jpeg)
            or digest(jpeg) != meta.get("image_sha256") or not jpeg.startswith(b"\xff\xd8\xff")):
        raise ValueError("native reference image changed")
    png = _png(jpeg)
    # This is an adapter protocol family, not a rewrite of the stored renderer.
    wire = {**meta, "kind": "default-body", "render_method": "frozen-retarget-v1",
            "image_sha256": digest(png), "bytes": len(png)}
    response_headers = headers(wire)
    response_headers.update({
        "X-Standin-Native-Render-Method": meta["render_method"],
        "X-Standin-Native-Artifact-SHA256": meta["image_sha256"],
        "X-Standin-Preview-View": "front",
        "X-Standin-Poster-Asset-Version": "native-front-png-v1",
        "X-Standin-Asset-Library-Version": manifest["library_version"],
    })
    return png, response_headers
