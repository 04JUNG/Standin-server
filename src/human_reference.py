"""Pinned human skin reference assets; no request-time Converter/Blender."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from converter.framing import validate_scope

ROOT = Path(__file__).resolve().parents[1]
METHOD = "reference-skinning-v1"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def template_manifest():
    return json.loads((ROOT / "config/default_body_template.v1.json").read_text())


def reference_identity(source, scope):
    validate_scope(scope)
    manifest = template_manifest()
    asset = manifest["scopes"][scope]
    algorithm = json.loads(
        (ROOT / "config/human_reference_renderer.v1.json").read_text()
    )["algorithm_sha256"]
    revision = digest((METHOD + algorithm + asset["sha256"]).encode())
    return dict(
        version="default-body-poster-v1",
        source_bvh_sha256=source,
        character_id=manifest["character_id"],
        character_sha256=manifest["character_sha256"],
        renderer_revision=revision,
        view="front",
        scope=scope,
        projection="orthographic",
        projection_matches_search=False,
        selected_body=False,
        hands_applied=False,
        render_method=METHOD,
        pose_approximate=True,
        template_sha256=asset["sha256"],
    )


def read_template(directory: Path, scope: str):
    validate_scope(scope)
    manifest = template_manifest()
    asset = manifest["scopes"][scope]
    path = directory / (scope + ".json.gz")
    if (
        path.is_symlink()
        or path.stat().st_size != asset["bytes"]
        or asset["bytes"] > 2_000_000
    ):
        raise ValueError("invalid reference template")
    data = path.read_bytes()
    if digest(data) != asset["sha256"]:
        raise ValueError("reference template changed")
    return data, {
        "Content-Type": "application/gzip",
        "Cache-Control": "private, max-age=86400",
        "X-Standin-Template-Version": "default-body-template-v1",
        "X-Standin-Character-Id": manifest["character_id"],
        "X-Standin-Character-SHA256": manifest["character_sha256"],
        "X-Standin-Artifact-SHA256": asset["sha256"],
        "X-Standin-Template-JSON-SHA256": asset["json_sha256"],
        "X-Standin-Output-Scope": scope,
        "X-Standin-Preview-Method": METHOD,
        "X-Standin-Preview-Only": "true",
    }
