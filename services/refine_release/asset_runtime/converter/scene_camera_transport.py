"""Artifact-bound camera metadata; importing FBX does not apply a CSP camera."""
import base64
import json
import re

HEADER = "X-Standin-Scene-Camera"
VERSION = "scene-camera-artifact-v1"


def camera_binding(recipe, artifact_sha256):
    if not re.fullmatch(r"[a-f0-9]{64}", artifact_sha256):
        raise ValueError("invalid camera artifact hash")
    if not isinstance(recipe, dict) or recipe.get("version") != "scene-camera-v1":
        raise ValueError("invalid camera recipe")
    return {"version": VERSION, "artifact_sha256": artifact_sha256, "recipe": recipe}


def camera_header(recipe, artifact_sha256):
    value = camera_binding(recipe, artifact_sha256)
    encoded = base64.b64encode(json.dumps(value, allow_nan=False, separators=(",", ":")).encode()).decode()
    if len(encoded) > 8192:
        raise ValueError("camera metadata exceeds header limit")
    return encoded
