"""Versioned read-only preview contract; no Blender or asset download is started."""
import hashlib
from pathlib import Path

from converter.preview_model import MODEL_VERSION, model_revision
from converter.protocol import SOLVER_VERSION
from converter.framing import FRAMING_VERSION


def preview_contract() -> dict:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    # Include render/camera implementation as well as frozen solver + Blender lineage.
    digest.update(model_revision().encode())
    for name in ("converter/framing.py", "converter/camera.py", "converter/worker.py",
                 "converter/framed_model.py", "converter_api/body_preview.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes().replace(b"\r\n", b"\n"))
    return {
        "schema_version": "body-preview-runtime.v1",
        "preview_revision": digest.hexdigest(),
        "model_version": MODEL_VERSION,
        "model_revision": model_revision(),
        "solver_version": SOLVER_VERSION,
        "framing_version": FRAMING_VERSION,
    }
