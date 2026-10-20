"""Dedicated CPU/Gemini staging runtime; never imports the /analyze app."""

import os
from pathlib import Path
import subprocess
import tempfile

from .assets import verify_release


class StartupError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def required(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(name + " is required")
    return value


def _load_engine(progress):
    progress[0] = "SCENE_ASSET_INVALID"
    paths, character_sha = verify_release(
        required("SCENE_ASSET_MANIFEST"), required("SCENE_ASSET_MANIFEST_SHA256")
    )
    progress[0] = "SCENE_CONFIG_INVALID"
    if required("SCENE_VLM_PROVIDER") != "gemini":
        raise ValueError("container runtime supports gemini only")
    required("GEMINI_API_KEY")
    model = required("SCENE_VLM_MODEL")
    # Separate opt-in worker uses manifest-owned calibration, independent of
    # /analyze routing. Set before importing its immutable CFG singleton.
    os.environ["POSE_CANARY_STAGE"] = "shadow"
    os.environ["REFINE_POSE_QUARANTINE_PATH"] = str(
        Path(__file__).resolve().parents[2] / "config/refine_pose_quarantine.v1.json"
    )
    progress[0] = "SCENE_POLICY_INVALID"
    from .engine import Engine, Settings
    from .evidence import Reader, VisualEvidence
    from . import plan  # Validate host-side planning dependencies too.
    from src.semantic_adapter_285 import excluded_pose_ids

    excluded_pose_ids()
    settings = Settings(
        root=Path(required("SCENE_REFINEMENT_ROOT")),
        blender=Path(required("SCENE_BLENDER_BINARY")),
        catalog=paths["catalog"],
        library=paths["library"],
        character=paths["character"],
        head_axes=paths["head_axes"],
        character_sha256=character_sha,
    )
    # Capture locally; do not publish Blender's filesystem/asset output in probes.
    progress[0] = "SCENE_BLENDER_PREFLIGHT_FAILED"
    probe = subprocess.run(
        [
            str(settings.blender),
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(Path(__file__).with_name("preflight_blender.py")),
            "--",
            str(settings.character),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=True,
    )
    if b"SCENE_PREFLIGHT_OK " not in probe.stdout:
        raise ValueError("Blender preflight did not complete")
    progress[0] = "SCENE_MODELS_NOT_READY"
    visual = VisualEvidence(str(paths["pose_manifest"]), str(paths["clip"]))
    # Exercise real offline CLIP inference; HumanArt constructor runs an ONNX probe.
    from PIL import Image

    for features in (
        visual.features(text=["a person standing"]),
        visual.features(images=[Image.new("RGB", (224, 224), "white")]),
    ):
        if not bool(visual.torch.isfinite(features).all()):
            raise ValueError("CLIP warmup returned nonfinite features")
    progress[0] = "SCENE_ENGINE_NOT_READY"
    engine = Engine(settings, Reader("gemini", model), visual)
    with tempfile.TemporaryDirectory(
        prefix="preflight-", dir=settings.root
    ) as directory:
        (Path(directory) / "write-probe").write_bytes(b"")
    return engine


def load_engine():
    progress = ["SCENE_STARTUP_FAILED"]
    try:
        return _load_engine(progress)
    except Exception as exc:
        raise StartupError(progress[0]) from exc
