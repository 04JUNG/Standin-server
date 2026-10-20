"""python -m src.scene_refinement (explicit opt-in, separate worker)."""

import os
from pathlib import Path
from .engine import Engine, Settings
from .evidence import Reader, VisualEvidence
from .api import create_app


def required(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(name + " is required")
    return value


def main():
    if os.environ.get("SCENE_REFINEMENT_ENABLED") != "1":
        raise ValueError("scene refinement is default-off")
    os.environ.setdefault(
        "REFINE_POSE_QUARANTINE_PATH",
        str(
            Path(__file__).resolve().parents[2]
            / "config/refine_pose_quarantine.v1.json"
        ),
    )
    settings = Settings(
        root=Path(required("SCENE_REFINEMENT_ROOT")),
        blender=Path(required("SCENE_BLENDER_BINARY")),
        catalog=Path(required("SCENE_POSE_CATALOG")),
        library=Path(required("SCENE_POSE_LIBRARY")),
        character=Path(required("SCENE_CHARACTER_FBX")),
        head_axes=Path(required("SCENE_HEAD_AXES")),
        character_sha256=required("SCENE_CHARACTER_SHA256"),
    )
    reader = Reader(required("SCENE_VLM_PROVIDER"), required("SCENE_VLM_MODEL"))
    visual = VisualEvidence(
        required("SCENE_POSE_MODEL_MANIFEST"),
        os.environ.get("SCENE_CLIP_MODEL", "openai/clip-vit-base-patch32"),
    )
    app = create_app(
        Engine(settings, reader, visual), required("SCENE_REFINEMENT_TOKEN")
    )
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("SCENE_REFINEMENT_HOST", "127.0.0.1"),
        port=int(os.environ.get("SCENE_REFINEMENT_PORT", "8002")),
    )


if __name__ == "__main__":
    main()
