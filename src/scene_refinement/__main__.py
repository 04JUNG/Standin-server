"""python -m src.scene_refinement: explicit opt-in, single private worker."""

import os
from .api import create_app
from .runtime import load_engine, required


def main():
    if os.environ.get("SCENE_REFINEMENT_ENABLED") != "1":
        raise ValueError("scene refinement is default-off")
    os.umask(0o077)
    app = create_app(token=required("SCENE_REFINEMENT_TOKEN"), factory=load_engine)
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("SCENE_REFINEMENT_HOST", "127.0.0.1"),
        port=int(os.environ.get("SCENE_REFINEMENT_PORT", "8002")),
        workers=1,
    )


if __name__ == "__main__":
    main()
