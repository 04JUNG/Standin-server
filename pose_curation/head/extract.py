"""Optional isolated MediaPipe worker. Reads allowlisted local images only.

Run with data/tools/face-runtime/Scripts/python -m pose_curation.head.extract.
The review server never imports MediaPipe or downloads a model on request.
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ..storage import contained_path, read_json, sha256, write_json

EXTRACT_VERSION = "face-observation-v1"
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


def run(curation, limit=None):
    os.environ.setdefault('MPLCONFIGDIR', str((curation / 'head-direction/matplotlib-cache').resolve()))
    import mediapipe as mp

    roots = sorted((curation / "coverage").glob("*/inputs.json"))
    if not roots:
        raise ValueError("No allowlisted local roughs")
    root = roots[-1].parent
    model = curation / "head-direction/models/face_landmarker.task"
    if sha256(model) != MODEL_SHA256:
        raise ValueError("Unexpected face model revision")
    if mp.__version__ != "0.10.21":
        raise ValueError("Use the pinned, isolated face runtime")
    config = {"version": EXTRACT_VERSION, "mediapipe": mp.__version__,
              "model_sha256": MODEL_SHA256, "detection_threshold": .7,
              "presence_threshold": .7, "max_faces": 8, "max_image_side": 1600,
              "flip_check": True}
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model.resolve())),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=config["max_faces"],
        min_face_detection_confidence=.7, min_face_presence_confidence=.7)
    files = read_json(root / "report.json").get("files", {})
    items = read_json(root / "inputs.json")[:limit]
    output = curation / "head-direction/observations"
    totals = {"images": 0, "detected_images": 0, "faces": 0, "errors": 0}
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        for item in items:
            identity = item["id"]
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", identity) or identity not in files:
                continue
            target = output / (identity + ".json")
            path = contained_path(root, files[identity])
            if sha256(path) != item["sha256"]:
                raise ValueError(f"Image changed: {identity}")
            old = read_json(target) if target.exists() else {}
            if old.get("config") == config and old.get("image_sha256") == item["sha256"] and "error" not in old:
                record = old
            else:
                record = {"image_id": identity, "image_sha256": item["sha256"], "config": config}
                try:
                    with Image.open(path) as source:
                        original = source.convert("RGB")
                    # Coverage sizes/coordinates refer to raw pixels, not EXIF-transposed ones.
                    if list(original.size) != item["size"]:
                        raise ValueError("Image dimensions changed")
                    reduced = original.copy()
                    reduced.thumbnail((1600, 1600))
                    record["size"] = list(original.size)
                    for name, frame in (("faces", reduced), ("flipped_faces", ImageOps.mirror(reduced))):
                        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(frame)))
                        record[name] = [[[p.x * original.width, p.y * original.height]
                                         for p in face] for face in result.face_landmarks]
                except (OSError, ValueError, RuntimeError) as exc:
                    record["error"] = str(exc)
                    totals["errors"] += 1
                if sha256(path) != item["sha256"]:
                    raise ValueError(f"Image changed while extracting: {identity}")
                write_json(target, record)
            totals["images"] += 1
            totals["faces"] += len(record.get("faces", []))
            totals["detected_images"] += bool(record.get("faces"))
            if totals["images"] % 25 == 0:
                print(totals, flush=True)
    write_json(curation / "head-direction/extraction-summary.json", {**totals, "config": config})
    print(totals, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curation-dir", type=Path, default=Path("data/curation"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.curation_dir, args.limit)
