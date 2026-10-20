"""Measure MediaPipe on body-proposed crops without changing automatic UI policy."""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .extract import MODEL_SHA256
from .queries import HeadQueries
from .regions import body_head_regions
from ..storage import read_json, sha256, write_json


def main():
    root = Path("data/curation")
    os.environ.setdefault(
        "MPLCONFIGDIR", str((root / "head-direction/matplotlib-cache").resolve())
    )
    import mediapipe as mp

    model = root / "head-direction/models/face_landmarker.task"
    if sha256(model) != MODEL_SHA256:
        raise ValueError("Model changed")
    queries = HeadQueries(root)
    coverage = sorted((root / "coverage").glob("*/inputs.json"))[-1].parent
    config = {
        "version": "body-crop-mediapipe-v1",
        "model_sha256": MODEL_SHA256,
        "regions_sha256": sha256(Path(__file__).with_name("regions.py")),
        "threshold": 0.7,
    }
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model.resolve())),
        num_faces=8,
        min_face_detection_confidence=0.7,
        min_face_presence_confidence=0.7,
    )
    summary = {
        "images": 0,
        "proposed_images": 0,
        "regions": 0,
        "detected_images": 0,
        "faces": 0,
    }
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        for row in queries.list():
            if row["excluded"]:
                continue
            row = queries.get(row["key"])
            extraction = coverage / "extraction" / f"{row['key']}.json"
            body = read_json(extraction) if extraction.exists() else {}
            if body.get("image_sha256") != row["content_hash"]:
                continue
            regions = body_head_regions(body["people"], row["size"])
            target = root / "head-direction/crop-observations" / f"{row['key']}.json"
            old = read_json(target) if target.exists() else {}
            if (
                old.get("config") == config
                and old.get("image_sha256") == row["content_hash"]
            ):
                record = old
            else:
                record = {
                    "image_id": row["key"],
                    "image_sha256": row["content_hash"],
                    "size": row["size"],
                    "config": config,
                    "regions": regions,
                    "faces": [],
                }
                with Image.open(row["path"]) as source:
                    original = source.convert("RGB")
                for region in regions:
                    x0, y0, x1, y1 = region["bbox"]
                    crop = original.crop(region["bbox"])
                    pair = []
                    for image in (crop, ImageOps.mirror(crop)):
                        result = detector.detect(
                            mp.Image(
                                image_format=mp.ImageFormat.SRGB, data=np.asarray(image)
                            )
                        )
                        pair.append(
                            [
                                [[p.x * crop.width, p.y * crop.height] for p in face]
                                for face in result.face_landmarks
                            ]
                        )
                    for face in pair[0]:
                        record["faces"].append(
                            {
                                "region": region,
                                "points": face,
                                "flipped_faces": pair[1],
                                "crop_size": list(crop.size),
                            }
                        )
                queries.get(row["key"], row["content_hash"])
                write_json(target, record)
            summary["images"] += 1
            summary["proposed_images"] += bool(regions)
            summary["regions"] += len(regions)
            summary["detected_images"] += bool(record["faces"])
            summary["faces"] += len(record["faces"])
            if summary["images"] % 50 == 0:
                print(summary, flush=True)
    write_json(root / "head-direction/crop-summary.json", {**summary, "config": config})
    print(summary, flush=True)


if __name__ == "__main__":
    main()
