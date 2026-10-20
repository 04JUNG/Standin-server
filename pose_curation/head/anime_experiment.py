"""Pinned local drawing detector, with independent MediaPipe landmark crop checks.

Anime detector boxes propose regions; its 28 landmarks are retained for audit,
not silently mapped onto the unrelated 478-point face template.
"""

import argparse
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .extract import MODEL_SHA256
from .queries import HeadQueries
from ..storage import read_json, sha256, write_json

WEIGHT_HASHES = {
    "yolov3": "23bbc708146bcbc1c910f00fe152adbc70d7658d875a0121eaf4ee61d978b2c4",
    "hrnetv2": "e71271376406a743c01528a0460637fcc06e72aeeea583f85007cc72dc8b7a4a",
}


def run(ids=None):
    root = Path("data/curation")
    models = root / "head-direction/models"
    os.environ.setdefault(
        "MPLCONFIGDIR", str((root / "head-direction/matplotlib-cache").resolve())
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import torch
    import cv2
    import mediapipe as mp
    from importlib.metadata import version
    from anime_face_detector.detector import LandmarkDetector

    torch.set_num_threads(2)
    for name, digest in WEIGHT_HASHES.items():
        if sha256(models / f"anime-{name}.safetensors") != digest:
            raise ValueError("Anime weights changed")
    if sha256(models / "face_landmarker.task") != MODEL_SHA256:
        raise ValueError("Face weights changed")
    config = {
        "version": "anime-box-mediapipe-v1",
        "anime_face_detector": version("anime-face-detector"),
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "mediapipe": mp.__version__,
        "weight_hashes": WEIGHT_HASHES,
        "face_model_sha256": MODEL_SHA256,
        "box_threshold": 0.8,
        "face_threshold": 0.7,
        "max_boxes": 12,
        "crop_margin": 1.6,
    }
    if config["anime_face_detector"] != "0.1.0":
        raise ValueError("Use pinned anime package")
    detector = LandmarkDetector(
        models / "anime-hrnetv2.safetensors",
        face_detector_name="yolov3",
        face_detector_checkpoint_path=models / "anime-yolov3.safetensors",
        device="cpu",
    )
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str((models / "face_landmarker.task").resolve())
        ),
        num_faces=8,
        min_face_detection_confidence=0.7,
        min_face_presence_confidence=0.7,
    )
    queries = HeadQueries(root)
    totals = {
        "images": 0,
        "box_images": 0,
        "boxes": 0,
        "landmark_images": 0,
        "faces": 0,
    }
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        for item in queries.list():
            if ids and item["key"] not in ids:
                continue
            if item["excluded"] and not ids:
                continue
            row = queries.get(item["key"], allow_excluded=bool(ids))
            target = root / "head-direction/anime-observations" / f"{row['key']}.json"
            old = read_json(target) if target.exists() else {}
            if (
                old.get("config") == config
                and old.get("image_sha256") == row["content_hash"]
            ):
                record = old
            else:
                started = time.perf_counter()
                with Image.open(row["path"]) as source:
                    original = source.convert("RGB")
                bgr = cv2.cvtColor(np.asarray(original), cv2.COLOR_RGB2BGR)
                # Detect first, cap boxes before landmark inference to bound memory.
                boxes = detector.face_detector.detect(bgr)
                boxes = sorted(
                    [b for b in boxes if b[4] >= 0.8], key=lambda b: -float(b[4])
                )[:12]
                predictions = (
                    detector(bgr, boxes=[b.copy() for b in boxes]) if boxes else []
                )
                record = {
                    "image_id": row["key"],
                    "image_sha256": row["content_hash"],
                    "size": row["size"],
                    "config": config,
                    "boxes": [],
                    "faces": [],
                }
                for index, pred in enumerate(predictions):
                    bbox = pred["bbox"]
                    low, high = bbox[:2], bbox[2:4]
                    center = (low + high) / 2
                    half = max(high - low) * 0.8
                    x0, y0 = np.maximum(np.floor(center - half), 0).astype(int)
                    x1, y1 = np.minimum(np.ceil(center + half), original.size).astype(
                        int
                    )
                    if min(x1 - x0, y1 - y0) < 48:
                        continue
                    region = {
                        "bbox": [int(x0), int(y0), int(x1), int(y1)],
                        "detector_box": bbox.tolist(),
                        "box_index": index,
                    }
                    record["boxes"].append(
                        {**region, "anime_keypoints": pred["keypoints"].tolist()}
                    )
                    crop = original.crop(region["bbox"])
                    pixels = np.asarray(crop)
                    pair = []
                    for frame in (pixels, np.ascontiguousarray(pixels[:, ::-1])):
                        result = landmarker.detect(
                            mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
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
                queries.get(row["key"], row["content_hash"], allow_excluded=bool(ids))
                record["seconds"] = time.perf_counter() - started
                write_json(target, record)
            totals["images"] += 1
            totals["box_images"] += bool(record["boxes"])
            totals["boxes"] += len(record["boxes"])
            totals["landmark_images"] += bool(record["faces"])
            totals["faces"] += len(record["faces"])
            if ids or totals["images"] % 25 == 0:
                print(
                    {
                        **totals,
                        "last": row["key"],
                        "seconds": round(record.get("seconds", 0), 2),
                    },
                    flush=True,
                )
    write_json(
        root
        / "head-direction"
        / ("anime-pilot-summary.json" if ids else "anime-summary.json"),
        {**totals, "config": config},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", action="append")
    run(parser.parse_args().id)
