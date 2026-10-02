"""Local recovery on undetected roughs using pixel transforms, never invented points."""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .extract import MODEL_SHA256
from .queries import HeadQueries
from ..storage import read_json, sha256, write_json

VERSION = "crop-recovery-mediapipe-v1"


def original_points(points, size, turns, mirrored=False):
    """Undo a quarter-turn and optional flip; output stays in original crop pixels."""
    points = np.asarray(points, float).copy()
    w, h = size
    rw, rh = (h, w) if turns % 2 else (w, h)
    points *= [rw, rh]
    if mirrored:
        points[:, 0] = rw - 1 - points[:, 0]
    x, y = points.T
    if turns == 1:
        points = np.stack((w - 1 - y, x), axis=1)
    elif turns == 2:
        points = np.stack((w - 1 - x, h - 1 - y), axis=1)
    elif turns == 3:
        points = np.stack((y, h - 1 - x), axis=1)
    if mirrored:
        points[:, 0] = w - 1 - points[:, 0]
    return points.tolist()


def run():
    root = Path("data/curation/head-direction")
    os.environ.setdefault("MPLCONFIGDIR", str((root / "matplotlib-cache").resolve()))
    import mediapipe as mp

    model = root / "models/face_landmarker.task"
    if sha256(model) != MODEL_SHA256 or mp.__version__ != "0.10.21":
        raise ValueError("Use the pinned face runtime and model")
    config = {
        "version": VERSION,
        "model_sha256": MODEL_SHA256,
        "threshold": 0.7,
        "crop_size": 512,
        "rotations": [0, 1, 2, 3],
        "code_sha256": sha256(Path(__file__)),
    }
    baseline = read_json(root / "candidates.json")
    queries = HeadQueries(root.parent)
    totals = {"images": 0, "detected_images": 0, "faces": 0}
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model.resolve())),
        num_faces=4,
        min_face_detection_confidence=0.7,
        min_face_presence_confidence=0.7,
    )
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        for item in queries.list():
            if item["excluded"] or any(
                c["status"] == "suggested"
                for c in baseline["images"].get(item["key"], {}).get("candidates", [])
            ):
                continue
            row = queries.get(item["key"])
            regions, inputs = [], {}
            for kind in ("anime-observations", "crop-observations"):
                path = root / kind / f"{row['key']}.json"
                if not path.exists():
                    continue
                raw = read_json(path)
                if raw["image_sha256"] != row["content_hash"]:
                    raise ValueError("Input changed")
                inputs[kind] = sha256(path)
                proposals = (
                    [{"bbox": b["bbox"]} for b in raw["boxes"]]
                    if kind == "anime-observations"
                    else raw["regions"]
                )
                for region in proposals:
                    if region["bbox"] not in [r["bbox"] for r in regions]:
                        regions.append(region)
            if not regions:
                continue
            output = root / "recovery-observations" / f"{row['key']}.json"
            cached = read_json(output) if output.exists() else {}
            if (
                cached.get("config") == config
                and cached.get("inputs") == inputs
                and cached.get("image_sha256") == row["content_hash"]
            ):
                record = cached
            else:
                record = {
                    "config": config,
                    "inputs": inputs,
                    "image_sha256": row["content_hash"],
                    "faces": [],
                }
                with Image.open(row["path"]) as source:
                    source = source.convert("RGB")
                    for region in regions[:8]:
                        crop = source.crop(region["bbox"])
                        frame = ImageOps.autocontrast(crop, cutoff=1)
                        scale = 512 / max(crop.size)
                        frame = frame.resize(
                            tuple(max(1, round(v * scale)) for v in crop.size),
                            Image.Resampling.LANCZOS,
                        )
                        for turns in (0, 1, 2, 3):
                            rotated = (
                                frame
                                if turns == 0
                                else frame.transpose(
                                    {
                                        1: Image.Transpose.ROTATE_90,
                                        2: Image.Transpose.ROTATE_180,
                                        3: Image.Transpose.ROTATE_270,
                                    }[turns]
                                )
                            )
                            pair = []
                            for mirrored, image in (
                                (False, rotated),
                                (True, ImageOps.mirror(rotated)),
                            ):
                                result = detector.detect(
                                    mp.Image(
                                        image_format=mp.ImageFormat.SRGB,
                                        data=np.asarray(image),
                                    )
                                )
                                pair.append(
                                    [
                                        original_points(
                                            [[p.x, p.y] for p in face],
                                            crop.size,
                                            turns,
                                            mirrored,
                                        )
                                        for face in result.face_landmarks
                                    ]
                                )
                            for face in pair[0]:
                                record["faces"].append(
                                    {
                                        "region": region,
                                        "crop_size": list(crop.size),
                                        "points": face,
                                        "flipped_faces": pair[1],
                                        "quarter_turns": turns,
                                    }
                                )
                queries.get(row["key"], row["content_hash"])
                write_json(output, record)
            totals["images"] += 1
            totals["detected_images"] += bool(record["faces"])
            totals["faces"] += len(record["faces"])
            if totals["images"] % 20 == 0:
                print(totals, flush=True)
    write_json(root / "recovery-summary.json", {**totals, "config": config})
    print(totals, flush=True)


if __name__ == "__main__":
    run()
