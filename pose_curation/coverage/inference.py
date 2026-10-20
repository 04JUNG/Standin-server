"""Resumable real RTMPose extraction; no mock or generated query joints."""
from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np
from PIL import Image

from src.pose import RTMPoseModel
from ..storage import read_json, sha256, write_json


def run(inputs: Path, output: Path) -> dict:
    model = RTMPoseModel()  # Fail loudly if weights/runtime are unavailable.
    identity = {"backend": "RTMPose Body performance/current-X", "rtmlib": version("rtmlib"),
                "onnxruntime": version("onnxruntime"),
                "detector_sha256": sha256(Path(model.model.det_model.onnx_model)),
                "pose_sha256": sha256(Path(model.model.pose_model.onnx_model)),
                "mode": "full_image_detection", "manual_query_coordinates": False}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "model.json", identity)
    rows = read_json(inputs)
    for index, row in enumerate(rows, 1):
        target = output / f"{row['id']}.json"
        if target.exists():
            cached = read_json(target)
            if cached.get("image_sha256") == row["sha256"] and cached.get("model") == identity:
                continue
        if sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError("input image changed")
        started = time.monotonic()
        with Image.open(row["path"]) as source:
            # Preserve transparent drawings on a white page, like the client.
            image = Image.new("RGBA", source.size, "white")
            image.alpha_composite(source.convert("RGBA"))
            skeletons = model.estimate(image.convert("RGB"), [], *source.size)
        people = []
        for person, skeleton in enumerate(skeletons):
            kp, scores = np.asarray(skeleton.keypoints), np.asarray(skeleton.scores)
            valid = scores >= .3
            torso = float(np.linalg.norm(kp[[5, 6]].mean(0) - kp[[11, 12]].mean(0)))
            people.append({"person": person, "keypoints": kp.tolist(), "scores": scores.tolist(),
                           "body_visible": int(valid[5:].sum()), "torso_visible": bool(valid[[5, 6, 11, 12]].all()),
                           "torso_pixels": torso, "mean_body_score": float(scores[5:].mean())})
        write_json(target, {"id": row["id"], "image_sha256": row["sha256"], "model": identity,
                            "people": people, "seconds": round(time.monotonic() - started, 3)})
        print(f"[{index}/{len(rows)}] {row['id']}: {len(people)} people", flush=True)
    return identity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs, args.output)
