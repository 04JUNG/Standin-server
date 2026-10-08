"""Select diverse frames from explicitly listed, locally cached CMU captures.

Acquisition is separate. This pool never approves or changes published poses.
Each clip is resumable and bound to its source/config hashes.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import numpy as np
from ..sources.acclaim import Acclaim
from ..sources.cmu_static import export, landmarks
from ..selection import select_frames
from ..storage import read_json, write_json, sha256, utc_now


BODY = [
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
    "LeftUpLeg",
    "RightUpLeg",
    "LeftLeg",
    "RightLeg",
    "LeftFoot",
    "RightFoot",
]


def _sampling_identity(count):
    """A cache hit must reproduce both the requested budget and conversion code."""
    from ..sources import acclaim, cmu_static

    return {
        "count": count,
        "implementation_sha256": {
            Path(path).name: sha256(Path(path))
            for path in (__file__, acclaim.__file__, cmu_static.__file__)
        },
    }


def _validate_cached(cached, config_hash, provenance, sampling):
    if (
        cached.get("config_sha256") != config_hash
        or cached.get("source_provenance") != provenance
        or cached.get("sampling") != sampling
        or any(sha256(Path(p["bvh"])) != p["bvh_sha256"] for p in cached["poses"])
    ):
        raise ValueError("pool changed; choose fresh output")


def _clip(job):
    source, output, detail, count, config_hash = job
    identity = detail["subject"] + "_" + detail["clip"]
    folder = output / "clips" / identity
    try:
        amc = source / (identity + ".amc")
        asf = source / (detail["subject"] + ".asf")
        provenance = {
            p.name: json.loads(
                p.with_suffix(p.suffix + ".source.json").read_text(encoding="utf-8-sig")
            )
            for p in (amc, asf)
        }
        for p in (amc, asf):
            if sha256(p) != provenance[p.name]["sha256"]:
                raise ValueError("cached capture hash changed")
        sampling = _sampling_identity(count)
        if (folder / "manifest.json").exists():
            cached = read_json(folder / "manifest.json")
            _validate_cached(cached, config_hash, provenance, sampling)
            return {
                "clip": identity,
                "manifest": str(folder / "manifest.json"),
                "reused": True,
            }
        motion = Acclaim.load(asf, amc)
        fps = detail["fps"]
        step = max(1, round(fps / 12))
        indices = list(
            range(
                max(1, round(fps * 0.12)), len(motion.frames) - round(fps * 0.12), step
            )
        )
        points = []
        floor = 1e9
        for i in indices:
            p, _ = landmarks(motion, i)
            floor = min(
                floor,
                p["LeftToeBase"][1],
                p["RightToeBase"][1],
                p["LeftFoot"][1],
                p["RightFoot"][1],
            )
            kp = np.zeros((17, 3))
            kp[:5] = p["Head"]
            kp[5:] = [p[name] for name in BODY]
            points.append(kp)
        picks = select_frames(
            np.stack(points),
            np.array(indices) / fps,
            count=count,
            minimum_distance=0.18,
            minimum_seconds=0.12,
        )
        rows = []
        for pick in picks:
            frame = indices[pick.sample_index]
            p, _ = landmarks(motion, frame)
            across = p["LeftUpLeg"] - p["RightUpLeg"]
            yaw = float(np.degrees(np.arctan2(across[2], across[0])))
            pid = f"sd_cmu_{identity}_f{frame:05d}"
            path = folder / "bvh" / (pid + ".bvh")
            checks = export(
                motion,
                frame,
                path,
                yaw=yaw,
                floor=floor,
                hands=detail.get("hands", ("relaxed", "relaxed")),
            )
            hand = checks.pop("hand_augmentation")
            rows.append(
                dict(
                    pose_id=pid,
                    clip=identity,
                    style=detail["label"],
                    movement=detail["category"],
                    category=detail["category"],
                    category_label=(
                        "스포츠" if detail["category"] == "sports" else "일상"
                    ),
                    family=detail["family"],
                    source="cmu",
                    author="CMU Graphics Lab",
                    license="CMU-use-in-products-no-data-resale",
                    license_url="https://mocap.cs.cmu.edu/",
                    source_url=provenance[amc.name]["url"],
                    source_sha256=sha256(amc),
                    source_skeleton_sha256=sha256(asf),
                    source_frame_0based=frame,
                    source_time_seconds=frame / fps,
                    selection=asdict(pick),
                    bvh=str(path.resolve()),
                    bvh_sha256=sha256(path),
                    thumbnails={},
                    preview_kind="pending",
                    rig_profile="mixamo_noprefix",
                    retarget_status="not_validated",
                    checks=checks,
                    hand_augmentation=hand,
                    near_duplicate=False,
                    transform="Captured CMU external body landmarks preserved; distributed spine and anatomical elbow axes; synthesized neutral wrist orientation and 30 fingers; heading/root travel normalized. Original finger channels are not captured and are omitted.",
                )
            )
        write_json(
            folder / "manifest.json",
            {
                "poses": rows,
                "config_sha256": config_hash,
                "source_provenance": provenance,
                "sampling": sampling,
            },
        )
        return {"clip": identity, "manifest": str(folder / "manifest.json")}
    except (OSError, ValueError, KeyError, IndexError) as error:
        return {"clip": identity, "error": str(error)}


def build(source, output, config_path, *, workers=4, count=12):
    if count < 1 or not 1 <= workers <= 4:
        raise ValueError("positive frame count and 1..4 workers are required")
    config = read_json(config_path)
    config_hash = sha256(config_path)
    jobs = []
    for group in config["groups"]:
        for clip, label in group["clips"].items():
            details = {k: v for k, v in group.items() if k != "clips"}
            details.update(clip=clip, label=label)
            jobs.append((source, output, details, count, config_hash))
    rows = []
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_clip, jobs):
            if "error" in result:
                failures.append(result)
                print("failed", result, flush=True)
            else:
                poses = read_json(Path(result["manifest"]))["poses"]
                rows.extend(poses)
                print("sampled", result["clip"], len(poses), flush=True)
    write_json(
        output / "manifest.json",
        {
            "poses": rows,
            "failures": failures,
            "created_at": utc_now(),
            "config_sha256": config_hash,
            "automatic_approval": False,
        },
    )
    return {"poses": len(rows), "failed_clips": len(failures)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    p.add_argument("--count", type=int, default=12)
    a = p.parse_args()
    result = build(a.source, a.output, a.config, workers=a.workers, count=a.count)
    print(result)
    raise SystemExit(1 if result["failed_clips"] else 0)
