"""Bounded, resumable independent inspection of published or candidate BVHs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess

from pose_curation.review.catalog import Catalog
from pose_curation.review.store import ReviewStore
from pose_curation.storage import read_json, sha256, write_json
from src.bvh import parse_bvh
from .batch import render_identity

PROJECT = Path(__file__).resolve().parents[2]


def finger_mode(pose):
    """Legacy partial/namespace rigs can still receive a body-only diagnostic.

    They must never be reported as having 30 transferred finger joints.
    Candidate rigs retain the strict finger adapter contract.
    """
    joints, _ = parse_bvh(str(pose.bvh))
    names = {j[0] for j in joints}
    required = {f"{side}Hand{finger}{n}" for side in ("Left", "Right")
                for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky")
                for n in (1, 2, 3)}
    return "legacy_body_only" if pose.group == "existing" and not required <= names else "strict"


def rig_profile(pose):
    if pose.metadata.get("rig_profile"):
        return pose.metadata["rig_profile"]
    joints, _ = parse_bvh(str(pose.bvh))
    names = {j[0] for j in joints}
    if "mixamorig:Hips" in names:
        return "mixamo"
    if "LowerBack" in names:
        return "cmu_bvh"
    if "Chest4" in names:
        return "100style"
    # G1 uses two spine bones. Spine2 is optional in the converter profile;
    # requiring it here incorrectly rejected otherwise supported existing BVHs.
    if {"Spine", "LeftArm", "RightArm"} <= names:
        return "mixamo_noprefix"
    raise ValueError(f"unsupported diagnostic rig: {pose.pose_id}")


def run(
    output: Path,
    *,
    ids=None,
    workers=4,
    render=False,
    triage_path: Path | None = None,
    data_dir=None,
    curation_dir=None,
    blender=None,
    character=None,
    render_flagged=True,
):
    if not 1 <= workers <= 4:
        raise ValueError("diagnostic workers must be between 1 and 4")
    output = output.resolve()
    data_dir = Path(data_dir or PROJECT / "data").resolve()
    curation_dir = Path(curation_dir or PROJECT / "data/curation").resolve()
    blender = Path(
        blender or PROJECT / "data/tools/blender-5.2.0-windows-x64/blender.exe"
    ).resolve()
    character = Path(
        character or curation_dir / "characters/standin-master-v2.fbx"
    ).resolve()
    poses = Catalog(data_dir, curation_dir).all()
    reviews = ReviewStore(curation_dir / "reviews.sqlite").all()
    existing = (
        {
            p["pose_id"]
            for p in read_json(triage_path)
            if p["group"] == "existing" and p["flags"]
        }
        if triage_path
        else set()
    )
    if ids is not None and set(ids) - {p.pose_id for p in poses}:
        raise ValueError("unknown diagnostic pose IDs")
    poses = [
        p
        for p in poses
        if (
            p.pose_id in ids
            if ids is not None
            else p.pose_id in existing
            or (
                p.group != "existing"
                and reviews.get((p.key, p.content_hash), {}).get("status") == "accepted"
            )
        )
    ]
    code = {
        **render_identity(character, blender),
        "runner": sha256(Path(__file__)),
        "diagnostic_worker": sha256(Path(__file__).with_name("diagnostic_worker.py")),
    }
    fingerprint = hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()
    prior = (
        read_json(output / "job-metadata.json")
        if (output / "job-metadata.json").exists()
        else {}
    )
    pending = []
    for p in poses:
        path = output / (p.pose_id + ".json")
        r = read_json(path) if path.exists() else {}
        if (
            prior.get("fingerprint") != fingerprint
            or not r.get("ok")
            or r.get("bvh_sha256") != p.content_hash
            or (render and not r.get("images"))
        ):
            pending.append(p)
    write_json(
        output / "job-metadata.json",
        {
            "fingerprint": fingerprint,
            "code": code,
            "pose_ids": [p.pose_id for p in poses],
        },
    )

    def worker(index):
        subset = pending[index::workers]
        if not subset:
            return
        job = output / f"job-{index}.json"
        write_json(
            job,
            {
                "character": str(character),
                "output": str(output),
                "render": render,
                "render_flagged": render_flagged,
                "poses": [
                    {
                        "pose_id": p.pose_id,
                        "bvh": str(p.bvh),
                        "rig_profile": rig_profile(p),
                        "finger_mode": finger_mode(p),
                    }
                    for p in subset
                ],
            },
        )
        with (output / f"worker-{index}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    str(blender),
                    "--background",
                    "--factory-startup",
                    "--threads",
                    "2",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(Path(__file__).with_name("diagnostic_worker.py")),
                    "--",
                    str(job),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                cwd=PROJECT,
                env=dict(
                    os.environ,
                    BLENDER_USER_RESOURCES=str(output / f"blender-user-{index}"),
                ),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, range(workers)))
    rows = [read_json(output / (p.pose_id + ".json")) for p in poses]
    for pose, row in zip(poses, rows):
        if (
            row.get("bvh_sha256") != pose.content_hash
            or sha256(pose.bvh) != pose.content_hash
        ):
            raise ValueError(f"BVH changed during diagnostics: {pose.pose_id}")
    summary = {
        "count": len(rows),
        "failed": sum(not r.get("ok") for r in rows),
        "flagged": sum(bool(r.get("anatomy", {}).get("flags")) for r in rows),
        "poses": rows,
        "fingerprint": fingerprint,
        "identity": code,
    }
    write_json(
        output / "summary.json",
        summary,
    )
    print(f"Inspected {len(rows)} converted poses", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose", action="append")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--triage",
        type=Path,
        help="Optional geometry triage JSON to include flagged existing poses.",
    )
    args = parser.parse_args()
    run(
        args.output,
        ids=args.pose,
        render=args.render,
        workers=args.workers,
        triage_path=args.triage,
    )
