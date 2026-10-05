"""Original ACCAD captures -> diverse, provenance-tracked single-frame BVHs."""

import argparse
from dataclasses import asdict
from pathlib import Path
import re

import numpy as np

from src.bvh import parse_bvh, fk, coco17_from_fk
from pose_curation.candidates import build_candidates
from pose_curation.hands.bvh import augment
from pose_curation.selection import select_frames
from pose_curation.storage import sha256, write_json, utc_now
from pose_curation.sources.rebase import export
from pose_curation.torso import reconstruct_accad

SOURCE_URL = "https://accad.osu.edu/research/motion-lab/mocap-system-and-data"
CLIPS = {
    "G1_SidekickLeadingLeft": "옆차기",
    "G4_SpinningBackKick": "회전 뒤차기",
    "G6_AxeKick": "내려차기",
    "G8_RoundhouseLeft": "돌려차기",
    "G16_DoubleKick": "연속 발차기",
    "E5_HookLeft": "왼손 훅",
    "E7_UppercutLeft": "어퍼컷",
    "E19_DodgeLeft": "옆으로 회피",
    "E21_DuckLeft": "숙여 피하기",
    "E17_BlockMiddleHigh": "상단 방어",
    "G7_Capoeira": "카포에라",
}


def run(source_dir, batch, count=2):
    if (batch / "manifest.json").exists():
        raise ValueError("choose a new batch")
    poses = []
    for clip, label in CLIPS.items():
        source = source_dir / f"Male2_{clip}.bvh"
        joints, frames = parse_bvh(str(source))
        match = re.search(r"Frame\s+Time:\s*([\d.eE+-]+)", source.read_text())
        dt = float(match.group(1))
        indices = list(range(0, len(frames), max(1, round(1 / 30 / dt))))
        points = np.stack(
            [coco17_from_fk(joints, fk(joints, frames[i]))[0] for i in indices]
        )
        across = points[0, 11] - points[0, 12]
        yaw = float(np.degrees(np.arctan2(across[2], across[0])))
        names = {j[0]: i for i, j in enumerate(joints)}
        floor = min(
            fk(joints, frames[i])[names[side + "Foot"]][1]
            for i in indices
            for side in ("Left", "Right")
        )
        picks = select_frames(
            points,
            np.array(indices) * dt,
            count=count,
            minimum_distance=0.28,
            minimum_seconds=0.1,
        )
        for pick in picks:
            frame = indices[pick.sample_index]
            identity = f"combat_accad_{clip}_f{frame:05d}"
            body = batch / "body" / (identity + ".bvh")
            rebased = batch / "rebased-body" / (identity + ".bvh")
            path = batch / "bvh" / (identity + ".bvh")
            checks = export(source, frame, rebased, yaw=yaw, floor=floor)
            checks["body_fk_check_scope"] = (
                "Original FK preservation is checked at rest-axis rebase, before the explicit torso reconstruction and optional arm clearance."
            )
            torso = reconstruct_accad(rebased, body)
            hands = augment(body, path, left="fist", right="fist")
            poses.append(
                {
                    "pose_id": identity,
                    "clip": clip,
                    "style": label,
                    "movement": "combat",
                    "source": "accad",
                    "author": "ACCAD / The Ohio State University",
                    "license": "CC-BY-3.0",
                    "license_url": "https://creativecommons.org/licenses/by/3.0/",
                    "source_url": SOURCE_URL,
                    "source_sha256": sha256(source),
                    "source_file": source.name,
                    "source_frame_0based": frame,
                    "source_time_seconds": frame * dt,
                    "selection": asdict(pick),
                    "bvh": path.relative_to(batch).as_posix(),
                    "bvh_sha256": sha256(path),
                    "thumbnails": {},
                    "preview_kind": "pending",
                    "rig_profile": "mixamo_noprefix",
                    "retarget_status": "not_validated",
                    "checks": checks,
                    "torso_correction": torso,
                    "hand_augmentation": {**hands, "captured_from_source": False},
                    "near_duplicate": False,
                    "transform": "Rest axes rebased; internal spine reconstructed from pelvis/chest landmarks; external body landmarks preserved; root travel removed and facing aligned; 30 procedural fist joints added.",
                }
            )
    projections = build_candidates(poses, batch, batch.name)
    write_json(
        batch / "manifest.json",
        {
            "schema_version": 1,
            "batch_id": batch.name,
            "created_at": utc_now(),
            "status": "complete",
            "poses": poses,
            "failures": [],
            "summary": {"poses": len(poses), "projections": projections},
        },
    )
    (batch / "ATTRIBUTION.txt").write_text(
        "Motion data: ACCAD / The Ohio State University, Open Motion Project. CC BY 3.0.\n"
        + SOURCE_URL
        + "\nhttps://creativecommons.org/licenses/by/3.0/\nChanges: selected static frames, rest-axis rebase, reconstruction of the internal torso with external landmarks preserved, facing/root translation, procedural finger joints and fist pose (not captured hand motion).\n",
        encoding="utf-8",
    )
    return len(poses)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--batch", type=Path, required=True)
    a = p.parse_args()
    print(run(a.source, a.batch))
