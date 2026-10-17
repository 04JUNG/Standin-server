#!/usr/bin/env python3
"""Reproducible opt-in rough body experiment. Never writes runtime selections.

observe: protocol crops -> cached Gemini visibility + pinned local SAM -> contours
prepare: body-independent safe refinement -> identical pose/view jobs for all bodies
compare: verify projection hashes -> same observed profiles -> diagnostics/abstention
"""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experimental.body_matching.rough_matching import sha256, compare_rough, VERSION


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    )


def observe(a):
    from PIL import Image
    from src.experimental.body_matching.rough_segmentation import Sam2PartSegmenter
    from src.experimental.body_matching.rough_pipeline import RoughBodyPipeline
    from src.experimental.body_matching.rough_evidence import GeminiPartVisibilityClient

    protocol = read(a.protocol)
    cached = read(a.visibility)["people"] if a.visibility else {}
    client = GeminiPartVisibilityClient() if a.allow_remote else None
    pipeline = RoughBodyPipeline(
        segmenter=Sam2PartSegmenter(a.checkpoint, a.checkpoint_sha256),
        visibility_client=client,
    )
    rows = []
    a.out.mkdir(parents=True, exist_ok=True)
    try:
        for case in protocol["cases"]:
            if sha256(case["crop"]) != case["crop_sha256"]:
                raise ValueError("crop_hash_mismatch")
            with Image.open(case["crop"]) as opened:
                im = opened.convert("RGB")
            row, masks = pipeline.observe(
                im, case, visibility=cached.get(case["case_id"])
            )
            # Hash-derived output names cannot escape the output directory.
            dest = a.out / hashlib.sha256(case["case_id"].encode()).hexdigest()[:16]
            dest.mkdir(exist_ok=True)
            for key in ("observed", "excluded_region"):
                Image.fromarray(masks[key]).save(dest / (key + ".png"))
            if masks["person"]:
                Image.fromarray((masks["person"]["mask"] * 255).astype("uint8")).save(
                    dest / "person.png"
                )
            row["mask_directory"] = str(dest.resolve())
            rows.append(row)
    finally:
        if client:
            client.close()
    write(
        a.out / "observations.json",
        dict(
            rows=rows,
            protocol_sha256=sha256(a.protocol),
            checkpoint_sha256=a.checkpoint_sha256,
            visibility_sha256=sha256(a.visibility) if a.visibility else None,
            implementation_sha256={
                name: sha256(
                    Path(__file__).resolve().parents[1]
                    / "src/experimental/body_matching"
                    / name
                )
                for name in (
                    "rough_evidence.py",
                    "rough_segmentation.py",
                    "rough_pipeline.py",
                )
            },
        ),
    )


def prepare(a):
    from src.refine_v2 import refine_bvh_v2

    catalog = read(a.catalog)
    a.out.mkdir(parents=True, exist_ok=True)
    assets = [
        dict(
            body_id=b["body_id"],
            asset_sha256=b["fbx_sha256"],
            fbx_path=str(
                (a.fbx_root / b["family"] / (b["preset"].lower() + ".fbx")).resolve()
            ),
        )
        for b in catalog["assets"]
    ]
    for asset in assets:
        if sha256(asset["fbx_path"]) != asset["asset_sha256"]:
            raise ValueError("body_asset_hash_mismatch")
    jobs = []
    for c in read(a.observations)["rows"]:
        if not c["ownership_verified"] or not c["segmentation"]["accepted_parts"]:
            continue
        for i, p in enumerate(c["poses"][:2]):
            if sha256(p["bvh_path"]) != p["pose_sha256"]:
                raise ValueError("pose_hash_mismatch")
            key = hashlib.sha256((c["case_id"] + str(i)).encode()).hexdigest()[:16]
            result = refine_bvh_v2(
                p["bvh_path"],
                np.asarray(c["points"]),
                np.asarray(c["mask"], float),
                p["view"],
                out_path=str(a.out / (key + ".bvh")),
                refine_mode="aggressive",
                lower_body_observed=all(c["mask"][i] for i in range(11, 17)),
                deadline=time.perf_counter() + 15,
            )
            jobs.append(
                dict(
                    key=key,
                    case_id=c["case_id"],
                    pose_id=p["pose_id"],
                    view=p["view"],
                    base_pose_sha256=p["pose_sha256"],
                    bvh_path=result.bvh_path,
                    pose_sha256=sha256(result.bvh_path),
                    refine=result.to_dict(),
                )
            )
    write(
        a.out / "jobs.json",
        dict(
            assets=assets,
            jobs=jobs,
            catalog_sha256=sha256(a.catalog),
            observations_sha256=sha256(a.observations),
        ),
    )


def compare(a):
    source = read(a.observations)
    data = read(a.jobs)
    index = read(a.projections)
    if index["jobs_sha256"] != sha256(a.jobs):
        raise ValueError("projection_jobs_mismatch")
    if data.get("observations_sha256") and data["observations_sha256"] != sha256(
        a.observations
    ):
        raise ValueError("observations_changed_after_pose_fit")
    rows = [
        compare_rough(
            c,
            index["records"],
            data["assets"],
            [j for j in data["jobs"] if j["case_id"] == c["case_id"]],
        )
        for c in source["rows"]
    ]
    count = len(rows)
    compared = sum(
        any(h.get("comparison", {}).get("ranking") for h in r["hypotheses"])
        for r in rows
    )
    metrics = dict(
        total=count,
        owned=sum(c["ownership_verified"] for c in source["rows"]),
        contour_supported=sum(
            bool(c["segmentation"]["accepted_parts"]) for c in source["rows"]
        ),
        numerically_compared=compared,
        review_candidates=sum(r["status"] == "review_candidate" for r in rows),
        auto_applied=sum(r["auto_apply"] for r in rows),
        held=sum(r["status"] == "held" for r in rows),
        body_top1_accuracy=None,
        body_top3_accuracy=None,
        segmentation_iou=None,
        ground_truth_status="artist labels and segmentation masks unavailable",
    )
    write(
        a.out,
        dict(
            version=VERSION,
            rows=rows,
            metrics=metrics,
            observations_sha256=sha256(a.observations),
            jobs_sha256=sha256(a.jobs),
            projection_index_sha256=sha256(a.projections),
            matching_code_sha256=sha256(
                Path(__file__).resolve().parents[1]
                / "src/experimental/body_matching/rough_matching.py"
            ),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    p = sub.add_parser("observe")
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--visibility", type=Path)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--allow-remote", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--fbx-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--jobs", type=Path, required=True)
    p.add_argument("--projections", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    globals()[a.phase](a)


if __name__ == "__main__":
    main()
