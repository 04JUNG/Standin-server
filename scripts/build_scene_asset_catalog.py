"""Verify reviewed scene artifacts and inventory missing physical objects.

Reads an immutable library and independent review receipts. No library edits,
uploads, guessed attachments, or automatic visual approvals.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from converter.scene_catalog import load_catalog, recipe_token
from converter.framed_model import metadata

ALIASES = {"greatsword": "sword", "goblet": "cup", "crate": "box",
           "chest": "box", "boulder": "stone", "bouquet": "flower"}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(base):
    tags = json.loads((base / "production/search-v2.json").read_text("utf-8"))
    if sha(base / "production/poses.db") != tags["source_db_sha256"]:
        raise ValueError("tags and source database do not match")
    decisions = json.loads((base / "final-scene-review-r7.json").read_text("utf-8"))
    reviewed = {p["pose_id"]: p for p in decisions["poses"]}
    rows = []
    for source, recipe in load_catalog().items():
        if recipe.review_status != "accepted":
            continue
        folder = base / "scene-recovery-r7/poses" / recipe.pose_id
        report = json.loads((folder / "report.json").read_text("utf-8"))
        decision = reviewed[recipe.pose_id]
        token = recipe_token(source)
        digest = sha(folder / "full.fbx")
        if (sha(base / "production/bvh" / (recipe.pose_id + ".bvh")) != source
                or not report["ok"] or not decision["approved"]
                or report["recipe_token"] != token
                or decision["visual_evidence"]["recipe_token"] != token
                or decision["visual_evidence"]["fbx_sha256"] != digest
                or decision["evidence"]["requires_contact_review"]):
            raise ValueError("stale or unreviewed scene: " + recipe.pose_id)
        if set(report["outputs"]) != {"full", "half", "bust", "shoulder_head"}:
            raise ValueError("scene is missing a reviewed output scope: " + recipe.pose_id)
        for scope, output in report["outputs"].items():
            # Resolve by known scope inside this folder, not paths in receipts.
            fbx, glb = folder / (scope + ".fbx"), folder / (scope + ".glb")
            if scope not in ("full", "half", "bust", "shoulder_head"):
                raise ValueError("unsupported scope")
            if (sha(fbx) != output["skin"]["artifact_sha256"]
                    or sha(glb) != output["model"]["sha256"]
                    or metadata(glb.read_bytes())["base_fbx_sha256"] != sha(fbx)):
                raise ValueError("FBX and preview diverged: " + recipe.pose_id)
        kinds = {s.split(":")[0] for s in recipe.prop_guides.split(";") if s}
        objects = {ALIASES.get(k, k) for k in kinds}
        if recipe.support == "seat":
            objects.add("chair")
        rows.append(dict(pose_id=recipe.pose_id, source_bvh_sha256=source,
                         recipe_token=token, reviewed_fbx_sha256=digest,
                         objects=sorted(objects), geometry_kinds=sorted(kinds),
                         support=recipe.support))
    physical = {r["pose_id"]: set(r["objects"]) for r in rows}
    missing, tagged, provided = [], Counter(), Counter()
    for pose in tags["poses"]:
        objects = {t["value"] for t in pose["groups"]["objects"]}
        tagged.update(objects)
        actual = physical.get(pose["pose_id"], set())
        provided.update(objects & actual)
        absent = objects - actual
        if absent:
            missing.append(dict(pose_id=pose["pose_id"],
                                source_bvh_sha256=pose["bvh_sha256"],
                                tagged_objects=sorted(objects),
                                missing_registered_objects=sorted(absent),
                                review_status="needs_attachment_review"))
    catalog = dict(schema_version=1, source_db_sha256=tags["source_db_sha256"],
                   library_version=tags["library_version"],
                   poses=sorted(rows, key=lambda r: r["pose_id"]))
    coverage = dict(library_poses=len(tags["poses"]), registered_scenes=len(rows),
                    handheld_scenes=sum(bool(r["geometry_kinds"]) for r in rows),
                    categories={k: dict(tagged=n, registered=provided[k],
                                        needs_review=n-provided[k])
                                for k, n in tagged.most_common()},
                    missing=missing,
                    policy="Tags identify intent; missing registration is not proof that every pose should receive an object.")
    return catalog, coverage


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--coverage", type=Path, required=True)
    args = p.parse_args()
    catalog, coverage = build(args.base.resolve(strict=True))
    for path, value in ((args.catalog, catalog), (args.coverage, coverage)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in coverage.items() if k != "missing"}, ensure_ascii=True))


if __name__ == "__main__":
    main()
