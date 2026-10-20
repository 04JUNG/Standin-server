"""python -m pose_curation {build,hands,render,publish,serve}."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standin pose acquisition and local review"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    scoped = commands.add_parser("scoped-build", help="Build revision-bound half/bust representative libraries")
    scoped.add_argument("--data-dir", type=Path, default=Path("data"))
    scoped.add_argument("--curation-dir", type=Path, default=Path("data/curation"))
    qa = commands.add_parser(
        "qa", help="Run revision-bound QA and build a visual review queue"
    )
    qa.add_argument("--data-dir", type=Path, default=Path("data"))
    qa.add_argument("--curation-dir", type=Path, default=Path("data/curation"))
    qa.add_argument("--output", type=Path, default=Path("data/curation/qa/latest"))
    qa.add_argument("--batch")
    qa.add_argument("--pose", action="append")
    qa.add_argument("--scope", choices=["accepted", "candidates"])
    qa.add_argument("--include-excluded", action="store_true")
    qa.add_argument(
        "--prepare",
        action="store_true",
        help="Generate missing previews and refresh mesh checks; never approve",
    )
    qa.add_argument("--blender", type=Path)
    qa.add_argument("--character", type=Path)
    qa.add_argument("--workers", type=int, default=2)
    build = commands.add_parser("build", help="Download and build review candidates")
    build.add_argument(
        "--config", type=Path, default=Path("config/pose_curation_pilot.json")
    )
    build.add_argument("--batch", required=True)
    build.add_argument("--data-dir", type=Path, default=Path("data"))
    build.add_argument("--curation-dir", type=Path, default=Path("data/curation"))
    build.add_argument("--offline", action="store_true")
    build.add_argument("--limit", type=int)
    hands = commands.add_parser(
        "hands", help="Add synthetic finger joints and a hand pose to candidate BVHs"
    )
    hands.add_argument("--batch-dir", type=Path, required=True)
    hands.add_argument("--left", choices=("open", "relaxed", "fist"), default="relaxed")
    hands.add_argument(
        "--right", choices=("open", "relaxed", "fist"), default="relaxed"
    )
    hands.add_argument(
        "--pose",
        action="append",
        help="Exact pose ID; repeat to select several. Default: all poses.",
    )
    render = commands.add_parser(
        "render", help="Render candidates with the service character"
    )
    render.add_argument("--batch-dir", type=Path, required=True)
    render.add_argument("--blender", type=Path, required=True)
    render.add_argument(
        "--character",
        type=Path,
        default=Path("data/curation/characters/standin-master-v2.fbx"),
    )
    render.add_argument("--workers", type=int, default=2)
    render.add_argument("--limit", type=int)
    render.add_argument(
        "--pose",
        action="append",
        help="Exact pose ID; repeat to render selected revisions only.",
    )
    publish = commands.add_parser(
        "publish",
        help="Build a local search DB with exclusions applied and accepted new poses",
    )
    publish.add_argument("--data-dir", type=Path, default=Path("data"))
    publish.add_argument("--curation-dir", type=Path, default=Path("data/curation"))
    audit = commands.add_parser(
        "audit", help="Triage BVH geometry and generate character contact sheets"
    )
    audit.add_argument("--data-dir", type=Path, default=Path("data"))
    audit.add_argument("--curation-dir", type=Path, default=Path("data/curation"))
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--group", choices=("all", "existing", "new"), default="all")
    audit.add_argument("--batch", default="")
    audit.add_argument("--flagged-only", action="store_true")
    serve = commands.add_parser("serve", help="Run the local review server")
    serve.add_argument("--data-dir", type=Path, default=Path("data"))
    serve.add_argument("--curation-dir", type=Path, default=Path("data/curation"))
    serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command == "scoped-build":
        from .scoped.index import build
        result = build(args.data_dir, args.curation_dir)
        print({"sources": result["source_count"], "scopes": {k: v["count"] for k, v in result["scopes"].items()},
               "omitted": result["omitted_counts"]})
        return 0
    if args.command == "qa":
        from .qa.workflow import run

        result = run(
            args.data_dir,
            args.curation_dir,
            args.output,
            batch=args.batch,
            pose_ids=args.pose,
            scope=args.scope,
            include_excluded=args.include_excluded,
            prepare_evidence=args.prepare,
            blender=args.blender,
            character=args.character,
            workers=args.workers,
        )
        return (
            2
            if result["errors"]
            or result["counts"].get("blocked")
            or result["counts"].get("visual_review")
            else 0
        )
    if args.command == "audit":
        from .audit import run

        run(
            args.data_dir,
            args.curation_dir,
            args.output,
            group=args.group,
            batch=args.batch,
            flagged_only=args.flagged_only,
        )
        return 0
    if args.command == "publish":
        from .publication import run

        run(args.data_dir, args.curation_dir)
        return 0
    if args.command == "build":
        from .pipeline import run

        result = run(
            args.config,
            args.data_dir,
            args.curation_dir,
            args.batch,
            offline=args.offline,
            limit=args.limit,
        )
        return 0 if result["status"] == "complete" else 1
    if args.command == "render":
        from .rendering.batch import run

        result = run(
            args.batch_dir,
            args.blender,
            args.character,
            workers=args.workers,
            limit=args.limit,
            pose_ids=args.pose,
        )
        return 0 if result["status"] == "complete" else 1
    if args.command == "hands":
        from .hands.batch import run

        run(args.batch_dir, left=args.left, right=args.right, pose_ids=args.pose)
        return 0
    import uvicorn
    from .review.app import create_app

    uvicorn.run(
        create_app(args.data_dir.resolve(), args.curation_dir.resolve()),
        host="127.0.0.1",
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
