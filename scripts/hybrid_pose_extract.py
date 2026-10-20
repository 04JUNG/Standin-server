#!/usr/bin/env python3
"""Explicit single-image extraction; does not enable/replace a production route."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hybrid_pose import HybridPosePipeline
from src.hybrid_pose.gemini import GeminiReviewer
from src.hybrid_pose.frontend import DEFAULT_FRONTEND, build_frontend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory only; existing results are never overwritten",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("HYBRID_GEMINI_MODEL"),
        help="Explicit Gemini model ID; no automatic model selection",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180,
        help="Per-call network timeout, not the 20s performance target",
    )
    parser.add_argument(
        "--pose-model", choices=["humanart-m", "current-x"], default=DEFAULT_FRONTEND
    )
    parser.add_argument("--vlm-mode", choices=["single", "two-stage"], default="single")
    parser.add_argument(
        "--thinking-level",
        choices=["low", "default"],
        default="low",
        help="Low requires a supporting Gemini model; default omits thinkingConfig",
    )
    parser.add_argument(
        "--pose-manifest",
        type=Path,
        help="Verified Human-Art bundle; existing rollout guards remain enforced",
    )
    parser.add_argument(
        "--rtm-threshold",
        type=float,
        default=None,
        help="Default: selected model's threshold (Human-Art manifest; current-X 0.30)",
    )
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error("Input image does not exist")
    if args.output_dir.exists():
        parser.error("Output directory already exists; use a new directory")
    try:
        reviewer = GeminiReviewer(
            os.getenv("GEMINI_API_KEY"),
            args.model,
            args.timeout,
            thinking_level=args.thinking_level,
        )
        if args.rtm_threshold is not None and not 0 <= args.rtm_threshold <= 1:
            raise ValueError("RTM threshold must be within 0..1")
    except ValueError as error:
        parser.error(str(error))
    # Atomic exclusive directory creation prevents accidental result overwrite.
    args.output_dir.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        path = args.output_dir / name
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(
                json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )

    cold_started = time.perf_counter()
    try:
        model, default_threshold = build_frontend(args.pose_model, args.pose_manifest)
        threshold = (
            default_threshold if args.rtm_threshold is None else args.rtm_threshold
        )
        if not 0 <= threshold <= 1:
            raise ValueError("Invalid model-owned localization threshold")
    except Exception as error:
        save(
            "result.json",
            {
                "status": "error",
                "final": None,
                "pose_model": args.pose_model,
                "error": {"stage": "rtm_initialization", "type": type(error).__name__},
            },
        )
        print(
            "Pose initialization failed; check model bundle/rollout configuration and result.json",
            file=sys.stderr,
        )
        return 1
    cold_ms = (time.perf_counter() - cold_started) * 1000
    pipeline = HybridPosePipeline(model, reviewer, threshold, mode=args.vlm_mode)
    result = pipeline.extract(args.image, artifact=save)
    result["rtm_initialization_ms"] = cold_ms
    result["pose_model"] = args.pose_model
    if args.pose_model == "humanart-m":
        result["pose_runtime_identity"] = model.runtime_identity()
    result["timing_contract"] = (
        "Warm input loading through final JSON write, including image preparation, "
        "provider latency and local checks. Cold RTM initialization is separate. "
        "No 15s benchmark pacing or client upload/download latency is included."
    )
    save("result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "delivery_status": result["delivery_status"],
                "total_ms": round(result["timings_ms"]["total"], 1),
                "api_calls_started": result["api_calls_started"],
            }
        )
    )
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
