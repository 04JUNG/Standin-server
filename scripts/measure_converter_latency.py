#!/usr/bin/env python3
"""Measure converter readiness, first-request, and warm-request latency."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import uuid


SERVER_TIMING = re.compile(r"([A-Za-z][A-Za-z0-9_-]*);dur=([0-9]+(?:\.[0-9]+)?)")


def _percentile_nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("at least one latency sample is required")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _latency_summary(samples: list[dict], key: str) -> dict[str, float]:
    values = [float(sample[key]) for sample in samples if sample[key] is not None]
    return {
        "p50_ms": round(_percentile_nearest_rank(values, 0.50), 3),
        "p95_ms": round(_percentile_nearest_rank(values, 0.95), 3),
        "max_ms": round(max(values), 3),
    }


def _multipart_body(
    *,
    bvh: bytes,
    filename: str,
    character_id: str,
) -> tuple[bytes, str]:
    boundary = f"standin-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    fields = {
        "character_id": character_id,
        "frame": "0",
        "mirror": "false",
        "output_mode": "rigged_rest",
        "apply_root_translation": "false",
    }
    for name, value in fields.items():
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )
    chunks.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="bvh"; filename="{filename}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8")
    )
    chunks.extend((bvh, b"\r\n", f"--{boundary}--\r\n".encode("ascii")))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _wait_for_health(base_url: str, timeout_seconds: float) -> tuple[float, dict]:
    started = time.monotonic()
    deadline = started + timeout_seconds
    last_error = "not ready"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"{base_url.rstrip('/')}/healthz", timeout=10
            ) as response:
                payload = json.load(response)
                if response.status == 200 and payload.get("ok") is True:
                    return (time.monotonic() - started) * 1000.0, payload
                last_error = f"status={response.status} ok={payload.get('ok')}"
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = type(exc).__name__
        time.sleep(0.25)
    raise TimeoutError(f"converter did not become healthy: {last_error}")


def _convert_once(
    *,
    base_url: str,
    body: bytes,
    content_type: str,
    source_sha256: str,
) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/convert",
        data=body,
        method="POST",
        headers={"Content-Type": content_type},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=120) as response:
        artifact = response.read()
        wall_ms = (time.monotonic() - started) * 1000.0
        headers = {key.lower(): value for key, value in response.headers.items()}
    artifact_sha256 = hashlib.sha256(artifact).hexdigest()
    if response.status != 200 or not artifact.startswith(b"Kaydara FBX Binary"):
        raise RuntimeError("converter response is not a binary FBX")
    if headers.get("x-standin-source-bvh-sha256") != source_sha256:
        raise RuntimeError("converter source lineage mismatch")
    if headers.get("x-standin-artifact-sha256") != artifact_sha256:
        raise RuntimeError("converter artifact lineage mismatch")
    timings = {
        name: float(value)
        for name, value in SERVER_TIMING.findall(headers.get("server-timing", ""))
    }
    return {
        "conversion_id": headers.get("x-standin-conversion-id"),
        "task_cold_start": headers.get("x-standin-task-cold-start") == "true",
        "wall_ms": round(wall_ms, 3),
        "queue_ms": timings.get("queue"),
        "execution_ms": timings.get("blender"),
        "server_total_ms": timings.get("total"),
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": len(artifact),
    }


def measure(args: argparse.Namespace) -> dict:
    bvh = args.bvh.read_bytes()
    source_sha256 = hashlib.sha256(bvh).hexdigest()
    body, content_type = _multipart_body(
        bvh=bvh,
        filename=args.bvh.name,
        character_id=args.character_id,
    )
    readiness_ms, health = _wait_for_health(args.base_url, args.readiness_timeout_seconds)
    first = _convert_once(
        base_url=args.base_url,
        body=body,
        content_type=content_type,
        source_sha256=source_sha256,
    )
    warm = [
        _convert_once(
            base_url=args.base_url,
            body=body,
            content_type=content_type,
            source_sha256=source_sha256,
        )
        for _ in range(args.warm_iterations)
    ]
    warm_wall = [sample["wall_ms"] for sample in warm]
    warm_p95_ms = round(_percentile_nearest_rank(warm_wall, 0.95), 3)
    return {
        "schema_version": 1,
        "base_url": args.base_url,
        "character_id": args.character_id,
        "source_bvh_sha256": source_sha256,
        "health": health,
        "readiness_ms_from_probe_start": round(readiness_ms, 3),
        "first_request": first,
        "warm": {
            "iterations": len(warm),
            "wall": _latency_summary(warm, "wall_ms"),
            "queue": _latency_summary(warm, "queue_ms"),
            "execution": _latency_summary(warm, "execution_ms"),
            "server_total": _latency_summary(warm, "server_total_ms"),
            "target_p95_ms": args.warm_p95_target_ms,
            "meets_target": warm_p95_ms <= args.warm_p95_target_ms,
            "samples": warm,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--bvh", required=True, type=Path)
    parser.add_argument("--character-id", default="standin-master-v2")
    parser.add_argument("--warm-iterations", type=int, default=20)
    parser.add_argument("--readiness-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--warm-p95-target-ms", type=float, default=3000.0)
    parser.add_argument("--enforce-warm-target", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.warm_iterations < 1:
        parser.error("--warm-iterations must be positive")
    return args


def main() -> int:
    args = parse_args()
    report = measure(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.enforce_warm_target and not report["warm"]["meets_target"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
