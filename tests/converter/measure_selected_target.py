"""Local-only selected-target benchmark; no network, VLM, flags or deployment.

Run in a fresh dedicated Python process. Model/source paths come from a local
manifest, never committed. Cold/warm refer to runner metadata and OS cache;
every solve/inspection uses a fresh Blender process. No warm bpy mutation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from converter.selected_snapshot import encode, sha
from converter.shape_spec import resolve_shape
from converter_api.runner import RunnerSettings
from converter_api.selected_target import SelectedBlenderRunner, inspect_target, SelectedTargetProvider, TargetSelection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shape-registry", type=Path, required=True)
    parser.add_argument("--standing", type=Path, required=True)
    parser.add_argument("--sitting", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CONVERTER_SHAPE_REGISTRY"] = str(args.shape_registry)
    # Set once, before any worker starts; never change a process-global per request.
    os.environ["STANDIN_LAB_PREVIEW_TRACE"] = "1"
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_bytes())
    tasks = []
    for sex, pose, source_path, shaped in (
        ("male", "standing", args.standing, False),
        ("female", "standing", args.standing, False),
        ("male", "sitting-diagnostic-only", args.sitting, False),
        ("female", "standing-shaped", args.standing, True)):
        model = manifest["models"][sex]
        source = source_path.read_bytes()
        recipe = resolve_shape(model["characterId"], model["fbxSha256"]) if shaped else None
        tasks.append((sex, pose, model, source, recipe))
    settings = replace(RunnerSettings.from_env(), timeout_seconds=60,
                       shape_timeout_seconds=300, retarget_cache_bytes=0)
    output = {"kind": "recorded-source-real-approved-target-post-selection",
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "logical_cpus": os.cpu_count(), "cpu_quota": None,
                 "one_vcpu_conclusion": False, "blas_threads": 1},
        "inputs": [{"sex": sex, "pose": pose, "source_sha256": sha(source),
                    "character_id": model["characterId"], "character_sha256": model["fbxSha256"],
                    "shape": recipe is not None, "hands": {"left": "relaxed", "right": "relaxed"}}
                   for sex, pose, model, source, recipe in tasks],
        "source_solver_calls": 0, "vlm_calls": 0, "png_renders": 0,
        "warm_definition": "runner version cache/OS warm; all Blender processes cold-isolated",
        "runs": []}

    def run_one(runner, task, label, index):
        sex, pose, model, source, recipe = task
        started = time.monotonic()
        result = runner.convert(bvh_bytes=source, character_path=Path(model["fbxPath"]),
            character_id=model["characterId"], character_sha256=model["fbxSha256"],
            conversion_id=f"{label}-{index}", hands={"left": "relaxed", "right": "relaxed"},
            shape_recipe=recipe)
        convert_done = time.monotonic()
        preview, report = inspect_target(runner, result.artifact,
            {"output_scope": "full", "phase": "selected_target", "case": index})
        root = args.out / f"{label}-{index}"
        root.mkdir(exist_ok=True)
        (root / "final.fbx").write_bytes(result.artifact)
        (root / "preview.glb").write_bytes(preview)
        (root / "inspection.json").write_bytes(encode(report))
        # Frozen legacy diagnostics may contain Infinity; keep their established
        # encoding outside the authoritative finite inspection receipt.
        (root / "conversion.json").write_text(json.dumps(result.report, sort_keys=True))
        blocked = bool(report["mesh"]["hard_violations"] or report["hands"]["status"] == "intersection")
        return {"sex": sex, "pose": pose, "convert_ms": (convert_done-started)*1000,
            "queue_ms": result.queue_wait_ms, "execution_ms": result.execution_ms,
            "inspection_wall_ms": (time.monotonic()-convert_done)*1000,
            "inspection_worker_ms": report["elapsed_ms"],
            "fbx_serialization": result.report.get("selected_fbx_serialization"),
            "stages": result.report.get("lab_stage_timings"),
            "total_ms": (time.monotonic()-started)*1000,
            "artifact_sha256": sha(result.artifact), "artifact_bytes": len(result.artifact),
            "preview_bytes": len(preview), "structure": report["structure"],
            "baseline_restored": report["baseline_restored"],
            "baseline_sha256": report["baseline_sha256"],
            "correspondence": report["correspondence"], "mesh": report["mesh"],
            "hands": report["hands"], "skin_safe": False,
            "strict_adapter_outcome": "blocked_not_cached" if blocked else "structural_checked_not_skin_safe"}

    for capacity in (1, 2):
        runner = SelectedBlenderRunner(replace(settings, max_concurrent_processes=capacity))
        for mode in ("cold", "warm"):
            label = f"c{capacity}-{mode}"
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=capacity) as executor:
                futures = [executor.submit(run_one, runner, task, label, i)
                           for i, task in enumerate(tasks)]
                rows = [future.result() for future in futures]
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            run = {"concurrency": capacity, "mode": mode, "samples": len(tasks),
                "wall_ms": (time.monotonic()-started)*1000,
                "child_cpu_seconds": after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
                "child_peak_rss_bytes": after.ru_maxrss*(1 if sys.platform == "darwin" else 1024),
                "rss_scope": "process-lifetime child high-water; not sum of simultaneous child RSS",
                "rows": rows}
            output["runs"].append(run)
            (args.out / "raw.json").write_bytes(encode(output))
            print(encode({key: value for key, value in run.items() if key != "rows"}).decode(), flush=True)
    # A real successful STRUCTURAL result is cacheable. The same receipt still
    # explicitly says skin_safe=false; no unsupported certification is invented.
    sex, pose, model, source, recipe = tasks[1]
    runner = SelectedBlenderRunner(replace(settings, retarget_cache_bytes=32*1024*1024))
    selection = TargetSelection(model["characterId"], Path(model["fbxPath"]), model["fbxSha256"],
                                hands_json=encode({"left": "relaxed", "right": "relaxed"}))
    binding = {"phase": "selected_target", "owner_id": "local-benchmark", "job_id": "fixture",
        "person_id": "female", "candidate_id": "standing", "candidate_set_revision": "1",
        "selection_revision": 1, "prepared_source_id": "recorded-source",
        "context_fingerprint": "b"*64, "output_sha256": sha(source)}
    target = {"selection_id": "selected", "revision": sha(encode(selection.identity()))}
    provider = SelectedTargetProvider(runner, {"selected": selection}, is_current=lambda value: True)
    started = time.monotonic()
    prepared = provider.prepare(source, binding, target)
    miss_ms = (time.monotonic()-started)*1000
    started = time.monotonic()
    hit = provider.prepare(source, binding, target)
    hit_ms = (time.monotonic()-started)*1000
    started = time.monotonic()
    reused = provider.reuse(source, binding, target)
    reuse_ms = (time.monotonic()-started)*1000
    output["cache_hit"] = {"kind": "actual-approved-female-recorded-source-local-selected-binding",
        "miss_ms": miss_ms, "hit_ms": hit_ms, "reuse_ms": reuse_ms,
        "exact_object_and_bytes": prepared is hit is reused,
        "skin_safe": prepared.receipt["skin_safe"], "artifact_sha256": sha(prepared.artifact)}
    changed = replace(selection, hands_json=encode({"left": "fist", "right": "relaxed"}))
    provider.selections["selected"] = changed
    target["revision"] = sha(encode(changed.identity()))
    binding["selection_revision"] = 2
    started = time.monotonic()
    hand_changed = provider.prepare(source, binding, target)
    output["hand_change"] = {"wall_ms": (time.monotonic()-started)*1000,
        "body_cache": hand_changed.receipt["body_cache"], "source_solver_calls": 0,
        "skin_safe": hand_changed.receipt["skin_safe"]}
    (args.out / "raw.json").write_bytes(encode(output))


if __name__ == "__main__":
    main()
