"""Run bounded Blender workers and publish only verified, complete previews."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from ..storage import contained_path, read_json, sha256, utc_now, write_json
from ..qa.policy import CHARACTER_SHA256

VIEWS = {"front", "three_quarter", "side", "back"}
PROJECT = Path(__file__).resolve().parents[2]


def render_identity(character: Path, blender: Path) -> dict:
    character_hash = sha256(character)
    if character_hash != CHARACTER_SHA256:
        raise ValueError("expected the registered Standin Master V2 character")
    modules = sorted((PROJECT / "converter").glob("*.py"))
    modules += sorted((PROJECT / "converter").glob("*.json"))
    modules += [Path(__file__).with_name(name) for name in ("profiles.py", "scene.py", "worker.py", "neutral.sl", "fingers.py", "anatomy.py", "props.py")]
    modules += [PROJECT / "pose_curation" / "anatomy.py", PROJECT / "src" / "collision.py"]
    modules += [PROJECT / "pose_curation" / "hands" / "presets.py"]
    modules += [PROJECT / "pose_curation" / "qa" / "policy.py"]
    return {"renderer": "standin-character-v1", "character": "standin-master-v2",
            "character_sha256": character_hash, "blender_binary_sha256": sha256(blender),
            "code": {path.relative_to(PROJECT).as_posix(): sha256(path) for path in modules}}


def apply_result(batch: Path, pose: dict, result: dict, fingerprint: str, identity: dict) -> bool:
    """No manifest mutation until all four files and the BVH revision agree."""
    if (not result.get("ok") or result.get("render_fingerprint") != fingerprint
            or result.get("bvh_sha256") != pose["bvh_sha256"]):
        return False
    if (pose.get('prop_guides') or pose.get('support')) and result.get('scene_spec') != {k: pose.get(k, '') for k in ('prop_guides', 'support')}:
        return False
    if (pose.get("hand_augmentation") or pose.get("source_hand_joints") == 30) and result.get("fingers", {}).get("applied_joints") != 30:
        return False
    if sha256(contained_path(batch, pose["bvh"])) != pose["bvh_sha256"]:
        raise ValueError("BVH changed during rendering")
    thumbnails = result.get("thumbnails", {})
    if set(thumbnails) != VIEWS:
        return False
    verified = {}
    for view, item in thumbnails.items():
        path = contained_path(batch, item["path"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            return False
        verified[view] = {"path": path.relative_to(batch).as_posix(), "sha256": item["sha256"]}
    pose.update(thumbnails=verified, preview_kind="character", retarget_status="rendered_for_review",
                preview={"renderer": identity["renderer"], "character": identity["character"],
                         "character_sha256": identity["character_sha256"], "fingerprint": fingerprint,
                         "blender_version": result["blender_version"],
                         "report": f"render-results/{fingerprint[:16]}/{pose['pose_id']}.json"})
    if result.get('anatomy'):
        pose['anatomy_check'] = {**result['anatomy'], 'bvh_sha256': pose['bvh_sha256'], 'render_fingerprint': fingerprint}
    return True


def run(batch: Path, blender: Path, character: Path, *, workers: int = 2,
        limit: int | None = None, pose_ids: list[str] | None = None) -> dict:
    if not 1 <= workers <= 4 or (limit is not None and limit < 1):
        raise ValueError("workers must be 1..4 and limit must be positive")
    batch, blender, character = batch.resolve(), blender.resolve(), character.resolve()
    identity = render_identity(character, blender)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest_path = batch / "manifest.json"
    manifest = read_json(manifest_path)
    poses = manifest["poses"][:limit] if limit else manifest["poses"]
    if pose_ids is not None:
        unknown = set(pose_ids) - {p['pose_id'] for p in poses}
        if unknown:raise ValueError(f'unknown render pose IDs: {sorted(unknown)}')
        poses = [p for p in poses if p['pose_id'] in pose_ids]
    result_dir = batch / "render-results" / fingerprint[:16]
    output_dir = batch / "character-thumbs" / fingerprint[:16]
    result_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for pose in poses:
        result_path = result_dir / f"{pose['pose_id']}.json"
        if result_path.is_file() and apply_result(batch, pose, read_json(result_path), fingerprint, identity):
            continue
        if result_path.is_file():
            result_path.unlink()  # retry only this invalid per-pose render result
        pending.append(pose)

    progress = {"status": "rendering", "requested": len(poses), "completed": len(poses) - len(pending),
                "fingerprint": fingerprint, "identity": identity, "failures": []}

    def publish() -> None:
        manifest["character_render"] = progress
        manifest["updated_at"] = utc_now()
        write_json(manifest_path, manifest)

    publish()
    jobs = [pending[i::workers] for i in range(workers) if pending[i::workers]]
    processes = []
    seen = set()
    with ExitStack() as stack:
        try:
            for index, rows in enumerate(jobs):
                job_path = result_dir / f"job-{index}.json"
                write_json(job_path, {"fingerprint": fingerprint, "character": str(character),
                    "output_dir": str(output_dir), "result_dir": str(result_dir),
                    "poses": [{"pose_id": p["pose_id"], "bvh": str(contained_path(batch, p["bvh"])),
                               "bvh_sha256": p["bvh_sha256"], "rig_profile": p.get("rig_profile", "100style"),
                               "prop_guides": p.get('prop_guides', ''), "support": p.get('support', '')} for p in rows]})
                log = stack.enter_context((result_dir / f"worker-{index}.log").open("w", encoding="utf-8"))
                env = dict(os.environ, BLENDER_USER_RESOURCES=str(result_dir / f"blender-user-{index}"))
                command = [str(blender), "--background", "--factory-startup", "--threads", "2",
                           "--python-exit-code", "1", "--python", str(Path(__file__).with_name("worker.py")),
                           "--", str(job_path)]
                processes.append(subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
            while True:
                active = any(process.poll() is None for process in processes)
                changed = False
                for pose in pending:
                    pid = pose["pose_id"]
                    path = result_dir / f"{pid}.json"
                    if pid in seen or not path.is_file():
                        continue
                    result = read_json(path)
                    if apply_result(batch, pose, result, fingerprint, identity):
                        progress["completed"] += 1
                        print(f"Character previews {progress['completed']}/{len(poses)}: {pid}", flush=True)
                    else:
                        progress["failures"].append({"pose_id": pid, "error": result.get("error", "invalid render output")})
                    seen.add(pid)
                    changed = True
                if changed:
                    publish()
                if not active:
                    break
                time.sleep(1)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                process.wait()
    for pose in pending:
        if pose["pose_id"] not in seen:
            progress["failures"].append({"pose_id": pose["pose_id"], "error": "Blender exited without a result; see worker log"})
    progress["status"] = "complete" if progress["completed"] == len(poses) else "partial"
    publish()
    return progress
