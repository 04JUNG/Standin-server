"""Blender 5.2 child-process entrypoint for exactly one conversion job.

The HTTP process never imports this module through ``converter.convert``.  The
heavy Blender imports are deliberately delayed until :func:`run_job`, after a
server-created job file has been validated.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from converter.protocol import (  # noqa: E402 - repo path is intentional
    ANKLE_POLICY_SHA256,
    APPLY_ROOT_TRANSLATION,
    EMBED_TEXTURES,
    EXPECTED_BLENDER_BUILD_HASH,
    EXPECTED_BLENDER_VERSION,
    FRAME,
    JOB_SCHEMA_VERSION,
    OUTPUT_MODE,
    RETARGET_SHA256,
    SOLVER_VERSION,
)


class JobValidationError(ValueError):
    """The runner/worker contract is malformed or unsafe."""


class FrozenLineageError(RuntimeError):
    """A promoted frozen solver file no longer matches its approved hash."""


class ArtifactValidationError(RuntimeError):
    """The worker claimed success without producing the contracted artifact."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise JobValidationError(f"{field} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise JobValidationError(f"{field} must be an absolute path")
    return path


def _inside(path: Path, root: Path, field: str) -> Path:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise JobValidationError(f"{field} must stay inside the job tempdir") from exc
    return resolved


def load_job(job_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and fail-closed validate a server-authored job JSON file."""
    path = Path(job_path)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise JobValidationError("job path must be an existing absolute regular file")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise JobValidationError("job JSON is unreadable") from exc
    if not isinstance(raw, dict):
        raise JobValidationError("job JSON must be an object")

    required = {
        "schema_version", "conversion_id", "temp_dir", "bvh_path",
        "character_fbx", "output_path", "report_path", "character_id",
        "character_sha256", "frame", "mirror", "output_mode",
        "apply_root_translation", "embed_textures",
    }
    unknown = set(raw) - required
    missing = required - set(raw)
    if missing or unknown:
        raise JobValidationError(
            f"job fields mismatch (missing={sorted(missing)}, unknown={sorted(unknown)})"
        )
    if raw["schema_version"] != JOB_SCHEMA_VERSION:
        raise JobValidationError("unsupported job schema_version")
    if not isinstance(raw["conversion_id"], str) or not raw["conversion_id"]:
        raise JobValidationError("conversion_id must be a non-empty string")
    if not isinstance(raw["character_id"], str) or not raw["character_id"]:
        raise JobValidationError("character_id must be a non-empty string")
    sha = raw["character_sha256"]
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise JobValidationError("character_sha256 must be lowercase SHA256")
    if type(raw["frame"]) is not int or raw["frame"] != FRAME:
        raise JobValidationError(f"frame must be locked to {FRAME}")
    if type(raw["mirror"]) is not bool:
        raise JobValidationError("mirror must be boolean")
    if raw["output_mode"] != OUTPUT_MODE:
        raise JobValidationError(f"output_mode must be {OUTPUT_MODE}")
    if raw["apply_root_translation"] is not APPLY_ROOT_TRANSLATION:
        raise JobValidationError("apply_root_translation must be false")
    if raw["embed_textures"] is not EMBED_TEXTURES:
        raise JobValidationError("embed_textures must be false")

    unresolved_temp = _absolute_path(raw["temp_dir"], "temp_dir")
    if unresolved_temp.is_symlink():
        raise JobValidationError("temp_dir must not be a symlink")
    temp_dir = unresolved_temp.resolve(strict=True)
    if not temp_dir.is_dir():
        raise JobValidationError("temp_dir must be an existing regular directory")
    if path.resolve() != _inside(path, temp_dir, "job path"):
        raise JobValidationError("job path must stay inside temp_dir")

    bvh_path = _inside(_absolute_path(raw["bvh_path"], "bvh_path"), temp_dir, "bvh_path")
    if not bvh_path.is_file() or bvh_path.is_symlink() or bvh_path.suffix.lower() != ".bvh":
        raise JobValidationError("bvh_path must be a regular .bvh file inside temp_dir")

    character = _absolute_path(raw["character_fbx"], "character_fbx").resolve(strict=True)
    if not character.is_file() or character.suffix.lower() != ".fbx":
        raise JobValidationError("character_fbx must be an existing .fbx file")

    output = _inside(_absolute_path(raw["output_path"], "output_path"), temp_dir, "output_path")
    report = _inside(_absolute_path(raw["report_path"], "report_path"), temp_dir, "report_path")
    if output.suffix.lower() != ".fbx" or report.suffix.lower() != ".json":
        raise JobValidationError("output/report extensions must be .fbx/.json")
    if output.exists() or report.exists():
        raise JobValidationError("output/report paths must not already exist")

    normalized = dict(raw)
    normalized.update({
        "temp_dir": str(temp_dir),
        "bvh_path": str(bvh_path),
        "character_fbx": str(character),
        "output_path": str(output),
        "report_path": str(report),
    })
    return normalized


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _frozen_lineage() -> dict[str, str]:
    retarget = Path(__file__).with_name("retarget.py")
    ankle_policy = Path(__file__).with_name("ankle_policy.json")
    actual_retarget = _sha256(retarget)
    actual_ankle = _sha256(ankle_policy)
    if actual_retarget != RETARGET_SHA256:
        raise FrozenLineageError("frozen retarget.py SHA256 mismatch")
    if actual_ankle != ANKLE_POLICY_SHA256:
        raise FrozenLineageError("frozen ankle_policy.json SHA256 mismatch")
    return {
        "retarget_sha256": actual_retarget,
        "ankle_policy_sha256": actual_ankle,
    }


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    """Execute one validated job inside Blender and persist its report."""
    lineage = _frozen_lineage()
    source_bvh_sha256 = _sha256(Path(job["bvh_path"]))
    from converter.convert import convert  # imports bpy only in this child process
    import bpy

    blender_version = bpy.app.version_string
    raw_build_hash = bpy.app.build_hash
    blender_build_hash = (
        raw_build_hash.decode("ascii", errors="replace")
        if isinstance(raw_build_hash, bytes) else str(raw_build_hash)
    )
    if not blender_version.startswith(EXPECTED_BLENDER_VERSION):
        raise FrozenLineageError("Blender version does not match the frozen runtime")
    if blender_build_hash != EXPECTED_BLENDER_BUILD_HASH:
        raise FrozenLineageError("Blender build hash does not match the frozen runtime")
    if _sha256(Path(job["character_fbx"])) != job["character_sha256"]:
        raise ArtifactValidationError("character artifact SHA256 changed before conversion")

    report = convert(
        bvh_path=job["bvh_path"],
        character_fbx=job["character_fbx"],
        out_path=job["output_path"],
        frame=FRAME,
        mirror=job["mirror"],
        output_mode=OUTPUT_MODE,
        apply_root_translation=APPLY_ROOT_TRANSLATION,
        embed_textures=EMBED_TEXTURES,
    )
    payload = report.as_dict()
    payload.update({
        "conversion_id": job["conversion_id"],
        "solver_version": SOLVER_VERSION,
        "blender_version": blender_version,
        "blender_build_hash": blender_build_hash,
        "character_id": job["character_id"],
        "character_sha256": job["character_sha256"],
        "source_bvh_sha256": source_bvh_sha256,
        **lineage,
    })
    output = Path(job["output_path"])
    if report.ok:
        if not output.is_file() or output.is_symlink() or output.stat().st_size <= 0:
            raise ArtifactValidationError(
                "converter reported success without a non-empty FBX artifact"
            )
        payload["artifact_sha256"] = _sha256(output)
        payload["artifact_size"] = output.stat().st_size
    else:
        payload["error_code"] = "conversion_rejected"
        payload["error_message"] = "converter rejected the BVH/profile mapping"
    return payload


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else (
        sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ))
    if len(args) != 2 or args[0] != "--job":
        print("[FAIL] usage: worker.py -- --job <absolute-job.json>", file=sys.stderr)
        return 2

    try:
        job = load_job(args[1])
    except Exception as exc:
        print(f"[FAIL] invalid job: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    report_path = Path(job["report_path"])
    try:
        payload = run_job(job)
        _write_report(report_path, payload)
        if not payload.get("ok"):
            print(f"[FAIL] conversion rejected: {job['conversion_id']}", file=sys.stderr)
            return 1
        print(f"[OK] conversion {job['conversion_id']}")
        return 0
    except Exception as exc:
        error_code = "internal_error" if isinstance(
            exc, (FrozenLineageError, ArtifactValidationError)
        ) else "invalid_input" if isinstance(exc, (ValueError, RuntimeError)) else "internal_error"
        payload = {
            "ok": False,
            "conversion_id": job["conversion_id"],
            "solver_version": SOLVER_VERSION,
            "character_id": job["character_id"],
            "character_sha256": job["character_sha256"],
            "error_code": error_code,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:500],
        }
        try:
            payload["source_bvh_sha256"] = _sha256(Path(job["bvh_path"]))
        except OSError:
            pass
        try:
            _write_report(report_path, payload)
        except Exception:
            pass
        print(f"[FAIL] conversion error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
