from __future__ import annotations

"""Seal post-run CSP/avatar mesh evidence onto a frozen refine result."""

from pathlib import Path

from .labels import resolve_run
from .refine_mesh import (
    MESH_CHECK_CONTRACT_VERSION,
    MESH_REQUIRED_CHECKS,
    require_valid_mesh_evidence_bundle,
)
from .util import (
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)


def seal_mesh_evidence(
    run: str | Path,
    mesh_evidence: str | Path,
) -> Path:
    """Copy validated JSONL into the run and chain it to ``result_manifest``.

    This step is intentionally separate from the runner: the CSP/avatar mesh
    evaluator runs after BVHs exist.  Run it before human labels are collected.
    A different evidence bundle cannot silently replace an existing seal.
    """
    run_dir = resolve_run(run)
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("mode") != "refine_three_arm":
        raise ValueError("mesh evidence can only be sealed onto refine_three_arm runs")

    result_pointer = manifest.get("result_seal")
    if not isinstance(result_pointer, dict):
        raise ValueError("run has no sealed pre-label result manifest")
    result_path = run_dir / str(result_pointer.get("path") or "result_manifest.json")
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result_sha256 = sha256_file(result_path)
    if result_pointer.get("sha256") != result_sha256:
        raise ValueError("result manifest hash differs from the run seal")
    result_manifest = read_json(result_path)
    template_name = "mesh_safety_evidence.template.jsonl"
    template_path = run_dir / template_name
    template_metadata = (result_manifest.get("files") or {}).get(template_name)
    if not template_path.is_file():
        raise FileNotFoundError(
            "run has no frozen mesh safety evidence template"
        )
    if not isinstance(template_metadata, dict) or (
        template_metadata.get("sha256") != sha256_file(template_path)
        or template_metadata.get("bytes") != template_path.stat().st_size
    ):
        raise ValueError("mesh evidence template differs from the sealed result")
    template_rows = read_jsonl(template_path)
    template_validation = require_valid_mesh_evidence_bundle(
        template_rows, require_complete=False, allow_placeholders=True,
    )

    source = Path(mesh_evidence).expanduser().resolve()
    rows = read_jsonl(source)
    evidence_validation = require_valid_mesh_evidence_bundle(
        rows,
        expected_rows=template_rows,
        require_complete=True,
        allow_placeholders=False,
    )

    target = run_dir / "mesh_safety_evidence.jsonl"
    existing_pointer = manifest.get("evidence_seal")
    if target.exists():
        previous_rows = read_jsonl(target)
        if previous_rows != rows:
            raise FileExistsError(
                "a different mesh_safety_evidence.jsonl is already sealed for this run"
            )
    else:
        write_jsonl(target, rows)

    evidence_manifest_path = run_dir / "evidence_manifest.json"
    evidence_manifest = {
        "schema_version": 1,
        "run_id": manifest.get("run_id") or run_dir.name,
        "sealed_at": utc_now(),
        "result_manifest_sha256": result_sha256,
        "mesh_check_contract_version": MESH_CHECK_CONTRACT_VERSION,
        "required_checks": list(MESH_REQUIRED_CHECKS),
        "row_count": evidence_validation["row_count"],
        "template_sha256": template_metadata["sha256"],
        "template_validation": {
            "valid": template_validation["valid"],
            "row_count": template_validation["row_count"],
        },
        "files": {
            "mesh_safety_evidence.jsonl": {
                "sha256": sha256_file(target),
                "bytes": target.stat().st_size,
            },
        },
    }
    if evidence_manifest_path.exists():
        current = read_json(evidence_manifest_path)
        comparable = dict(current)
        comparable.pop("sealed_at", None)
        expected = dict(evidence_manifest)
        expected.pop("sealed_at", None)
        if comparable != expected:
            raise FileExistsError("a different evidence_manifest.json is already sealed")
    else:
        write_json(evidence_manifest_path, evidence_manifest)

    pointer = {
        "path": "evidence_manifest.json",
        "sha256": sha256_file(evidence_manifest_path),
    }
    if isinstance(existing_pointer, dict) and existing_pointer != pointer:
        raise FileExistsError("manifest already points to a different evidence seal")
    manifest["evidence_seal"] = pointer
    write_json(manifest_path, manifest)
    return evidence_manifest_path


__all__ = ["seal_mesh_evidence"]
