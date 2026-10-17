"""Build a draft inventory from hashed authoring sources, without importing Blender code.

Source/FBX identity is verified here. Shape measurement and product QA are separate
work: a successful authoring contact report never grants rendering eligibility.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

MANIFEST_VERSION = "body-preset-manifest.v1"
BUILDER_VERSION = "body-assets-builder.v1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def repository_path(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError("asset_path_outside_repository")
    return path.relative_to(root).as_posix()


def file_reference(path, root, expected=None):
    path = Path(path).resolve()
    relative = repository_path(path, root)
    digest = sha256(path)
    if expected is not None and digest != expected:
        raise ValueError("asset_checksum_mismatch: " + relative)
    return dict(path=relative, sha256=digest)


def controller_literals(path):
    """Only literal top-level declarations; controller imports/functions never run."""
    fields = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"VERSION", "RIG_VERSION", "PRESETS"}:
                    if target.id in fields:
                        raise ValueError("duplicate_controller_declaration")
                    fields[target.id] = ast.literal_eval(node.value)
    if set(fields) != {"VERSION", "RIG_VERSION", "PRESETS"}:
        raise ValueError("missing_controller_declaration")
    if not all(isinstance(fields[k], str) and fields[k] for k in ("VERSION", "RIG_VERSION")):
        raise ValueError("invalid_controller_version")
    presets = fields["PRESETS"]
    if not isinstance(presets, dict) or not presets:
        raise ValueError("invalid_controller_presets")
    for key, value in presets.items():
        if (not isinstance(key, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key)
                or not isinstance(value, (tuple, list)) or len(value) != 2
                or not isinstance(value[0], str) or not value[0] or not isinstance(value[1], dict)):
            raise ValueError("invalid_controller_preset")
        for name, number in value[1].items():
            if (not isinstance(name, str) or not name or type(number) not in (int, float)
                    or not math.isfinite(number)):
                raise ValueError("invalid_controller_parameter")
    return fields


def build_manifest(delivery_path, repository_root):
    root, delivery_path = Path(repository_root).resolve(), Path(delivery_path).resolve()
    delivery = json.loads(delivery_path.read_text())
    if set(delivery.get("models", {})) != {"male", "female"}:
        raise ValueError("expected_male_and_female_sources")
    delivery_ref = file_reference(delivery_path, root)
    index_path = (delivery_path.parent / delivery["named_fbx_index"]).resolve()
    index_ref = file_reference(index_path, root)
    index = json.loads(index_path.read_text())
    neutrals, identities = {}, set()
    for record in index:
        key = (record["sex"], record["preset"], record["pose"])
        if key in identities:
            raise ValueError("duplicate_fbx_identity")
        identities.add(key)
        # Check paths even for posed fixtures that are not read by this builder.
        repository_path(delivery_path.parent / record["fbx"], root)
        if record["pose"] == "T":
            neutrals[key[:2]] = record
    sources, assets = {}, []
    for family in ("male", "female"):
        model = delivery["models"][family]
        version = model["version"]
        if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
            raise ValueError("invalid_source_version")
        model_path = Path(model["model"])
        controller_path = Path(model["controller"])
        model_path = model_path if model_path.is_absolute() else root / model_path
        controller_path = controller_path if controller_path.is_absolute() else root / controller_path
        model_ref = file_reference(model_path, root, model["model_sha256"])
        controller_ref = file_reference(controller_path, root, model["controller_sha256"])
        literals = controller_literals(controller_path)
        if not literals["VERSION"].endswith("-" + version):
            raise ValueError("controller_delivery_version_mismatch")
        if len(literals["PRESETS"]) != model["presets"]:
            raise ValueError("controller_delivery_preset_count_mismatch")
        sources[family] = dict(version=version, model=model_ref, controller=controller_ref,
                               rules=literals["VERSION"], rig_version=literals["RIG_VERSION"])
        for preset, (label, overrides) in literals["PRESETS"].items():
            record = neutrals.get((family, preset))
            if record is None:
                raise ValueError("neutral_fbx_missing: " + family + "/" + preset)
            fbx = file_reference(delivery_path.parent / record["fbx"], root, record["sha256"])
            qa_path = delivery_path.parent / family / (preset + "--T.json")
            qa_ref = file_reference(qa_path, root)
            qa = json.loads(qa_path.read_text())
            source_hashes = {str(model_path.resolve()): model_ref["sha256"],
                             str(controller_path.resolve()): controller_ref["sha256"]}
            recorded_hashes = {str((root / k).resolve()): v for k, v in qa.get("input_hashes", {}).items()}
            if (qa.get("sex") != family or qa.get("preset") != preset or qa.get("pose") != "T"
                    or qa.get("pass") is not True or recorded_hashes != source_hashes):
                raise ValueError("neutral_qa_source_mismatch: " + family + "/" + preset)
            body_id = family + "-" + preset.lower().replace("_", "-") + "-v" + version.replace(".", "-")
            assets.append(dict(body_id=body_id, character_id=body_id, source_family=family,
                body_version=version, preset=preset, display_name=label, neutral_fbx=fbx,
                authoring=dict(overrides=overrides, resolved_parameters=None,
                               resolution_status="requires_blender_mesh_defaults"),
                metadata=dict(presentation_style="masculine" if family == "male" else "feminine",
                              rendering_style="stylized", age_design=None),
                measurements=None, projections=[],
                readiness=dict(source_identity="verified", authoring_neutral_qa="passed",
                               measurement="pending", recommendation="pending", rendering="pending",
                               converter_pin="pending", runtime="pending"),
                authoring_qa=qa_ref, supported_pose_ids=[], availability="draft"))
    if set(neutrals) != {(a["source_family"], a["preset"]) for a in assets}:
        raise ValueError("unregistered_neutral_fbx")
    manifest = dict(schema_version=MANIFEST_VERSION, builder_version=BUILDER_VERSION,
        path_base="repository_root", status="draft_not_publishable", delivery=delivery_ref,
        neutral_fbx_index=index_ref, sources=sources, assets=assets,
        retired_for_new_analysis=["teen-male", "teen-female", "child"],
        legacy_job_policy="preserve_exact_asset_identity_no_automatic_replacement")
    validate_manifest(manifest, root)
    return manifest


def validate_manifest(manifest, repository_root):
    """Verify all referenced bytes and the draft-only contract before emission."""
    root = Path(repository_root).resolve()
    if (manifest.get("schema_version") != MANIFEST_VERSION
            or manifest.get("path_base") != "repository_root"
            or manifest.get("status") != "draft_not_publishable"):
        raise ValueError("invalid_preset_manifest")
    def verify(reference):
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise ValueError("invalid_asset_reference")
        if Path(reference["path"]).is_absolute():
            raise ValueError("manifest_reference_must_be_relative")
        file_reference(root / reference["path"], root, reference["sha256"])
    verify(manifest["delivery"])
    verify(manifest["neutral_fbx_index"])
    delivery = json.loads((root / manifest["delivery"]["path"]).read_text())
    expected_index = file_reference((root / manifest["delivery"]["path"]).parent / delivery["named_fbx_index"], root)
    if manifest["neutral_fbx_index"] != expected_index:
        raise ValueError("manifest_index_identity_mismatch")
    neutrals = {}
    for record in json.loads((root / expected_index["path"]).read_text()):
        if record["pose"] == "T":
            identity = (record["sex"], record["preset"])
            if identity in neutrals:
                raise ValueError("duplicate_fbx_identity")
            path = (root / manifest["delivery"]["path"]).parent / record["fbx"]
            neutrals[identity] = dict(path=repository_path(path, root), sha256=record["sha256"])
    if set(manifest["sources"]) != set(delivery["models"]) or set(manifest["sources"]) != {"male", "female"}:
        raise ValueError("manifest_source_coverage_mismatch")
    literals = {}
    for family, source in manifest["sources"].items():
        verify(source["model"])
        verify(source["controller"])
        recorded = delivery["models"][family]
        if (source["version"] != recorded["version"]
                or source["model"]["sha256"] != recorded["model_sha256"]
                or source["controller"]["sha256"] != recorded["controller_sha256"]):
            raise ValueError("manifest_delivery_source_mismatch")
        literals[family] = controller_literals(root / source["controller"]["path"])
        if source["rules"] != literals[family]["VERSION"] or source["rig_version"] != literals[family]["RIG_VERSION"]:
            raise ValueError("manifest_rules_mismatch")
    ids, presets = set(), set()
    for asset in manifest["assets"]:
        identity = (asset["source_family"], asset["preset"])
        if asset["body_id"] in ids or identity in presets:
            raise ValueError("duplicate_manifest_identity")
        ids.add(asset["body_id"])
        presets.add(identity)
        if (asset["availability"] != "draft" or asset["supported_pose_ids"]
                or asset["measurements"] is not None or asset["projections"]
                or any(asset["readiness"][k] != "pending" for k in
                       ("measurement", "recommendation", "rendering", "converter_pin", "runtime"))):
            raise ValueError("builder_cannot_grant_product_eligibility")
        source = manifest["sources"][asset["source_family"]]
        literal = literals[asset["source_family"]]
        expected_id = (asset["source_family"] + "-" + asset["preset"].lower().replace("_", "-")
                       + "-v" + source["version"].replace(".", "-"))
        if (asset["body_version"] != source["version"]
                or asset["body_id"] != expected_id or asset["character_id"] != expected_id
                or asset["preset"] not in literal["PRESETS"]
                or asset["authoring"]["overrides"] != literal["PRESETS"][asset["preset"]][1]
                or asset["authoring"]["resolved_parameters"] is not None
                or asset["metadata"]["presentation_style"] != ("masculine" if asset["source_family"] == "male" else "feminine")
                or asset["metadata"]["rendering_style"] != "stylized"):
            raise ValueError("manifest_source_definition_mismatch")
        if asset["neutral_fbx"] != neutrals.get(identity):
            raise ValueError("manifest_neutral_preset_mismatch")
        verify(asset["neutral_fbx"])
        verify(asset["authoring_qa"])
        qa = json.loads((root / asset["authoring_qa"]["path"]).read_text())
        expected_hashes = {str((root / source[k]["path"]).resolve()): source[k]["sha256"]
                           for k in ("model", "controller")}
        recorded_hashes = {str((root / k).resolve()): v for k, v in qa.get("input_hashes", {}).items()}
        if (qa.get("sex") != asset["source_family"] or qa.get("preset") != asset["preset"]
                or qa.get("pose") != "T" or qa.get("pass") is not True
                or recorded_hashes != expected_hashes):
            raise ValueError("neutral_qa_source_mismatch")
    expected = {(family, preset) for family, source in manifest["sources"].items()
                for preset in controller_literals(root / source["controller"]["path"])["PRESETS"]}
    if presets != expected:
        raise ValueError("manifest_preset_coverage_mismatch")


def consumer_catalogs(manifest, repository_root):
    validate_manifest(manifest, repository_root)
    fingerprint = hashlib.sha256(canonical(manifest)).hexdigest()
    version = "body-p0-" + fingerprint[:12]
    common = dict(source_manifest_sha256=fingerprint, release_status="draft_not_publishable")
    inference = dict(schema_version="body-catalog.v1", catalog_version=version,
                     default_body_id=None, assets=[], **common)
    for asset in manifest["assets"]:
        source = manifest["sources"][asset["source_family"]]
        inference["assets"].append(dict(body_id=asset["body_id"], body_version=asset["body_version"],
            availability="draft", asset_sha256=asset["neutral_fbx"]["sha256"],
            fbx_path=asset["neutral_fbx"]["path"], path_base="repository_root",
            rig_version=source["rig_version"], measurement_version="pending",
            metadata=asset["metadata"], attributes={}, projections=[], supported_pose_ids=[]))
    # Old BFFs do not check a draft flag. Do not put pending bodies in their assets array.
    pending = [a["body_id"] for a in manifest["assets"]]
    return {"manifest.json": manifest, "body-catalog.json": inference,
        "body-models.json": dict(version=version, defaultCharacterId=None, assets=[], pendingBodyIds=pending, **common),
        "characters.json": dict(schema_version=1, characters={}, pendingBodyIds=pending, **common)}


def write_draft_bundle(manifest, output, repository_root):
    """Publish all local draft files together; never replace an unrelated directory."""
    output = Path(output)
    catalogs = consumer_catalogs(manifest, repository_root)
    encoded = {name: json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
               for name, data in catalogs.items()}
    if output.exists():
        if set(p.name for p in output.iterdir()) == set(encoded) and all(
                (output / name).read_text() == text for name, text in encoded.items()):
            return
        raise ValueError("output_exists_with_different_content")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".body-draft-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        for name, text in encoded.items():
            (stage / name).write_text(text)
        os.replace(stage, output)
