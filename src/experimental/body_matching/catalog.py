"""Immutable per-request catalog snapshot with verified asset provenance."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

from .schema import AXES, GEOMETRY_AXES, GEOMETRY_VERSION, finite_number, valid_axis_value


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class Catalog:
    version: str
    sha256: str
    default_body_id: str | None
    assets: tuple[dict, ...]
    issues: tuple[dict, ...]
    presentation_defaults: dict = field(default_factory=dict)

    def eligible(self, pose_ids):
        # Empty/partial search results are not filled with invented poses.
        return [a for a in self.assets if a["availability"] == "eligible"
                and set(pose_ids).issubset(a["supported_pose_ids"])]


def load_catalog(path):
    path = Path(path).resolve()
    content = path.read_bytes()
    raw = json.loads(content)
    if raw.get("schema_version") != "body-catalog.v1" or not isinstance(raw.get("catalog_version"), str) or not raw["catalog_version"]:
        raise ValueError("invalid_body_catalog_version")
    if not isinstance(raw.get("assets"), list):
        raise ValueError("invalid_body_catalog_assets")
    ids, assets, issues = set(), [], []
    for item in raw["assets"]:
        asset = dict(item)
        body_id = asset.get("body_id")
        if not isinstance(body_id, str) or not body_id or body_id in ids:
            raise ValueError("duplicate_or_invalid_body_id")
        ids.add(body_id)
        if asset.get("availability") not in {"draft", "qa_pending", "eligible", "retired"}:
            raise ValueError("invalid_body_availability")
        metadata = asset.get("metadata", {})
        if not isinstance(metadata, dict) or metadata.get("presentation_style", "unspecified") not in {"feminine", "masculine", "unisex", "unspecified"}:
            raise ValueError("invalid_body_presentation_metadata")
        attrs = asset.get("attributes", {})
        if not isinstance(attrs, dict) or any(k not in AXES or (v is not None and not valid_axis_value(k, v)) for k, v in attrs.items()):
            raise ValueError("invalid_body_asset_attributes")
        asset["attributes"] = {key: attrs.get(key) for key in AXES}
        if asset["availability"] == "eligible":
            for key in ("body_version", "rig_version", "measurement_version", "qa_report", "fbx_path"):
                if not isinstance(asset.get(key), str) or not asset[key]:
                    raise ValueError("eligible_body_missing_" + key)
            if not is_sha(asset.get("asset_sha256")) or not is_sha(asset.get("qa_sha256")):
                raise ValueError("invalid_body_asset_hash")
            support = asset.get("supported_pose_ids")
            if not isinstance(support, list) or not support or any(not isinstance(x, str) or not x or x == "*" for x in support) or len(set(support)) != len(support):
                raise ValueError("explicit_pose_support_required")
            if not finite_number(asset.get("quality_priority", 0), 0, 100):
                raise ValueError("invalid_body_quality_priority")
            fbx = (path.parent / asset["fbx_path"]).resolve()
            qa = (path.parent / asset["qa_report"]).resolve()
            # Failed assets are quarantined individually; other verified assets remain usable.
            if fbx.suffix.lower() != ".fbx" or not fbx.is_file() or not qa.is_file():
                asset["availability"] = "qa_pending"
                issues.append({"body_id": body_id, "reason": "asset_or_qa_missing"})
            elif file_sha256(fbx) != asset["asset_sha256"] or file_sha256(qa) != asset["qa_sha256"]:
                asset["availability"] = "qa_pending"
                issues.append({"body_id": body_id, "reason": "asset_or_qa_hash_mismatch"})
            else:
                report = json.loads(qa.read_text())
                if (report.get("status") != "passed" or any(report.get(k) != asset[k] for k in
                        ("body_id", "body_version", "asset_sha256", "rig_version"))
                        or set(report.get("supported_pose_ids", [])) != set(support)):
                    asset["availability"] = "qa_pending"
                    issues.append({"body_id": body_id, "reason": "qa_identity_or_support_mismatch"})
        projections = asset.get("projections", [])
        validated_projections = []
        keys = set()
        for projection in projections:
            key = (projection.get("pose_id"), projection.get("view"), projection.get("pose_sha256"))
            if (key in keys or not isinstance(key[0], str) or not isinstance(key[1], str)
                    or not is_sha(key[2]) or projection.get("geometry_version") != GEOMETRY_VERSION):
                raise ValueError("invalid_body_projection_key")
            keys.add(key)
            ratios = projection.get("ratios", {})
            if not isinstance(ratios, dict) or not ratios or any(k not in GEOMETRY_AXES or not finite_number(v) for k, v in ratios.items()):
                raise ValueError("invalid_body_projection_ratios")
            if (projection.get("target_asset_sha256") != asset.get("asset_sha256")
                    or projection.get("rig_version") != asset.get("rig_version")
                    or not is_sha(projection.get("target_asset_sha256"))):
                issues.append({"body_id": body_id, "reason": "projection_asset_mismatch"})
                continue
            validated_projections.append(projection)
        asset["projections"] = validated_projections
        assets.append(asset)
    default = raw.get("default_body_id")
    if default is not None and default not in ids:
        raise ValueError("unknown_default_body_id")
    defaults = raw.get("presentation_defaults", {})
    if not isinstance(defaults, dict) or any(k not in {"feminine", "masculine", "androgynous"} or not isinstance(v,str) or v not in ids for k,v in defaults.items()):
        raise ValueError("invalid_presentation_defaults")
    by_id = {a["body_id"]: a for a in assets}
    if any(by_id[v].get("metadata",{}).get("presentation_style") != ("unisex" if k=="androgynous" else k) for k,v in defaults.items()):
        raise ValueError("presentation_default_metadata_mismatch")
    return Catalog(raw["catalog_version"], hashlib.sha256(content).hexdigest(), default, tuple(assets), tuple(issues), dict(defaults))
