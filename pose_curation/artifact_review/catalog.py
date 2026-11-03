"""Read the generated artifact receipts, never substitute historical thumbnails."""

import hashlib
import json
from pathlib import Path
import sqlite3


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class ArtifactCatalog:
    def __init__(self, base: Path):
        self.base = base.resolve()
        self.artifacts = self.base / "library-artifacts"
        with sqlite3.connect(self.base / "production/poses.db") as con:
            con.row_factory = sqlite3.Row
            metadata = {
                r["pose_id"]: dict(r) for r in con.execute("SELECT * FROM poses")
            }
        scenes = {
            r["pose_id"]: r
            for r in read(self.base / "final-scene-review-r7.json")["poses"]
        }
        old = {
            r["pose_id"]: r
            for r in read(self.base / "final-scene-review-r6.json")["poses"]
        }
        self.rows = {}
        self.files = {}
        for path in sorted((self.artifacts / "reports").glob("*.json")):
            report = read(path)
            if not report.get("ok"):
                continue
            pose = report["pose_id"]
            key = report["source_bvh_sha256"]
            raw = metadata.get(pose, {})
            meta = json.loads(raw.get("meta_json") or "{}")
            search = meta.get("search_v2", {})
            tags = [
                t.get("label") or t.get("value", "")
                for group in search.get("groups", {}).values()
                for t in group
            ]
            approved = scenes.get(pose, {}).get("approved", False)
            body_glb = self.artifacts / "pose-previews" / (report["model_key"] + ".glb")
            body_fbx = self.artifacts / "fbx" / (pose + ".fbx")
            self.files[(key, "full", "glb")] = body_glb
            self.files[(key, "full", "fbx")] = body_fbx
            scopes = ["full"]
            if approved:
                scene = read(
                    self.base / "scene-recovery-r7/poses" / pose / "report.json"
                )
                for scope in ("half", "bust", "shoulder_head"):
                    scopes.append(scope)
                    for kind in ("glb", "fbx"):
                        self.files[(key, scope, kind)] = Path(
                            scene["outputs"][scope][kind]
                        )
            # Source BVH remains a separate original-pose download.
            bvh = self.base / "production/bvh" / (pose + ".bvh")
            if bvh.is_file():
                self.files[(key, "source", "bvh")] = bvh
            asset_keys = [(key, s, k) for s in scopes for k in ("glb", "fbx")]
            if bvh.is_file():
                asset_keys.append((key, "source", "bvh"))
            for asset_key in asset_keys:
                asset = self.files[asset_key].resolve(strict=True)
                if not asset.is_relative_to(self.base):
                    raise ValueError("artifact escapes review root")
                self.files[asset_key] = asset
            if key in self.rows:
                raise ValueError("duplicate artifact source")
            self.rows[key] = dict(
                key=key,
                pose_id=pose,
                title=meta.get("style") or pose,
                group=meta.get("curation_group", "existing"),
                category=meta.get("category_label")
                or meta.get("category")
                or raw.get("action")
                or "기타",
                source=raw.get("source", ""),
                tags=tags,
                scopes=scopes,
                has_bvh=bvh.is_file(),
                scene_reviewed=approved,
                repaired=old.get(pose, {}).get("approved") is False,
                content_hash=report["skin"]["artifact_sha256"],
                description=search.get("description", {}).get("text", ""),
                fingers=report.get("source_fingers", {}).get("status", ""),
                _search=" ".join(
                    [
                        pose,
                        meta.get("style", ""),
                        raw.get("source", ""),
                        *tags,
                        search.get("description", {}).get("text", ""),
                    ]
                ).casefold(),
            )
        self.order = sorted(
            self.rows, key=lambda key: self.rows[key]["pose_id"].casefold()
        )
        self.load_plausibility()

    def load_plausibility(self):
        """Attach the automatic anatomy audit; keyed by source BVH sha256."""
        path = self.base / "plausibility/report.json"
        report = read(path) if path.is_file() else {"poses": {}}
        self.plausibility_version = report.get("version")
        self.corrected = self.base / "plausibility/corrected"
        for key, row in self.rows.items():
            audit = report["poses"].get(key)
            if not audit or audit.get("pose_id") != row["pose_id"]:
                row["auto"] = None
                continue
            fix = audit.get("correction")
            corrected = self.corrected / (row["pose_id"] + ".bvh")
            row["auto"] = dict(
                verdict=audit["verdict"],
                findings=audit["findings"],
                codes=sorted({f["code"] for f in audit["findings"]}),
                correction=None if not fix else dict(
                    verdict=fix["verdict"],
                    usable=fix["usable"] and corrected.is_file(),
                    max_move_torso=fix["max_move_torso"],
                    changed=fix["changed"],
                    ankle=fix.get("ankle"),
                ),
            )
            if row["auto"]["correction"] and row["auto"]["correction"]["usable"]:
                self.files[(key, "corrected", "bvh")] = corrected.resolve(strict=True)

    def public(self, key):
        return {k: v for k, v in self.rows[key].items() if not k.startswith("_")}

    def verify_fbx(self, key):
        path = self.files[(key, "full", "fbx")]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return digest == self.rows[key]["content_hash"]
