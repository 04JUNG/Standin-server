"""Private BFF boundary for native parts. No Blender imports in the HTTP host.

Only a hash-pinned operator catalog can supply geometry/rig recipes. Callers
send the current reviewed FBX, never filesystem paths, Python or donor files.
An accepted output is a review candidate; release still requires BFF review.
"""
import base64
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile

from . import VERSION
from src.scene_refinement.assets import inside
from src.scene_refinement.engine import fbx, digest, MAX_SOURCE, MAGIC

CONTRACT = "refine-v4-parts-http-v1"
SHA = re.compile(r"[0-9a-f]{64}")
ID = re.compile(r"[a-zA-Z0-9_-]{1,128}")


class PartsError(ValueError):
    def __init__(self, code, status=422):
        self.code, self.status = code, status
        super().__init__(code)


def text(value, limit=4096):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise PartsError("PARTS_INVALID_INPUT")
    return value


class Catalog:
    def __init__(self, path, expected_sha):
        self.path = Path(path).resolve(strict=True)
        raw = self.path.read_bytes()
        if not SHA.fullmatch(expected_sha) or digest(raw) != expected_sha:
            raise ValueError("parts catalog hash mismatch")
        data = json.loads(raw)
        review = data.get("review", {})
        if (data.get("version") != CONTRACT or review.get("status") != "approved"
                or not review.get("reference") or not isinstance(data.get("recipes"), list)
                or not data["recipes"] or not isinstance(data.get("files"), dict)):
            raise ValueError("approved parts catalog required")
        self.sha256 = expected_sha
        self.files = {}
        for name, sha in data.get("files", {}).items():
            path = inside(self.path.parent, name)
            if not isinstance(sha, str) or not SHA.fullmatch(sha) or digest(path.read_bytes()) != sha:
                raise ValueError("parts release file changed")
            self.files[name] = sha
        self.recipes = {}
        for recipe in data["recipes"]:
            identity = recipe.get("id")
            if not isinstance(identity, str) or not ID.fullmatch(identity) or identity in self.recipes:
                raise ValueError("unique parts recipe ID required")
            text(recipe.get("label"), 256)
            text(recipe.get("description"))
            request = recipe["request"]
            method = request.get("method")
            allowed = {"method", "part", "legs", "arms", "rig_profile"}
            if set(request) - allowed or method not in {"native_geometry", "directions", "arms"}:
                raise ValueError("unsupported catalog recipe")
            if method == "native_geometry":
                part = request["part"]
                if part.get("fbx") not in self.files or part.get("sha256") != self.files[part["fbx"]]:
                    raise ValueError("geometry donor must be pinned")
            if method == "directions" and request["legs"].get("part_file") not in self.files:
                raise ValueError("leg recipe must be pinned")
            if request.get("arms") and request.get("rig_profile") not in self.files:
                raise ValueError("arm calibration must be pinned")
            if method == "arms" and not request.get("arms"):
                raise ValueError("arm clauses required")
            self.recipes[identity] = copy.deepcopy(recipe)

    def materialize(self, identity, root, description):
        if not isinstance(identity, str) or identity not in self.recipes:
            raise PartsError("PARTS_RECIPE_UNAVAILABLE", 409)
        # The exact reviewed recipe sentence is the edit intent. Original VLM
        # descriptions belong in the BFF provenance, not a hidden override.
        if description != self.recipes[identity]["description"]:
            raise PartsError("PARTS_INTENT_MISMATCH", 409)
        request = copy.deepcopy(self.recipes[identity]["request"])
        # Copy only the exact files used by this recipe, into a private namespace.
        resources = []
        if request.get("method") == "native_geometry":
            resources.append((request["part"], "fbx"))
        if request.get("method") == "directions":
            resources.append((request["legs"], "part_file"))
        if request.get("arms"):
            resources.append((request, "rig_profile"))
        for index, (item, key) in enumerate(resources):
            name = item[key]
            data = inside(self.path.parent, name).read_bytes()
            if digest(data) != self.files[name]:
                raise PartsError("PARTS_RELEASE_CHANGED", 503)
            target = f"asset-{index}" + Path(name).suffix
            (root / target).write_bytes(data)
            item[key] = target
        return request


class PartsService:
    def __init__(self, *, root, blender, catalog=None, timeout=300):
        self.root, self.blender = Path(root), Path(blender)
        self.catalog, self.timeout = catalog, timeout
        self.root.mkdir(parents=True, exist_ok=True)

    def capabilities(self):
        return {"contract_version": CONTRACT, "engine_version": VERSION, "available": True,
                "operations": ["advance_foot"] + (["recipe"] if self.catalog else []),
                "catalog_sha256": self.catalog.sha256 if self.catalog else None,
                "recipes": [{"id": r["id"], "label": r["label"], "description": r["description"]}
                            for r in self.catalog.recipes.values()] if self.catalog else [],
                "automatic_image_extraction": False, "requires_explicit_intent": True,
                "requires_artifact_review": True}

    def run(self, path):
        subprocess.run([str(self.blender), "--background", "--factory-startup",
                        "--python-exit-code", "1", "--python",
                        str(Path(__file__).with_name("worker.py")), "--", str(path)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=self.timeout, check=True)

    def refine(self, payload):
        if not isinstance(payload, dict) or set(payload) - {
            "contract_version", "engine_version", "request_id", "current_asset", "intent", "operation", "context"
        }:
            raise PartsError("PARTS_INVALID_INPUT")
        if payload.get("contract_version") != CONTRACT or payload.get("engine_version") != VERSION:
            raise PartsError("PARTS_CONTRACT_UNSUPPORTED", 409)
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not ID.fullmatch(request_id):
            raise PartsError("PARTS_INVALID_INPUT")
        current = payload.get("current_asset", {})
        if not isinstance(current, dict) or set(current) != {"fbx_base64", "sha256", "review_reference"}:
            raise PartsError("PARTS_INVALID_INPUT")
        text(current["review_reference"], 256)
        original = fbx(current["fbx_base64"], current["sha256"])
        intent = payload.get("intent", {})
        if not isinstance(intent, dict) or set(intent) != {"source", "reference", "description"}:
            raise PartsError("PARTS_INVALID_INPUT")
        if intent["source"] not in {"user_confirmed", "reviewed_library"}:
            raise PartsError("PARTS_EXPLICIT_INTENT_REQUIRED", 409)
        text(intent["reference"], 256)
        text(intent["description"])
        context = payload.get("context")
        context_sha = None
        if context is not None:
            if not isinstance(context, dict) or set(context) != {
                "job_id", "person_index", "candidate_id", "input_sha256", "vlm_description", "user_instruction"
            }:
                raise PartsError("PARTS_INVALID_INPUT")
            text(context["job_id"], 256)
            text(context["candidate_id"], 256)
            if type(context["person_index"]) is not int or not 0 <= context["person_index"] <= 255:
                raise PartsError("PARTS_INVALID_INPUT")
            if not isinstance(context["input_sha256"], str) or not SHA.fullmatch(context["input_sha256"]):
                raise PartsError("PARTS_INVALID_INPUT")
            for field in ("vlm_description", "user_instruction"):
                if context[field] is not None:
                    text(context[field])
            context_sha = digest(json.dumps(context, sort_keys=True, allow_nan=False).encode())
        operation = payload.get("operation", {})
        if not isinstance(operation, dict):
            raise PartsError("PARTS_INVALID_INPUT")
        request_sha = digest(json.dumps(payload, sort_keys=True, allow_nan=False).encode())
        with tempfile.TemporaryDirectory(prefix="parts-", dir=self.root) as name:
            root = Path(name)
            (root / "current.fbx").write_bytes(original)
            if operation.get("type") == "advance_foot":
                if set(operation) != {"type", "side", "distance_m"} or operation["side"] not in {"left", "right"}:
                    raise PartsError("PARTS_INVALID_INPUT")
                distance = operation["distance_m"]
                if type(distance) not in (int, float) or not math.isfinite(distance) or not 0 < distance <= .1:
                    raise PartsError("PARTS_INVALID_INPUT")
                request = {"method": "advance_foot", "foot": {"side": operation["side"], "distance_m": distance}}
            elif operation.get("type") == "recipe" and set(operation) == {"type", "recipe_id"}:
                if not self.catalog:
                    raise PartsError("PARTS_RECIPE_UNAVAILABLE", 409)
                request = self.catalog.materialize(operation["recipe_id"], root, intent["description"])
            else:
                raise PartsError("PARTS_INVALID_INPUT")
            request.update(id=request_id, current_asset={"fbx": "current.fbx", "sha256": current["sha256"],
                           "confirmation": "user_confirmed", "evidence": current["review_reference"]})
            path = root / "request.json"
            path.write_text(json.dumps(request, allow_nan=False))
            try:
                self.run(path)
            except subprocess.TimeoutExpired:
                raise PartsError("PARTS_TIMEOUT", 504) from None
            except (subprocess.CalledProcessError, OSError):
                raise PartsError("PARTS_WORKER_FAILED", 502) from None
            try:
                receipt = json.loads((root / "result/receipt.json").read_text())
                result = (root / "result/after.fbx").read_bytes()
                status = receipt["status"]
                if (status not in {"applied", "unchanged", "not_applied"}
                        or receipt.get("version") != VERSION or receipt.get("id") != request_id
                        or receipt.get("source_sha256") != current["sha256"]
                        or not result.startswith(MAGIC) or len(result) > MAX_SOURCE
                        or digest(result) != receipt["after_sha256"]):
                    raise ValueError("invalid worker result")
                applied = status == "applied" and receipt.get("accepted") is True
                verification = receipt.get("saved_verification", {})
                if applied:
                    for field in ("matrix_error", "surface_error_m"):
                        value = verification.get(field)
                        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 2e-5:
                            raise ValueError("saved verification missing")
                    if verification.get("final_check", {}).get("accepted") is not True or result == original:
                        raise ValueError("unverified applied artifact")
                elif result != original:
                    raise ValueError("unapplied output must retain exact current bytes")
            except (OSError, KeyError, ValueError, TypeError):
                raise PartsError("PARTS_INTEGRITY", 502) from None
            # Detailed local Blender receipts can contain paths. The public
            # receipt exposes only typed, bounded verification/provenance.
            limbs = []
            if applied:
                if request["method"] == "advance_foot":
                    limbs.append(request["foot"]["side"] + "_leg")
                elif request["method"] == "native_geometry":
                    limbs.extend(side + "_leg" for side in request["part"]["sides"])
                elif request["method"] == "directions":
                    side = request["legs"]["selection"]
                    limbs.extend(s + "_leg" for s in (["left", "right"] if side == "both" else [side]))
                if request.get("arms"):
                    limbs.extend(side + "_arm" for side in ("left", "right"))
            return {"contract_version": CONTRACT, "engine_version": VERSION, "request_id": request_id,
                    "request_sha256": request_sha, "catalog_sha256": self.catalog.sha256 if self.catalog else None,
                    "context_sha256": context_sha,
                    "engine_executed": True, "status": "applied" if applied else "retained", "applied": applied,
                    "reason": None if applied else "PARTS_UNCHANGED" if status == "unchanged" else "PARTS_REJECTED",
                    "input_sha256": digest(original), "artifact_sha256": digest(result),
                    "fbx_base64": base64.b64encode(result).decode(),
                    "intent": intent, "operation": operation,
                    "adjusted_limbs": limbs,
                    "saved_verification": {k: verification[k] for k in ("matrix_error", "surface_error_m") if applied},
                    "artifact_contract": "verified-parts-fbx-v1" if applied else "retained-current-fbx-v1",
                    "requires_artifact_review": True, "release_allowed": False, "score_status": "not_scored"}
