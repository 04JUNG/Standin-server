#!/usr/bin/env python3
"""Explicit live Gemini + Human-Art rough pilot; never a labeled accuracy test.

Uses copied DB/metadata and read-only BVHs/models. All overrides are process-local.
Runs off and evidence/shadow phases, preserving raw provider replies and usage.
No production requests, deployment, model fallback, or fabricated authority.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import io
import json
import os
import secrets
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--remote", action="store_true", required=True)
    p.add_argument("--images", type=Path, nargs="+", required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--env-file", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="gemini-3.8-flash")
    p.add_argument("--phases", nargs="+", choices=["off", "evidence_shadow"], default=["off", "evidence_shadow"])
    p.add_argument("--converter-url", default="")
    p.add_argument("--character-id")
    p.add_argument("--character-sha256")
    p.add_argument("--preflight-only", action="store_true")
    a = p.parse_args()
    a.images = [x.resolve() for x in a.images]
    a.manifest = a.manifest.resolve()
    from dotenv import dotenv_values
    key = dotenv_values(a.env_file).get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY_missing")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    data = out/"data"
    data.mkdir()
    source = a.data_root.resolve()
    for name in ("poses.db", "search-v2.json", "search-taxonomy-v2.json"):
        shutil.copyfile(source/name, data/name)
    for name in ("bvh", "thumbs"):
        (data/name).symlink_to(source/name, target_is_directory=True)
    # Never load all .env values: a local evaluation must not inherit production
    # URLs, credentials, flags or writable asset destinations from that file.
    os.environ.update(GEMINI_API_KEY=key, APP_ENV="development", DATA_DIR=str(data),
        VLM_PROVIDER="gemini", GEMINI_MODEL=a.model, GEMINI_MAX_ATTEMPTS="1",
        GEMINI_REQUEST_TIMEOUT_MS="30000", GEMINI_TOTAL_BUDGET_SECONDS="30",
        VLM_PROMPT_VERSION="p3-output-scope", VLM_PERSON_TAGS="1", VLM_HAND_SHAPES="1",
        VLM_HAND_SHAPES_TIMEOUT_MS="15000", BODY_MATCHING_MODE="off",
        SEARCH_ROUTING="0", ROUTING_FULLBODY_ENABLED="1", ROUTING_PREPARE_ASSETS="1",
        POSE_BACKEND="rtmlib", POSE_MODEL_VARIANT="humanart-m", POSE_CANARY_STAGE="canary-100",
        POSE_MODEL_MANIFEST=str(a.manifest.resolve()), POSE_CALIBRATION_FROM_MANIFEST="1",
        POSE_GENERATION_ENABLED="0", SKELETON_REPAIR="auto", SKELETON_REPAIR_TIMEOUT_S="30",
        REFINE_ENABLED="1", REFINE_V2_ENABLED="1", REFINE_V3_MODE="off",
        REFINE_V3_EVIDENCE_ENABLED="0", REFINE_V3_BINDING_SECRET=secrets.token_hex(32),
        REFINE_POSE_QUARANTINE_PATH=str(ROOT/"config/refine_pose_quarantine.v1.json"),
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    os.chdir(out)  # Library relative paths resolve to the isolated data snapshot.
    from src.config import CFG
    from src.repo import load_entries, get_pose_meta
    from src.fullbody_routing import _build_fullbody_resources
    from src.pipeline import Pipeline
    from src.vlm.client import GeminiVLMClient
    from src.vlm import prompts
    from src.hybrid_pose.gemini import GeminiReviewer
    from fastapi import Response, UploadFile
    from api import app as api
    from api.models import RefineRequest
    from google import genai
    from google.genai import types

    receipt = dict(schema="refine-v3-live-rough-pilot-v1", model_requested=a.model,
        phases=a.phases, generation_enabled=False,
        hand_screen_local_converter=bool(a.converter_url), hand_screen_character=a.character_id,
        scope="local analyze/refine handlers and serialization; excludes BFF, upload, preview and startup latency",
        independent_quality_labels=False, selected_source_policy="first selected library pose per detected slot",
        images=[dict(name=x.name, sha256=sha(x)) for x in a.images],
        source_db_sha256=sha(source/"poses.db"), model_manifest_sha256=sha(a.manifest),
        script_sha256=sha(__file__), deployed=False, production_registered=False)
    write(out/"receipt.json", receipt)
    started = time.perf_counter()
    entries = load_entries(str(data/"poses.db"))
    pose, semantic, engine = _build_fullbody_resources(entries, db_path=str(data/"poses.db"))
    if a.converter_url:
        from urllib.parse import urlparse
        from src.hand_skin_screen import HandSkinClient
        if urlparse(a.converter_url).hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("evaluation_requires_local_converter")
        if not a.character_id or not a.character_sha256:
            raise ValueError("exact_character_identity_required")
        engine.hand_skin_gate.client = HandSkinClient(a.converter_url, a.character_id, a.character_sha256)
    receipt.update(startup_seconds=time.perf_counter()-started, pose_identity=pose.runtime_identity(),
        library_poses=len({e.pose_id for e in entries}), startup_ok=True)
    write(out/"receipt.json", receipt)
    print("PREFLIGHT_OK", receipt["library_poses"], flush=True)
    if a.preflight_only:
        return
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=30000,
        retry_options=types.HttpRetryOptions(attempts=1)))
    calls = []
    model_class = type(client.models)
    generate_original = model_class.generate_content
    repair_original = GeminiReviewer.request

    def record_error(row, error):
        row.update(error=type(error).__name__, message=str(error).replace(key, "[REDACTED]")[:300])

    def generate(model_self, **kw):
        cfg = kw.get("config")
        system = getattr(cfg, "system_instruction", "") or ""
        kind = next((name for name, value in (
            ("person_tags", prompts.PERSON_TAGS_SYSTEM), ("hands", prompts.HAND_SHAPES_SYSTEM),
            ("describe", prompts.POSE_DESCRIBE_SYSTEM)) if system == value), "other")
        if system.startswith(prompts.SYSTEM):
            kind = "analysis"
        row = dict(kind=kind, transport="google-genai", model_requested=kw["model"])
        calls.append(row)
        started = time.perf_counter()
        try:
            response = generate_original(model_self, **kw)
            row.update(text=response.text, model_actual=response.model_version,
                usage=response.usage_metadata.model_dump() if response.usage_metadata else None)
            return response
        except Exception as error:
            record_error(row, error)
            raise
        finally:
            row["elapsed_seconds"] = time.perf_counter()-started

    def repair(reviewer, *args, **kw):
        row = dict(kind="skeleton_review_repair", transport="urllib", model_requested=reviewer.model)
        calls.append(row)
        started = time.perf_counter()
        try:
            value, raw = repair_original(reviewer, *args, **kw)
            row.update(payload=value, model_actual=raw.get("modelVersion"), usage=raw.get("usageMetadata"))
            return value, raw
        except Exception as error:
            record_error(row, error)
            raise
        finally:
            row["elapsed_seconds"] = time.perf_counter()-started

    model_class.generate_content = generate
    GeminiReviewer.request = repair
    records = []
    try:
        for phase in receipt["phases"]:
            shadow = phase == "evidence_shadow"
            os.environ["REFINE_V3_MODE"] = "shadow" if shadow else "off"
            os.environ["REFINE_V3_EVIDENCE_ENABLED"] = "1" if shadow else "0"
            pipe = Pipeline(entries, vlm_client=GeminiVLMClient(client=client, model=a.model),
                pose_model=pose, semantic_searcher=semantic, fullbody_search=engine)
            api.STATE.update(pipeline=pipe, provider="gemini", pose_backend="rtmlib",
                             db_path=str(data/"poses.db"), pose_library={})
            for image in a.images:
                calls = []
                dest = out/phase/image.stem
                dest.mkdir(parents=True)
                row = dict(phase=phase, image=image.name, image_sha256=sha(image),
                           error=None, selected_sources=[])
                started = time.perf_counter()
                try:
                    result = api.analyze(file=UploadFile(file=io.BytesIO(image.read_bytes()), filename=image.name),
                                         hint="", rescue="", response=Response())
                    row["analyze"] = json.loads(result.model_dump_json())
                except Exception as error:
                    row["error"] = dict(type=type(error).__name__, status=getattr(error,"status_code",None),
                                        message=str(error).replace(key,"[REDACTED]")[:300])
                row["analyze_seconds"] = time.perf_counter()-started
                for person in (row.get("analyze") or {}).get("people", []):
                    selected = (person.get("dynamic_search") or {}).get("selected", [])
                    if not selected:
                        continue
                    chosen = selected[0]
                    meta = get_pose_meta(str(data/"poses.db"), chosen["pose_id"])
                    source_path = Path(meta["bvh_path"]).resolve()
                    value = dict(person_index=person["index"], pose_id=chosen["pose_id"],
                        selected_rank=1, source_sha256=sha(source_path), source_kind="selected_base",
                        hands=person["hands"], camera=chosen.get("camera"),
                        structural_refine_allowed=person["refine_allowed"], refine_called=False)
                    candidate = next((c for c in person["refine_candidates"]
                                      if c["pose_id"] == chosen["pose_id"]), None)
                    if candidate and person.get("keypoints"):
                        fields = {k:person[k] for k in ("keypoints","scores","refine_allowed","refinable_limbs",
                            "lower_body_observed","skeleton_state","coverage_class","slot_origin","skeleton_source",
                            "search_stability","distance_metric","confidence_threshold","refine_v3")}
                        req = RefineRequest(pose_id=chosen["pose_id"], view=candidate["view"],
                            search_distance=candidate["distance"], camera=candidate["camera"],
                            render_thumbnail=False, **fields)
                        refined = api.refine(req).model_dump(mode="json")
                        value.update(refine_called=True, refine_result=refined)
                        if refined.get("refined") and refined.get("bvh"):
                            source_path = dest/f"person-{person['index']}-refined.bvh"
                            source_path.write_text(refined["bvh"])
                            value["source_kind"] = "f25_refined"
                    copied = dest/f"person-{person['index']}-selected.bvh"
                    shutil.copyfile(source_path, copied)
                    value.update(delivered_source_sha256=sha(copied), source_path=str(copied))
                    row["selected_sources"].append(value)
                row["calls"] = copy.deepcopy(calls)
                row["provider_request_count"] = len(calls)
                write(dest/"result.json", row)
                records.append(row)
                write(out/"records.json", records)
                print("CUT", phase, image.name, "seconds", round(row["analyze_seconds"],2),
                      "calls", len(calls), "people", len((row.get("analyze") or {}).get("people",[])),
                      "error", row["error"], flush=True)
                if row["error"] and not records[-1].get("analyze"):
                    # Avoid repeatedly spending requests on an unavailable model/key.
                    if any("NOT_FOUND" in c.get("message", "") or "API_KEY_INVALID" in c.get("message", "") for c in calls):
                        raise RuntimeError("provider_identity_or_credentials_rejected")
    finally:
        model_class.generate_content = generate_original
        GeminiReviewer.request = repair_original
        client.close()
    receipt.update(completed=True, cuts=len(records), provider_requests=sum(r["provider_request_count"] for r in records),
                   original_db_unchanged=sha(source/"poses.db") == receipt["source_db_sha256"])
    write(out/"receipt.json", receipt)


if __name__ == "__main__":
    main()
