#!/usr/bin/env python3
"""Explicitly authorized full-cut evaluation of shared first-call body analysis.

Measures the inference API handler plus JSON serialization, never a crop-only
proxy for pipeline latency. Requires actual pose backend/library assets.
"""
from pathlib import Path
import argparse, copy, hashlib, io, json, os, shutil, sys, time


def iou(a, b):
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = x * y
    return inter / max(
        1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    )


def match_people(cases, people, min_iou=0.3):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    out = {c["id"]: None for c in cases}
    if not cases or not people:
        return out
    matrix = np.array(
        [
            [iou(c["box"], p["box"]) if p.get("box") else 0 for p in people]
            for c in cases
        ]
    )
    row, col = linear_sum_assignment(-matrix)
    for a, b in zip(row, col):
        if matrix[a, b] >= min_iou:
            out[cases[a]["id"]] = dict(index=int(b), iou=float(matrix[a, b]))
    return out


def pose_signature(person):
    """Compare the active contract, including develop's fullbody selected poses."""
    if not person:
        return []
    dynamic = person.get("dynamic_search")
    if dynamic is not None:
        return [
            (
                x["pose_id"],
                x.get("kind"),
                json.dumps(
                    x.get(
                        "camera",
                        x.get("evidence", {}).get("geometry", {}).get("camera"),
                    ),
                    sort_keys=True,
                ),
            )
            for x in dynamic.get("selected", [])
        ]
    return [(x["pose_id"], x["view"]) for x in person.get("candidates", [])]


def same_successful_count(before, after):
    return bool(
        before.get("error") is None
        and after.get("error") is None
        and before.get("result") is not None
        and after.get("result") is not None
        and before["result"].get("vlm_count") == after["result"].get("vlm_count")
    )


def primary_payload(calls):
    from src.vlm.client import _extract_json

    for call in calls:
        if call.get("kind") == "analysis" and call.get("text"):
            try:
                return _extract_json(call["text"])
            except (TypeError, ValueError):
                return None
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--env-file", type=Path, required=True)
    p.add_argument("--remote", action="store_true", required=True)
    p.add_argument("--profile", choices=["local-current-x", "fullbody"], required=True)
    p.add_argument("--model", default="gemini-3.8-flash")
    p.add_argument("--thinking-level", default="low")
    p.add_argument("--request-timeout-seconds", type=float, default=30)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--hand-shapes", action="store_true")
    p.add_argument("--person-tags", action="store_true")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from dotenv import dotenv_values

    key = dotenv_values(a.env_file).get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY_missing")
    os.environ["GEMINI_API_KEY"] = key
    root = a.data_root.resolve()
    a.out = a.out.resolve()
    # No .env mutation. All profile overrides are local to this evaluation process.
    env = dict(
        APP_ENV="development",
        REFINE_POSE_QUARANTINE_PATH=str(
            Path(__file__).resolve().parents[1]
            / "config/refine_pose_quarantine.v1.json"
        ),
        VLM_PROVIDER="gemini",
        GEMINI_MODEL=a.model,
        VLM_PROMPT_VERSION="p3-output-scope",
        GEMINI_MAX_ATTEMPTS="1",
        GEMINI_REQUEST_TIMEOUT_MS=str(int(a.request_timeout_seconds * 1000)),
        GEMINI_TOTAL_BUDGET_SECONDS=str(a.request_timeout_seconds),
        VLM_PERSON_TAGS="1" if a.person_tags else "0",
        SEARCH_ROUTING="0",
        ROUTING_PREPARE_ASSETS="1" if a.profile == "fullbody" else "0",
        VLM_HAND_SHAPES="1" if a.hand_shapes else "0",
        VLM_BODY_THINKING_LEVEL=a.thinking_level,
        BODY_MATCHING_MODE="off",
        BODY_CLASSIFICATION_STRATEGY="integrated",
        DATA_DIR=str(root),
        ROUTING_FULLBODY_ENABLED="1" if a.profile == "fullbody" else "0",
        POSE_BACKEND="rtmlib",
        POSE_MODEL_VARIANT="humanart-m" if a.profile == "fullbody" else "current-x",
        POSE_CANARY_STAGE="canary-100" if a.profile == "fullbody" else "off",
    )
    if a.manifest:
        env["POSE_MODEL_MANIFEST"] = str(a.manifest.resolve())
        env["POSE_CALIBRATION_FROM_MANIFEST"] = "1"
    os.environ.update(env)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import CFG
    from src.repo import load_entries
    from src.vlm.client import GeminiVLMClient, _coerce
    from src.vlm import prompts
    from src.pipeline import Pipeline
    from src.pose import RTMPoseModel
    from src.experimental.body_matching.integrated import VERSION
    from fastapi import Response, UploadFile
    from api import app as api
    from google import genai
    from google.genai import types

    protocol = json.loads(a.protocol.read_text())
    reference = json.loads(a.reference.read_text())
    refs = {x["case_id"]: x for x in reference["rows"]}
    cases = protocol["cases"]
    if set(refs) != {c["id"] for c in cases}:
        raise ValueError("reference_case_mismatch")
    cuts = {}
    for c in cases:
        source = Path(c["image"])
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != c["image_sha256"] or digest != refs[c["id"]]["image_sha256"]:
            raise ValueError("input_hash_mismatch")
        cuts.setdefault(c["cut_id"], []).append(c)
    receipt = dict(
        develop_commit="58c691ea",
        profile=a.profile,
        model=a.model,
        thinking_level=a.thinking_level,
        integrated_version=VERSION,
        scope="local inference API handler + JSON serialization; excludes BFF/network upload/startup",
        hardware=sys.platform,
        hand_shapes=a.hand_shapes,
        person_tags=a.person_tags,
        fullbody_enabled=a.profile == "fullbody",
        library_source_sha256=hashlib.sha256(
            (root / "poses.db").read_bytes()
        ).hexdigest(),
        protocol_sha256=hashlib.sha256(a.protocol.read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),
        thresholds=dict(match_iou=0.3, latency_seconds=12),
        repeats=a.repeats,
        resumed=a.resume,
    )
    (a.out / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2)
    )
    os.chdir(
        root.parent
    )  # Existing relative BVH paths are rooted at the actual library.
    snapshot = a.out / "library.db"
    shutil.copy2(root / "poses.db", snapshot)
    startup = time.perf_counter()
    try:
        entries = load_entries(str(snapshot))
        if a.profile == "fullbody":
            from src.fullbody_routing import _build_fullbody_resources

            pose, semantic, engine = _build_fullbody_resources(
                entries, db_path=str(snapshot)
            )
        else:
            pose, semantic, engine = RTMPoseModel(), None, None
    except Exception as e:
        receipt.update(
            startup_error=type(e).__name__,
            startup_message=str(e)[:250],
            startup_seconds=time.perf_counter() - startup,
        )
        (a.out / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2)
        )
        print("STARTUP_BLOCKED", receipt["startup_message"], flush=True)
        return
    receipt.update(
        startup_seconds=time.perf_counter() - startup,
        pose_identity=(
            pose.runtime_identity() if hasattr(pose, "runtime_identity") else {}
        ),
        pose_adapter=type(pose).__name__,
        library_poses=len({e.pose_id for e in entries}),
    )
    (a.out / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2)
    )
    print("STARTUP", json.dumps(receipt, ensure_ascii=False), flush=True)
    outputs = {}
    for repeat in range(a.repeats):
        for phase in ["baseline", "integrated"]:
            CFG.body_matching_mode = "off" if phase == "baseline" else "shadow"
            CFG.body_coarse_catalog_path = str(
                Path(__file__).resolve().parents[1] / "config/body_morphology.v1.json"
            )
            client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    timeout=int(a.request_timeout_seconds * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            vlm = GeminiVLMClient(client=client, model=a.model)
            calls = []
            model_class = type(client.models)
            real = model_class.generate_content

            def generate(model_self, **kw):
                cfg = kw["config"]
                system = cfg.system_instruction
                if system == prompts.PERSON_TAGS_SYSTEM:
                    kind = "person_tags"
                elif system == prompts.HAND_SHAPES_SYSTEM:
                    kind = "hands"
                elif system == prompts.POSE_DESCRIBE_SYSTEM:
                    kind = "describe"
                elif system and system.startswith(prompts.SYSTEM):
                    kind = "analysis"
                else:
                    kind = "other"
                # Controlled comparison: the first call uses identical thinking in both phases.
                if kind == "analysis" and a.thinking_level:
                    cfg.thinking_config = types.ThinkingConfig(
                        thinking_level=a.thinking_level
                    )
                started = time.perf_counter()
                row = dict(kind=kind, model=kw["model"])
                sink = calls
                sink.append(row)
                try:
                    response = real(model_self, **kw)
                    row.update(
                        text=response.text,
                        model_actual=response.model_version,
                        usage=response.usage_metadata.model_dump(),
                    )
                    return response
                except Exception as e:
                    row["error"] = type(e).__name__
                    chain = []
                    cause = e
                    while cause is not None and len(chain) < 5:
                        chain.append(
                            dict(
                                type=type(cause).__name__,
                                message=str(cause).replace(key, "[REDACTED]")[:250],
                            )
                        )
                        cause = cause.__cause__
                    row["error_chain"] = chain
                    raise
                finally:
                    row["elapsed_seconds"] = time.perf_counter() - started

            model_class.generate_content = generate
            pipe = Pipeline(
                entries,
                vlm_client=vlm,
                pose_model=pose,
                semantic_searcher=semantic,
                fullbody_search=engine,
            )
            api.STATE.update(
                pipeline=pipe,
                provider="gemini",
                pose_backend="rtmlib",
                db_path=str(snapshot),
                pose_library={},
            )
            for cut, group in cuts.items():
                dest = a.out / f"{phase}-{repeat}" / cut
                dest.mkdir(parents=True, exist_ok=True)
                if a.resume and (dest / "result.json").exists():
                    previous = json.loads((dest / "result.json").read_text())
                    outputs[(repeat, phase, cut)] = previous
                    print("RESUMED", repeat, phase, cut, flush=True)
                    continue
                calls = []
                response = Response()
                started = time.perf_counter()
                result = None
                error = None
                try:
                    result = api.analyze(
                        file=UploadFile(
                            file=io.BytesIO(Path(group[0]["image"]).read_bytes()),
                            filename=cut + ".png",
                        ),
                        hint="",
                        rescue="",
                        response=response,
                    )
                    encoded = result.model_dump_json()
                    result = json.loads(encoded)
                except Exception as e:
                    error = dict(
                        type=type(e).__name__, status=getattr(e, "status_code", None)
                    )
                elapsed = time.perf_counter() - started
                primary = primary_payload(calls)
                record = dict(
                    cut_id=cut,
                    phase=phase,
                    repeat=repeat,
                    elapsed_seconds=elapsed,
                    within_12_seconds=elapsed <= 12 and error is None,
                    error=error,
                    result=result,
                    primary=primary,
                    server_timing=response.headers.get("server-timing"),
                    calls=copy.deepcopy(calls),
                )
                (dest / "result.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2)
                )
                outputs[(repeat, phase, cut)] = record
                print(
                    "CUT",
                    repeat,
                    phase,
                    cut,
                    round(elapsed, 3),
                    "calls",
                    len(calls),
                    "people",
                    len(result["people"]) if result else None,
                    "error",
                    error,
                    flush=True,
                )
            model_class.generate_content = real
            client.close()
    rows = []
    for repeat in range(a.repeats):
        for cut, group in cuts.items():
            before = outputs[(repeat, "baseline", cut)]
            after = outputs[(repeat, "integrated", cut)]
            pa = (before["result"] or {}).get("people", [])
            pb = (after["result"] or {}).get("people", [])
            ma = match_people(group, pa)
            mb = match_people(group, pb)
            for c in group:
                ident = c["id"]
                label = refs[ident]
                m = mb[ident]
                old = ma[ident]
                person = pb[m["index"]] if m else None
                body_people = (
                    (after["result"] or {}).get("body_matching", {}).get("people", [])
                )
                rec = (
                    body_people[m["index"]].get("coarse_recommendation")
                    if m and m["index"] < len(body_people)
                    else None
                )

                ta = pose_signature(pa[old["index"]] if old else None)
                tb = pose_signature(person)
                rows.append(
                    dict(
                        repeat=repeat,
                        case_id=ident,
                        expected_family=label["expected_family"],
                        expected_body_id=label["expected_body_id"],
                        match=m,
                        baseline_match=old,
                        recommendation=rec,
                        family_correct=bool(
                            rec
                            and rec["recommended_family"] == label["expected_family"]
                        ),
                        observed_correct=bool(
                            rec and rec["observed_family"] == label["expected_family"]
                        ),
                        preset_correct=bool(
                            rec
                            and rec["recommended_body_id"] == label["expected_body_id"]
                        ),
                        baseline_top_k=ta,
                        integrated_top_k=tb,
                        pose_top_k_equal=ta == tb and bool(old and m),
                        pose_top1_equal=bool(ta and tb and ta[0] == tb[0]),
                        baseline_people=len(pa),
                        integrated_people=len(pb),
                    )
                )
    import statistics

    phases = {}
    for phase in ["baseline", "integrated"]:
        rs = [v for k, v in outputs.items() if k[1] == phase]
        times = [r["elapsed_seconds"] for r in rs if r["elapsed_seconds"] is not None]
        successful_times = [
            r["elapsed_seconds"]
            for r in rs
            if r["elapsed_seconds"] is not None and r["error"] is None
        ]
        phases[phase] = dict(
            cuts=len(rs),
            unmeasured_failures=sum(r["elapsed_seconds"] is None for r in rs),
            successful=sum(r["error"] is None for r in rs),
            within_12_seconds=sum(r["within_12_seconds"] for r in rs),
            median_seconds=statistics.median(times),
            max_seconds=max(times),
            median_successful_seconds=(
                statistics.median(successful_times) if successful_times else None
            ),
            p95_successful_seconds=(
                float(__import__("numpy").percentile(successful_times, 95))
                if successful_times
                else None
            ),
            p95_seconds=float(__import__("numpy").percentile(times, 95)),
            call_counts=[len(r["calls"]) for r in rs],
        )
    summary = dict(
        reference_scope=reference["scope"],
        receipt=receipt,
        phases=phases,
        total_cases=len(rows),
        family_correct=sum(r["family_correct"] for r in rows),
        observed_correct=sum(r["observed_correct"] for r in rows),
        preset_correct=sum(r["preset_correct"] for r in rows),
        matched=sum(r["match"] is not None for r in rows),
        count_equal_cuts=sum(
            same_successful_count(
                outputs[(i, "baseline", c)], outputs[(i, "integrated", c)]
            )
            for i in range(a.repeats)
            for c in cuts
        ),
        pose_top_k_equal=sum(r["pose_top_k_equal"] for r in rows),
        rows=rows,
    )
    (a.out / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(
        "SUMMARY",
        json.dumps(
            {k: v for k, v in summary.items() if k not in ["rows", "receipt"]},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
