#!/usr/bin/env python3
"""Whole-image VLM coarse evaluation; every case records its recommended preset.

Explicit --remote permits Gemini calls. --recorded replays stored observations.
Neither mode changes runtime selection or deploys anything.
"""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, os, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experimental.body_matching.coarse import (
    GeminiCoarseClient,
    VERSION,
    parse_response,
    recommend,
    unknown,
)


def load_cases(protocol, out):
    """Freeze shared live person-crop inputs, including target marker, for audit."""
    from PIL import Image
    from src.experimental.body_matching.coarse_input import INPUT_VERSION, prepare_input

    cases, inputs = [], {}
    for i, raw in enumerate(json.loads(protocol.read_text())["cases"]):
        if "image" in raw and "box" in raw:
            source = Path(raw["image"])
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest != raw["image_sha256"]:
                raise ValueError("source_hash_mismatch")
            with Image.open(source) as opened:
                item = prepare_input(opened, raw["box"])
            dest = out / "inputs" / f"{i:03}"
            dest.mkdir(parents=True, exist_ok=True)
            crop, context = dest / "crop.png", dest / "context.png"
            item.crop.save(crop)
            item.context.save(context)
            c = dict(
                case_id=raw["id"],
                source_image=str(source),
                image_sha256=digest,
                target_box=list(item.target_box),
                crop_box=list(item.crop_box),
                input_version=INPUT_VERSION,
                crop=str(crop.resolve()),
                context=str(context.resolve()),
                crop_sha256=hashlib.sha256(crop.read_bytes()).hexdigest(),
                context_sha256=hashlib.sha256(context.read_bytes()).hexdigest(),
            )
        else:
            c = dict(raw)
            if (
                hashlib.sha256(Path(c["crop"]).read_bytes()).hexdigest()
                != c["crop_sha256"]
            ):
                raise ValueError("crop_hash_mismatch")
            with Image.open(c["crop"]) as opened:
                item = opened.convert("RGB")
        cases.append(c)
        inputs[c["case_id"]] = item
    return cases, inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument(
        "--catalog", type=Path, default=Path("config/body_morphology.v1.json")
    )
    p.add_argument("--out", type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--remote", action="store_true")
    g.add_argument("--recorded", type=Path)
    p.add_argument("--env-file", type=Path)
    p.add_argument("--model", default="gemini-flash-lite-latest")
    a = p.parse_args()
    if a.env_file:
        from dotenv import dotenv_values

        key = dotenv_values(a.env_file).get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY_missing")
        os.environ["GEMINI_API_KEY"] = key
    assets = json.loads(a.catalog.read_text())["assets"]
    a.out.mkdir(parents=True, exist_ok=True)
    cases, inputs = load_cases(a.protocol, a.out)
    (a.out / "inputs.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    ids = [c["case_id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_eval_case")

    def evaluate(start):
        batch = cases[start : start + 4]
        dest = a.out / "provider" / str(start)
        dest.mkdir(parents=True, exist_ok=True)

        class Client(GeminiCoarseClient):
            def record_response(self, r):
                (dest / "response.json").write_text(
                    json.dumps(
                        dict(
                            text=r.text,
                            model_version=r.model_version,
                            usage=r.usage_metadata.model_dump(),
                        ),
                        ensure_ascii=False,
                        indent=2,
                    )
                )

        client = None
        err = None
        try:
            client = Client(a.model)
            crops = [
                (f"Q{start+i:03}", inputs[c["case_id"]]) for i, c in enumerate(batch)
            ]
            response = client.analyze(crops)
            observations = {
                c["case_id"]: {**response[ident], "person_id": c["case_id"]}
                for c, (ident, _) in zip(batch, crops)
            }
        except Exception as e:
            err = type(e).__name__
            observations = {
                c["case_id"]: unknown(c["case_id"], "VLM 오류: " + err) for c in batch
            }
        finally:
            if client:
                client.close()
        rows = [
            dict(
                case_id=c["case_id"],
                crop=c["crop"],
                crop_sha256=c["crop_sha256"],
                input_metadata={
                    k: c[k]
                    for k in (
                        "source_image",
                        "image_sha256",
                        "target_box",
                        "crop_box",
                        "context",
                        "context_sha256",
                        "input_version",
                    )
                    if k in c
                },
                observation=observations[c["case_id"]],
                recommendation=recommend(observations[c["case_id"]], assets),
                provider_error=err,
            )
            for c in batch
        ]
        print(
            "batch",
            start,
            "recommendations",
            len(rows),
            "provider_error",
            err,
            flush=True,
        )
        return rows

    recorded = None
    if a.remote:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            rows = [r for b in pool.map(evaluate, range(0, len(cases), 4)) for r in b]
    else:
        recorded = json.loads(a.recorded.read_text())
        old = {r["case_id"]: r for r in recorded["rows"]}
        if set(old) != set(ids):
            raise ValueError("recorded_case_mismatch")
        rows = []
        for c in cases:
            prev = old[c["case_id"]]
            if c["crop_sha256"] != prev["crop_sha256"]:
                raise ValueError("recorded_crop_mismatch")
            if c.get("input_version") or prev.get("input_metadata", {}).get(
                "input_version"
            ):
                for field in (
                    "context_sha256",
                    "image_sha256",
                    "target_box",
                    "input_version",
                ):
                    if c.get(field) != prev.get("input_metadata", {}).get(field):
                        raise ValueError("recorded_context_mismatch")
            o = parse_response(
                {
                    "people": [
                        {k: v for k, v in prev["observation"].items() if k != "version"}
                    ]
                },
                [c["case_id"]],
            )[c["case_id"]]
            o["version"] = prev["observation"]["version"]
            rows.append(
                {**prev, "observation": o, "recommendation": recommend(o, assets)}
            )
    from collections import Counter

    metrics = dict(
        total=len(rows),
        recommendations=sum(
            bool(r["recommendation"]["recommended_body_id"]) for r in rows
        ),
        family_counts=dict(
            Counter(r["recommendation"]["observed_family"] or "unknown" for r in rows)
        ),
        preset_counts=dict(
            Counter(
                r["recommendation"]["recommended_family"]
                + "/"
                + r["recommendation"]["recommended_preset"]
                for r in rows
            )
        ),
        basis_counts=dict(
            Counter(r["recommendation"]["recommendation_basis"] for r in rows)
        ),
        provider_errors=sum(bool(r["provider_error"]) for r in rows),
        auto_applied=0,
        accuracy=None,
        accuracy_note="작가의 정답 라벨 없음. 추천 출력률은 정확도가 아님.",
    )
    result = dict(
        version=VERSION,
        classification_versions=sorted({r["observation"]["version"] for r in rows}),
        mode="remote" if a.remote else "recorded",
        model_requested=a.model if a.remote else recorded.get("model_requested"),
        recorded_from=str(a.recorded.resolve()) if a.recorded else None,
        source_classification_implementation_sha256=(
            (
                recorded.get("source_classification_implementation_sha256")
                or recorded.get("implementation_sha256")
            )
            if recorded
            else None
        ),
        protocol_sha256=hashlib.sha256(a.protocol.read_bytes()).hexdigest(),
        catalog_sha256=hashlib.sha256(a.catalog.read_bytes()).hexdigest(),
        input_implementation_sha256=hashlib.sha256(
            (
                Path(__file__).resolve().parents[1]
                / "src/experimental/body_matching/coarse_input.py"
            ).read_bytes()
        ).hexdigest(),
        implementation_sha256=hashlib.sha256(
            (
                Path(__file__).resolve().parents[1]
                / "src/experimental/body_matching/coarse.py"
            ).read_bytes()
        ).hexdigest(),
        metrics=metrics,
        rows=rows,
    )
    (a.out / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
