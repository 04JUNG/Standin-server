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
    cases = json.loads(a.protocol.read_text())["cases"]
    assets = json.loads(a.catalog.read_text())["assets"]
    a.out.mkdir(parents=True, exist_ok=True)
    ids = [c["case_id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_eval_case")
    for c in cases:
        if hashlib.sha256(Path(c["crop"]).read_bytes()).hexdigest() != c["crop_sha256"]:
            raise ValueError("crop_hash_mismatch")

    def evaluate(start):
        from PIL import Image

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
                (f"Q{start+i:03}", Image.open(c["crop"]).convert("RGB"))
                for i, c in enumerate(batch)
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

    if a.remote:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            rows = [r for b in pool.map(evaluate, range(0, len(cases), 4)) for r in b]
    else:
        old = {r["case_id"]: r for r in json.loads(a.recorded.read_text())["rows"]}
        if set(old) != set(ids):
            raise ValueError("recorded_case_mismatch")
        rows = []
        for c in cases:
            prev = old[c["case_id"]]
            if c["crop_sha256"] != prev["crop_sha256"]:
                raise ValueError("recorded_crop_mismatch")
            o = parse_response(
                {
                    "people": [
                        {k: v for k, v in prev["observation"].items() if k != "version"}
                    ]
                },
                [c["case_id"]],
            )[c["case_id"]]
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
        mode="remote" if a.remote else "recorded",
        protocol_sha256=hashlib.sha256(a.protocol.read_bytes()).hexdigest(),
        catalog_sha256=hashlib.sha256(a.catalog.read_bytes()).hexdigest(),
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
