#!/usr/bin/env python3
"""Replay recorded first/body-review responses through the current body policy.

No model/network call. Existing people, boxes, masks and selected poses remain
byte-for-byte equal. Source API timings are retained as SOURCE measurements,
never presented as new end-to-end latency of the policy replay.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Exercise deferred review and slot-based join",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from PIL import Image
    from src.vlm.client import _coerce
    from src.schema import BBox
    from src.experimental.body_matching.integrated import VERSION, detect_body
    from src.experimental.body_matching.integrated_review import (
        review_uncertain,
        start_pipeline_review,
        finish_pipeline_review,
        abandon_pipeline_review,
    )
    from src.experimental.body_matching.coarse_service import CoarseBodyService
    from api.models import CutResultOut
    from api.body_models import validated_body_sidecar

    protocol = json.loads(args.protocol.read_text())["cases"]
    images = {c["cut_id"]: c for c in protocol}
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((args.source_run / "receipt.json").read_text())
    receipt.update(
        integrated_version=VERSION,
        source_run=str(args.source_run.resolve()),
        body_review_mode="parallel" if args.parallel else "serial",
        execution_scope="Recorded response body-policy replay only; no network or pose execution",
        timing_scope="Copied source VLM/API measurements; NOT newly measured end-to-end latency",
    )
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2)
    )
    rows = []
    for cut, case in images.items():
        source = args.source_run / "integrated-0" / cut / "result.json"
        old = json.loads(source.read_text())
        if old["error"] or old["result"] is None:
            raise ValueError("source_unavailable_" + cut)
        image_path = Path(case["image"])
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != case["image_sha256"]:
            raise ValueError("source_image_hash_mismatch")
        image = Image.open(image_path).convert("RGB")
        analysis = _coerce(old["primary"], *image.size)
        controls = asdict(analysis)
        controls.pop("person_bodies")
        recorded = [c for c in old["calls"] if c["kind"] == "body_review"]
        calls = []

        def generate(**kw):
            if len(calls) >= len(recorded):
                raise AssertionError("unexpected_new_review_call")
            call = recorded[len(calls)]
            calls.append(call)
            if call.get("error"):
                raise RuntimeError("recorded_review_failure")
            if kw["model"] != call["model"]:
                raise AssertionError("review_model_changed")
            return SimpleNamespace(text=call["text"])

        client = SimpleNamespace(
            _model=receipt["model"],
            _clock=time.perf_counter,
            _client=SimpleNamespace(models=SimpleNamespace(generate_content=generate)),
        )
        started = time.perf_counter()
        run_review = start_pipeline_review if args.parallel else review_uncertain
        run_review(
            client,
            image,
            analysis,
            SimpleNamespace(
                body_max_people=8,
                body_timeout_seconds=12,
                vlm_body_thinking_level=receipt["thinking_level"],
            ),
            started + 30,
        )
        now = asdict(analysis)
        now.pop("person_bodies")
        assert controls == now
        descriptors = []
        for person in old["result"]["people"]:
            slot = person["quality_trace"].get("vlm_person_index")
            box = BBox(*person["box"], source="recorded") if person.get("box") else None
            descriptors.append(
                SimpleNamespace(
                    box=box,
                    body_classification=detect_body(analysis, slot),
                    quality_trace=copy.deepcopy(person["quality_trace"]),
                )
            )
        service = CoarseBodyService(
            Path(__file__).resolve().parents[1] / "config/body_morphology.v1.json",
            integrated=True,
            provider="gemini",
            model=receipt["model"],
            family_review_model=receipt["model"],
        )
        replay_result = SimpleNamespace(
            route=old["result"]["route"],
            descriptors=descriptors,
            person_candidates=[[] for _ in descriptors],
        )
        if args.parallel:
            try:
                finish_pipeline_review(analysis, replay_result, started)
            finally:
                abandon_pipeline_review(analysis)
        assert len(calls) == len(recorded), "review_eligibility_changed"
        sidecar = service.analyze(
            image, replay_result, mode=old["result"]["body_matching"]["mode"]
        )
        if hasattr(replay_result, "_body_review_execution"):
            sidecar["parallel_review"] = replay_result._body_review_execution
        typed = CutResultOut.model_validate(old["result"])
        wire = validated_body_sidecar(sidecar, typed.people)
        assert wire is not None and wire.status != "unavailable"
        new = copy.deepcopy(old)
        new["result"]["body_matching"] = wire.model_dump(mode="json")
        new["result"]["inference_metadata"]["vlm_body_prompt_version"] = VERSION
        assert new["result"]["people"] == old["result"]["people"]
        for a, b in zip(
            old["result"]["body_matching"]["people"],
            new["result"]["body_matching"]["people"],
        ):
            assert a["person_id"] == b["person_id"]
            assert b["auto_body_id"] is None and b["applied_body_id"] is None
            before = a["coarse_recommendation"]["recommended_body_id"]
            after = b["coarse_recommendation"]["recommended_body_id"]
            rows.append(
                dict(
                    cut_id=cut,
                    index=b["person_index"],
                    before=before,
                    after=after,
                    changed=before != after,
                )
            )
        new.update(
            execution_scope=receipt["execution_scope"],
            timing_scope=receipt["timing_scope"],
            policy_replay_elapsed_ms=(time.perf_counter() - started) * 1000,
            source_record_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        )
        dest = args.out / "integrated-0" / cut / "result.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(new, ensure_ascii=False, indent=2))
        old_baseline = args.source_run / "baseline-0" / cut / "result.json"
        dest = args.out / "baseline-0" / cut / "result.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_baseline, dest)
    summary = dict(
        scope=receipt["execution_scope"],
        cuts=len(images),
        people=len(rows),
        controls_and_pose_unchanged=True,
        changed=[r for r in rows if r["changed"]],
        rows=rows,
    )
    (args.out / "POLICY_REPLAY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
