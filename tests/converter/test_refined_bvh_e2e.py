"""Executable Phase 3 contract for inference/refine -> converter handoff.

The real BFF lives in a separate repository.  These tests deliberately model
only its frozen boundary algorithm, then exercise the real inference base-BVH
route and the real internal converter HTTP route around that boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import zipfile

from fastapi.testclient import TestClient
import pytest

import api.app as inference_api
from converter_api.app import create_app
from converter_api.registry import CharacterRegistry
from converter_api.runner import (
    BlenderInfo,
    BlenderRunner,
    ConversionResult,
    RunnerSettings,
)


BASE_BVH = (
    b"HIERARCHY\nROOT Hips\n{\n"
    b"  OFFSET 0 0 0\n  CHANNELS 6 Xposition Yposition Zposition "
    b"Zrotation Xrotation Yrotation\n}\n"
    b"MOTION\nFrames: 1\nFrame Time: 0.033333\n0 0 0 0 0 0\n"
)
REFINED_BVH_TEXT = BASE_BVH.decode("utf-8").replace(
    "0 0 0 0 0 0\n", "0 0 0 35 0 0\n"
)


@dataclass(frozen=True)
class FinalBvh:
    data: bytes
    artifact_kind: str
    sha256: str
    base_bvh_url: str


def _select_final_bvh(
    *,
    inference_client: TestClient,
    base_bvh_url: str,
    refine_response: dict | None,
) -> FinalBvh:
    """Mirror the source-of-truth BFF algorithm from the V3.2 handoff."""
    if refine_response is not None and refine_response.get("refined") is True:
        inline = refine_response.get("bvh")
        assert isinstance(inline, str) and inline, (
            "refined=true requires the inline RefineResponse.bvh artifact"
        )
        data = inline.encode("utf-8")
        artifact_kind = "refined"
    else:
        response = inference_client.get(base_bvh_url)
        assert response.status_code == 200
        data = response.content
        artifact_kind = "base"
    return FinalBvh(
        data=data,
        artifact_kind=artifact_kind,
        sha256=hashlib.sha256(data).hexdigest(),
        base_bvh_url=base_bvh_url,
    )


class LineageRunner:
    def __init__(self):
        self.calls: list[dict] = []

    def inspect_blender(self, *, refresh=False):
        return BlenderInfo(version="5.2.0", build_hash="fbe6228777e7")

    def check_tempdir(self):
        return None

    def convert(self, **kwargs):
        self.calls.append(kwargs)
        source_sha = hashlib.sha256(kwargs["bvh_bytes"]).hexdigest()
        artifact = f"FBX:{source_sha}:mirror={int(kwargs['mirror'])}".encode("ascii")
        report = {
            "src_profile": "mixamo_noprefix",
            "dst_profile": "mixamo",
            "mapped_bones": 22,
            "warnings": [],
            "source_bvh_sha256": source_sha,
            "mirrored": kwargs["mirror"],
        }
        return ConversionResult(
            conversion_id=kwargs["conversion_id"],
            artifact=artifact,
            artifact_sha256=hashlib.sha256(artifact).hexdigest(),
            source_bvh_sha256=source_sha,
            report=report,
        )


def _inference_client(monkeypatch, tmp_path: Path, artifacts: dict[str, bytes]):
    paths: dict[str, str] = {}
    for pose_id, data in artifacts.items():
        path = tmp_path / f"{pose_id}.bvh"
        path.write_bytes(data)
        paths[pose_id] = str(path)
    monkeypatch.setitem(inference_api.STATE, "db_path", "phase3-test-db")
    monkeypatch.setattr(inference_api, "quarantine_record", lambda *_args: None)
    monkeypatch.setattr(
        inference_api,
        "get_bvh_path",
        lambda _db_path, pose_id: paths.get(pose_id),
    )
    return TestClient(inference_api.app)


def _converter_client(tmp_path: Path, runner: LineageRunner):
    character = tmp_path / "character.fbx"
    character.write_bytes(b"character")
    character_sha = hashlib.sha256(character.read_bytes()).hexdigest()
    config = tmp_path / "characters.json"
    config.write_text(
        json.dumps({
            "schema_version": 1,
            "characters": {
                "standin-master-v2": {
                    "display_name": "Standin Master V2",
                    "artifact_uri_env": "PHASE3_CHARACTER_URI",
                    "sha256": character_sha,
                    "rig_profile": "mixamo",
                    "revision": "v2",
                }
            },
        }),
        encoding="utf-8",
    )
    registry = CharacterRegistry(
        config,
        environ={"PHASE3_CHARACTER_URI": str(character)},
    )
    return TestClient(create_app(registry=registry, runner=runner))


def _convert(client: TestClient, final: FinalBvh, *, mirror: bool = False):
    return client.post(
        "/convert",
        files={"bvh": ("final.bvh", final.data, "application/octet-stream")},
        data={"mirror": str(mirror).lower()},
    )


def _convert_bundle(client: TestClient, final: FinalBvh, *, mirror: bool = False):
    return client.post(
        "/convert-bundle",
        files={"bvh": ("final.bvh", final.data, "application/octet-stream")},
        data={
            "artifact_kind": final.artifact_kind,
            "expected_bvh_sha256": final.sha256,
            "mirror": str(mirror).lower(),
        },
    )


def _complete_log(caplog, conversion_id: str) -> dict:
    payloads = []
    for record in caplog.records:
        try:
            payloads.append(json.loads(record.getMessage()))
        except json.JSONDecodeError:
            continue
    return next(
        payload for payload in reversed(payloads)
        if payload.get("event") == "converter_complete"
        and payload.get("conversion_id") == conversion_id
    )


def _assert_lineage(response, final: FinalBvh, caplog) -> None:
    assert response.status_code == 200
    conversion_id = response.headers["x-standin-conversion-id"]
    assert response.headers["x-standin-source-bvh-sha256"] == final.sha256
    assert response.headers["x-standin-artifact-sha256"] == hashlib.sha256(
        response.content
    ).hexdigest()
    assert response.headers["x-standin-solver-version"] == "chain-transport-v3.2.5"
    payload = _complete_log(caplog, conversion_id)
    assert payload["source_bvh_sha256"] == final.sha256
    assert payload["report"]["source_bvh_sha256"] == final.sha256


def test_base_bvh_url_reaches_converter_with_exact_lineage(
    monkeypatch, tmp_path, caplog,
):
    inference = _inference_client(monkeypatch, tmp_path, {"base": BASE_BVH})
    runner = LineageRunner()
    converter = _converter_client(tmp_path, runner)
    final = _select_final_bvh(
        inference_client=inference,
        base_bvh_url="/pose/base/bvh",
        refine_response=None,
    )

    with caplog.at_level(logging.INFO, logger="standin.converter"):
        response = _convert(converter, final)

    assert final.artifact_kind == "base"
    assert runner.calls[0]["bvh_bytes"] == BASE_BVH
    _assert_lineage(response, final, caplog)


def test_base_bundle_preserves_exact_fallback_bvh(monkeypatch, tmp_path):
    inference = _inference_client(monkeypatch, tmp_path, {"base": BASE_BVH})
    runner = LineageRunner()
    converter = _converter_client(tmp_path, runner)
    final = _select_final_bvh(
        inference_client=inference,
        base_bvh_url="/pose/base/bvh",
        refine_response={"refined": False, "bvh": None},
    )

    response = _convert_bundle(converter, final)

    assert response.status_code == 200
    assert response.headers["x-standin-artifact-kind"] == "base"
    with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
        assert bundle.read("final.bvh") == BASE_BVH
        manifest = json.loads(bundle.read("manifest.json"))
    assert manifest["artifact_kind"] == "base"
    assert manifest["artifacts"]["bvh"]["sha256"] == final.sha256


def test_refined_true_uses_inline_bvh_without_fetching_base(
    monkeypatch, tmp_path, caplog,
):
    inference = _inference_client(monkeypatch, tmp_path, {})
    runner = LineageRunner()
    converter = _converter_client(tmp_path, runner)
    final = _select_final_bvh(
        inference_client=inference,
        base_bvh_url="/pose/must-not-be-fetched/bvh",
        refine_response={"refined": True, "bvh": REFINED_BVH_TEXT},
    )

    with caplog.at_level(logging.INFO, logger="standin.converter"):
        response = _convert(converter, final)

    assert final.artifact_kind == "refined"
    assert runner.calls[0]["bvh_bytes"] == REFINED_BVH_TEXT.encode("utf-8")
    _assert_lineage(response, final, caplog)


def test_refined_bundle_preserves_exact_inline_bvh_and_matching_fbx(
    monkeypatch, tmp_path, caplog,
):
    inference = _inference_client(monkeypatch, tmp_path, {})
    runner = LineageRunner()
    converter = _converter_client(tmp_path, runner)
    final = _select_final_bvh(
        inference_client=inference,
        base_bvh_url="/pose/must-not-be-fetched/bvh",
        refine_response={"refined": True, "bvh": REFINED_BVH_TEXT},
    )

    with caplog.at_level(logging.INFO, logger="standin.converter"):
        response = _convert_bundle(converter, final)

    assert response.status_code == 200
    assert response.headers["x-standin-artifact-kind"] == "refined"
    assert response.headers["x-standin-source-bvh-sha256"] == final.sha256
    assert response.headers["x-standin-artifact-sha256"] == hashlib.sha256(
        response.content
    ).hexdigest()
    with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
        bundled_bvh = bundle.read("final.bvh")
        bundled_fbx = bundle.read("final.fbx")
        manifest = json.loads(bundle.read("manifest.json"))

    assert bundled_bvh == REFINED_BVH_TEXT.encode("utf-8")
    assert runner.calls[0]["bvh_bytes"] == bundled_bvh
    assert bundled_fbx == f"FBX:{final.sha256}:mirror=0".encode("ascii")
    assert manifest["artifact_kind"] == "refined"
    assert manifest["artifacts"]["bvh"]["sha256"] == final.sha256
    assert manifest["artifacts"]["fbx"]["sha256"] == hashlib.sha256(
        bundled_fbx
    ).hexdigest()
    assert response.headers["x-standin-fbx-artifact-sha256"] == manifest[
        "artifacts"
    ]["fbx"]["sha256"]


def test_refined_false_falls_back_to_exact_base_bvh(
    monkeypatch, tmp_path, caplog,
):
    inference = _inference_client(monkeypatch, tmp_path, {"base": BASE_BVH})
    runner = LineageRunner()
    converter = _converter_client(tmp_path, runner)
    final = _select_final_bvh(
        inference_client=inference,
        base_bvh_url="/pose/base/bvh",
        refine_response={"refined": False, "bvh": None, "reason": "skeleton_policy"},
    )

    with caplog.at_level(logging.INFO, logger="standin.converter"):
        response = _convert(converter, final)

    assert final.artifact_kind == "base"
    assert runner.calls[0]["bvh_bytes"] == BASE_BVH
    _assert_lineage(response, final, caplog)


def test_refined_true_without_inline_bvh_fails_before_conversion(monkeypatch, tmp_path):
    inference = _inference_client(monkeypatch, tmp_path, {"base": BASE_BVH})
    with pytest.raises(AssertionError, match="inline RefineResponse.bvh"):
        _select_final_bvh(
            inference_client=inference,
            base_bvh_url="/pose/base/bvh",
            refine_response={"refined": True, "bvh": None},
        )


def test_mirror_is_applied_once_by_converter(monkeypatch, tmp_path, caplog):
    inference = _inference_client(monkeypatch, tmp_path, {"base": BASE_BVH})
    runner = LineageRunner()
    converter = _converter_client(tmp_path, runner)
    final = _select_final_bvh(
        inference_client=inference,
        base_bvh_url="/pose/base/bvh",
        refine_response=None,
    )

    with caplog.at_level(logging.INFO, logger="standin.converter"):
        response = _convert(converter, final, mirror=True)

    assert runner.calls[0]["mirror"] is True
    assert response.content.endswith(b"mirror=1")
    _assert_lineage(response, final, caplog)


def test_multiple_people_are_converted_as_independent_items(
    monkeypatch, tmp_path, caplog,
):
    second_base = BASE_BVH.replace(b"0 0 0 0 0 0\n", b"0 0 0 0 10 0\n")
    inference = _inference_client(
        monkeypatch,
        tmp_path,
        {"person-0": BASE_BVH, "person-1": second_base},
    )
    runner = LineageRunner()
    converter = _converter_client(tmp_path, runner)
    finals = [
        _select_final_bvh(
            inference_client=inference,
            base_bvh_url="/pose/person-0/bvh",
            refine_response={"refined": False, "bvh": None},
        ),
        _select_final_bvh(
            inference_client=inference,
            base_bvh_url="/pose/person-1/bvh",
            refine_response={"refined": True, "bvh": REFINED_BVH_TEXT},
        ),
    ]

    with caplog.at_level(logging.INFO, logger="standin.converter"):
        responses = [_convert(converter, final) for final in finals]

    assert [final.artifact_kind for final in finals] == ["base", "refined"]
    assert len(runner.calls) == 2
    assert [call["bvh_bytes"] for call in runner.calls] == [
        BASE_BVH,
        REFINED_BVH_TEXT.encode("utf-8"),
    ]
    assert len({r.headers["x-standin-conversion-id"] for r in responses}) == 2
    for response, final in zip(responses, finals):
        _assert_lineage(response, final, caplog)


@pytest.mark.skipif(
    not (
        os.environ.get("CONVERTER_E2E_CHARACTER_FBX")
        and os.environ.get("CONVERTER_E2E_BASE_BVH")
        and os.environ.get("BLENDER_BINARY")
    ),
    reason="actual Blender Phase 3 assets are not configured",
)
def test_actual_blender_base_refined_mirror_and_multi_person_e2e(
    monkeypatch, tmp_path, caplog,
):
    character = Path(os.environ["CONVERTER_E2E_CHARACTER_FBX"]).resolve(strict=True)
    base_path = Path(os.environ["CONVERTER_E2E_BASE_BVH"]).resolve(strict=True)
    base = base_path.read_bytes()
    lines = base.decode("utf-8").splitlines()
    frame_time = next(
        i for i, line in enumerate(lines) if line.startswith("Frame Time:")
    )
    frame_values = lines[frame_time + 1].split()
    assert len(frame_values) >= 6
    frame_values[3] = "35.000000"
    lines[frame_time + 1] = " ".join(frame_values)
    refined = ("\n".join(lines) + "\n").encode("utf-8")

    inference = _inference_client(monkeypatch, tmp_path, {"base": base})
    character_sha = hashlib.sha256(character.read_bytes()).hexdigest()
    config = tmp_path / "actual-characters.json"
    config.write_text(
        json.dumps({
            "schema_version": 1,
            "characters": {
                "standin-master-v2": {
                    "display_name": "Standin Master V2",
                    "artifact_uri_env": "PHASE3_ACTUAL_CHARACTER_URI",
                    "sha256": character_sha,
                    "rig_profile": "mixamo",
                    "revision": "v2",
                }
            },
        }),
        encoding="utf-8",
    )
    registry = CharacterRegistry(
        config,
        environ={"PHASE3_ACTUAL_CHARACTER_URI": str(character)},
    )
    jobs = tmp_path / "actual-jobs"
    jobs.mkdir()
    runner = BlenderRunner(RunnerSettings(
        blender_binary=os.environ["BLENDER_BINARY"],
        worker_path=Path(__file__).resolve().parents[2] / "converter" / "worker.py",
        temp_root=jobs,
        timeout_seconds=30.0,
    ))
    converter = TestClient(create_app(registry=registry, runner=runner))
    finals = [
        _select_final_bvh(
            inference_client=inference,
            base_bvh_url="/pose/base/bvh",
            refine_response=None,
        ),
        _select_final_bvh(
            inference_client=inference,
            base_bvh_url="/pose/base/bvh",
            refine_response={"refined": True, "bvh": refined.decode("utf-8")},
        ),
    ]

    with caplog.at_level(logging.INFO, logger="standin.converter"):
        base_response = _convert(converter, finals[0])
        refined_response = _convert(converter, finals[1])
        mirror_response = _convert(converter, finals[1], mirror=True)
        second_person_response = _convert(converter, finals[0])

    responses = [
        base_response,
        refined_response,
        mirror_response,
        second_person_response,
    ]
    assert all(response.status_code == 200 for response in responses)
    assert base_response.content != refined_response.content
    assert refined_response.content != mirror_response.content
    assert len({r.headers["x-standin-conversion-id"] for r in responses}) == 4
    assert list(jobs.iterdir()) == []
    _assert_lineage(base_response, finals[0], caplog)
    _assert_lineage(refined_response, finals[1], caplog)
    _assert_lineage(mirror_response, finals[1], caplog)
    _assert_lineage(second_person_response, finals[0], caplog)
    mirror_log = _complete_log(
        caplog,
        mirror_response.headers["x-standin-conversion-id"],
    )
    assert mirror_log["report"]["mirrored"] is True
