"""Face fitting shares the FBX rotation contract and never silently auto-approves."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pose_curation.head.fitting import fit_manual, angle_distance
from pose_curation.orientation import Orientation
from pose_curation.scoped.matching import projection
from pose_curation.review.head_routes import head_router
from pose_curation.review.framing import FramedPreviews, NativeReference
from pose_curation.storage import sha256, write_json

# Six vertex facts from the SHA-pinned MediaPipe canonical face (Apache-2.0).
FACE = np.array(
    [
        [-4.445859, 2.663991, 3.173422],
        [4.445859, 2.663991, 3.173422],
        [0, 2.473255, 5.788627],
        [0, -0.463170, 7.586580],
        [0, -9.403378, 4.264492],
        [0, 8.261778, 4.481535],
    ]
)


@pytest.mark.parametrize(
    "angles", [(0, 0, 0), (30, 20, -10), (-35, -15, 12), (58, 40, 32)]
)
def test_manual_landmarks_recover_export_orientation(angles):
    orientation = Orientation(*angles)
    points = projection(FACE, orientation) * 12 + [230, 250]
    result = fit_manual(points, [512, 512], FACE)
    np.testing.assert_allclose(
        [result["orientation"][k] for k in ("yaw", "pitch", "roll")], angles, atol=0.02
    )
    np.testing.assert_allclose(result["projected"], points, atol=0.02)
    assert result["source"] == "user_landmarks" and result["requires_visual_review"]
    assert result["confidence"] == "unvalidated"


@pytest.mark.parametrize(
    "mutation", ["nan", "outside", "missing", "collapsed", "swapped", "side"]
)
def test_manual_fit_rejects_invalid_or_unobservable_face(mutation):
    points = projection(FACE, Orientation()) * 12 + [230, 250]
    if mutation == "nan":
        points[0, 0] = np.nan
    if mutation == "outside":
        points[0, 0] = -1
    if mutation == "missing":
        points = points[:-1]
    if mutation == "collapsed":
        points[:] = 100
    if mutation == "swapped":
        points[[0, 1]] = points[[1, 0]]
    if mutation == "side":
        points = projection(FACE, Orientation(80, 0, 0)) * 12 + [230, 250]
    with pytest.raises(ValueError):
        fit_manual(points, [512, 512], FACE)


def test_flip_rotation_check_understands_three_axes():
    assert (
        angle_distance(
            Orientation(35, 20, -12).public(), Orientation(-35, 20, 12).public()
        )
        < 1e-5
    )
    assert (
        angle_distance(
            Orientation(35, 20, -12).public(), Orientation(35, 20, -12).public()
        )
        > 50
    )


def test_neutral_reference_is_distinct_from_bvh_cache(tmp_path):
    service = FramedPreviews(tmp_path)
    service._identity = {"test": True}
    reference = NativeReference(tmp_path / "character.fbx", "a" * 64)
    pose = SimpleNamespace(content_hash="a" * 64, metadata={}, group="existing")
    assert service.identity(reference, "head") != service.identity(pose, "head")
    assert service.identity(reference, "head") != service.identity(reference, "bust")
    with pytest.raises(ValueError):
        service.identity(reference, "half")
    service.close()


def client_fixture(tmp_path, monkeypatch):
    import pose_curation.review.head_routes as routes

    coverage = tmp_path / "coverage/20261001"
    coverage.mkdir(parents=True)
    source = coverage / "image.png"
    source.write_bytes(b"image")
    content_hash = sha256(source)
    write_json(
        coverage / "inputs.json",
        [
            {
                "id": "rough_test",
                "size": [512, 512],
                "sha256": content_hash,
                "origin": "supplied",
            }
        ],
    )
    write_json(coverage / "report.json", {"files": {"rough_test": "image.png"}})
    service = FramedPreviews(tmp_path)
    service.character.parent.mkdir(parents=True)
    service.character.write_bytes(b"character")
    monkeypatch.setattr(routes, "canonical", lambda *args: FACE)
    monkeypatch.setattr(
        service, "status", lambda *args, **kwargs: {"status": "rendering"}
    )
    app = FastAPI()
    app.include_router(
        head_router(
            tmp_path, Path(__file__).parents[1] / "pose_curation/review/static", service
        )
    )
    return TestClient(app), service, source, content_hash


def test_head_api_contract_observations_and_source_binding(tmp_path, monkeypatch):
    client, service, source, content_hash = client_fixture(tmp_path, monkeypatch)
    try:
        config = client.get("/api/head").json()
        assert config["automatic_enabled"] is False
        assert len(config["items"]) == 1
        assert "path" not in config["items"][0]
        points = (projection(FACE, Orientation(30, 20, -10)) * 12 + [230, 250]).tolist()
        spec = {"query": "rough_test", "content_hash": content_hash, "points": points}
        assert client.post("/api/head/fit", json=spec).status_code == 403
        headers = {"X-Pose-Review": "1"}
        assert (
            client.post(
                "/api/head/fit",
                json=spec,
                headers={**headers, "Origin": "https://example.com"},
            ).status_code
            == 403
        )
        result = client.post("/api/head/fit", json=spec, headers=headers)
        assert result.status_code == 200
        assert result.json()["orientation"]["yaw"] == 30
        assert (
            client.post(
                "/api/head/fit", json={**spec, "points": points[:5]}, headers=headers
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/head/fit",
                json={**spec, "content_hash": "0" * 64},
                headers=headers,
            ).status_code
            == 409
        )
        source.write_bytes(b"changed")
        assert (
            client.post("/api/head/fit", json=spec, headers=headers).status_code == 409
        )
        assert client.get("/api/head/queries/rough_test/image").status_code == 404
        assert client.get("/api/head/queries/unknown/image").status_code == 404
    finally:
        service.close()


def test_reference_export_only_head_bust_and_fbx(tmp_path, monkeypatch):
    client, service, _, _ = client_fixture(tmp_path, monkeypatch)
    try:
        spec = {
            "content_hash": sha256(service.character),
            "scope": "head",
            "yaw": 35,
            "pitch": 20,
            "roll": -12,
        }
        headers = {"X-Pose-Review": "1"}
        assert (
            client.post(
                "/api/head/reference/oriented", json=spec, headers=headers
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/head/reference/oriented",
                json={**spec, "scope": "bust"},
                headers=headers,
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/head/reference/oriented",
                json={**spec, "scope": "half"},
                headers=headers,
            ).status_code
            == 422
        )
        assert (
            client.get(
                "/api/head/reference/oriented/bvh", params={**spec, "v": "x"}
            ).status_code
            == 422
        )
        service.character.write_bytes(b"new revision")
        assert (
            client.post(
                "/api/head/reference/oriented", json=spec, headers=headers
            ).status_code
            == 409
        )
    finally:
        service.close()


def test_headless_exclusion_is_persistent_reversible_and_hash_bound(
    tmp_path, monkeypatch
):
    from pose_curation.head.queries import HeadQueries

    client, service, source, content_hash = client_fixture(tmp_path, monkeypatch)
    try:
        spec = {
            "content_hash": content_hash,
            "excluded": True,
            "reason": "no head in this image",
        }
        endpoint = "/api/head/queries/rough_test/review"
        assert client.post(endpoint, json=spec).status_code == 403
        assert (
            client.post(
                endpoint,
                json=spec,
                headers={"X-Pose-Review": "1", "Origin": "https://example.com"},
            ).status_code
            == 403
        )
        assert (
            client.post(endpoint, json=spec, headers={"X-Pose-Review": "1"}).status_code
            == 200
        )
        assert HeadQueries(tmp_path).list()[0]["excluded"] is True
        with pytest.raises(ValueError):
            HeadQueries(tmp_path).get("rough_test")
        assert sha256(source) == content_hash
        assert client.get("/api/head/queries/rough_test/image").status_code == 200
        assert (
            client.post(
                endpoint,
                json={**spec, "content_hash": "0" * 64},
                headers={"X-Pose-Review": "1"},
            ).status_code
            == 409
        )
        assert (
            client.post(
                endpoint,
                json={**spec, "excluded": False},
                headers={"X-Pose-Review": "1"},
            ).status_code
            == 200
        )
        assert HeadQueries(tmp_path).get("rough_test")["content_hash"] == content_hash
    finally:
        service.close()


def test_same_face_merge_holds_conflicting_angles_and_reference_photos():
    from pose_curation.head.candidates import candidate_groups

    first = {
        "bbox": [100, 100, 200, 240],
        "status": "suggested",
        "reasons": [],
        "error": 0.05,
        "source": "body",
        "orientation": Orientation(10, 5, 0).public(),
    }
    second = {
        **first,
        "source": "anime",
        "error": 0.06,
        "orientation": Orientation(12, 6, 1).public(),
    }
    good = candidate_groups([first, second])
    assert (
        len(good) == 1
        and good[0]["status"] == "suggested"
        and good[0]["observation_count"] == 2
    )
    bad = candidate_groups(
        [first, {**second, "orientation": Orientation(-30, 5, 0).public()}]
    )[0]
    assert bad["status"] == "manual" and "orientation" not in bad
    photo = candidate_groups([first, {**second, "excluded_reference": True}])[0]
    assert photo["status"] == "manual" and "orientation" not in photo
    assert len(candidate_groups([first, {**second, "bbox": [400, 100, 500, 240]}])) == 2


def candidate_fixture(tmp_path, content_hash):
    from pose_curation.head.candidates import VERSION, code_revision

    root = tmp_path / "head-direction"
    raw = root / "crop-observations/rough_test.json"
    write_json(raw, {"test_observations": True})
    candidate = {
        "id": "a" * 24,
        "status": "suggested",
        "orientation": Orientation(30, 20, -10).public(),
        "bbox": [120, 120, 300, 350],
        "points": [[150, 200]] * 12,
        "projected": [[150, 200]] * 12,
        "error": 0.05,
        "sources": ["private-source-file"],
        "source": "private-source-file",
        "reasons": [],
    }
    row = {
        "image_sha256": content_hash,
        "size": [512, 512],
        "observations": {"crop-observations/rough_test.json": sha256(raw)},
        "candidates": [candidate, {**candidate, "id": "b" * 24, "status": "manual"}],
    }
    write_json(
        root / "candidates.json",
        {
            "version": VERSION,
            "code_revision": code_revision(),
            "source_review_sha256": None,
            "images": {"rough_test": row},
        },
    )
    return raw


def test_candidate_requires_explicit_target_and_current_image_observations(
    tmp_path, monkeypatch
):
    client, service, source, content_hash = client_fixture(tmp_path, monkeypatch)
    raw = candidate_fixture(tmp_path, content_hash)
    headers = {"X-Pose-Review": "1"}
    body = {
        "query": "rough_test",
        "content_hash": content_hash,
        "candidate": "a" * 24,
        "target_confirmed": True,
    }
    try:
        assert client.get("/api/head").json()["automatic_enabled"] is True
        result = client.get("/api/head/queries/rough_test/candidates")
        assert result.status_code == 200 and "private-source-file" not in result.text
        assert client.post("/api/head/suggest", json=body).status_code == 403
        assert (
            client.post(
                "/api/head/suggest",
                json=body,
                headers={**headers, "Origin": "https://elsewhere.test"},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/head/suggest",
                json={**body, "target_confirmed": False},
                headers=headers,
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/head/suggest",
                json={**body, "candidate": "b" * 24},
                headers=headers,
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/api/head/suggest",
                json={**body, "content_hash": "0" * 64},
                headers=headers,
            ).status_code
            == 409
        )
        result = client.post("/api/head/suggest", json=body, headers=headers)
        assert (
            result.status_code == 200
            and result.json()["source"] == "detected_face_user_selected"
        )
        assert result.json()["requires_visual_review"]
        client.post(
            "/api/head/queries/rough_test/review",
            json={"content_hash": content_hash, "excluded": True},
            headers=headers,
        )
        assert (
            client.post("/api/head/suggest", json=body, headers=headers).status_code
            == 409
        )
        client.post(
            "/api/head/queries/rough_test/review",
            json={"content_hash": content_hash, "excluded": False},
            headers=headers,
        )
        raw.write_text("{}")
        assert (
            client.post("/api/head/suggest", json=body, headers=headers).status_code
            == 409
        )
    finally:
        service.close()


def test_candidate_code_and_review_updates_invalidate_manifest(tmp_path, monkeypatch):
    from pose_curation.head import candidates
    from pose_curation.head.queries import HeadQueries

    client, service, _, digest = client_fixture(tmp_path, monkeypatch)
    candidate_fixture(tmp_path, digest)
    store = candidates.HeadCandidates(tmp_path, HeadQueries(tmp_path))
    try:
        assert store.get("rough_test")["candidates"]
        write_json(
            tmp_path / "head-direction/source-face-reviews.json", {"changed": True}
        )
        with pytest.raises(ValueError):
            store.get("rough_test")
        assert store.counts() == {}
        monkeypatch.setattr(candidates, "code_revision", lambda: "different-code")
        with pytest.raises(ValueError):
            store.get("rough_test")
    finally:
        service.close()


def test_body_head_regions_are_proposals_not_substitute_face_landmarks():
    from pose_curation.head.regions import body_head_regions

    person = {
        "person": 7,
        "keypoints": [[150, 130], [135, 115], [165, 115], [120, 120], [180, 120]]
        + [[0, 0]] * 12,
        "scores": [0.8] * 5 + [0.1] * 12,
    }
    regions = body_head_regions([person], [300, 300])
    assert len(regions) == 1 and regions[0]["proposal_person"] == 7
    assert "points" not in regions[0]
    assert body_head_regions([{**person, "scores": [0.1] * 17}], [300, 300]) == []
