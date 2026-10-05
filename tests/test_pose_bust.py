"""Image observations, neck limits and exact articulated export/review contracts."""

import numpy as np
import pytest
from pose_curation.head.bust import relative_rotation, shoulder_roll, validate_region
from pose_curation.head.recovery_experiment import original_points
from pose_curation.orientation import Orientation
from pose_curation.review.framing import NativeReference
from pose_curation.review.head_exports import with_body
from pose_curation.storage import sha256
from tests.test_pose_head import client_fixture


@pytest.mark.parametrize(
    "face,body",
    [
        ((30, 10, -8), (-5, 0, 12)),
        ((-30, 10, 8), (5, 0, -12)),
        ((180, 0, 0), (150, 0, 0)),
        ((-179, 0, 0), (179, 0, 0)),
    ],
)
def test_relative_neck_composes_to_requested_face(face, body):
    a, b = Orientation(*face), Orientation(*body)
    relative, angle = relative_rotation(a, b)
    np.testing.assert_allclose(b.matrix() @ relative, a.matrix(), atol=1e-12)
    assert angle <= 55


def test_neck_gate_and_region_bounds():
    assert relative_rotation(Orientation(55, 0, 0), Orientation())[1] == pytest.approx(
        55
    )
    with pytest.raises(ValueError, match="55"):
        relative_rotation(Orientation(56, 0, 0), Orientation())
    assert validate_region([0, 0, 16, 16], [64, 64]) == [0, 0, 16, 16]
    for region in (
        [0, 0, 15, 16],
        [-1, 0, 50, 50],
        [0, 0, 65, 64],
        [0, 0, np.nan, 40],
        [20, 20, 0, 0],
    ):
        with pytest.raises(ValueError):
            validate_region(region, [64, 64])


@pytest.mark.parametrize(
    "yaw,pitch,target", [(0, 0, 20), (30, 15, -12), (170, -10, 25)]
)
def test_shoulders_match_roll_without_inventing_depth(yaw, pitch, target):
    expected = Orientation(yaw, pitch, target)
    vector = (expected.matrix() @ [1.0, 0, 0])[:2] * [1, -1]
    if vector[0] < 0:
        vector *= -1
    points = np.array([250, 250]) + np.array([-100, 100])[:, None] * vector
    result = shoulder_roll(points, [512, 512], Orientation(yaw, pitch, 0))
    assert result.yaw == yaw and result.pitch == pitch
    assert result.roll == pytest.approx(target, abs=0.01)


def test_shoulders_refuse_unobservable_or_reversed_points():
    for points, body in [
        ([[200, 200], [100, 200]], Orientation()),
        ([[100, 100], [102, 102]], Orientation()),
        ([[100, 100], [300, 300]], Orientation(90, 0, 0)),
        ([[0, 0], [600, 300]], Orientation()),
    ]:
        with pytest.raises(ValueError):
            shoulder_roll(points, [512, 512], body)


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
@pytest.mark.parametrize("mirrored", [False, True])
def test_recovery_inverse_pixel_transforms_non_square(turns, mirrored):
    w, h = 300, 180
    original = np.array([[40.0, 50.0], [200, 130.0]])
    x, y = original.T
    rotated = [
        original,
        np.stack([y, w - 1 - x], axis=1),
        np.stack([w - 1 - x, h - 1 - y], axis=1),
        np.stack([h - 1 - y, x], axis=1),
    ][turns].copy()
    size = np.array([h, w] if turns % 2 else [w, h])
    if mirrored:
        rotated[:, 0] = size[0] - 1 - rotated[:, 0]
    expected = original.copy()
    if mirrored:
        expected[:, 0] = w - 1 - expected[:, 0]
    np.testing.assert_allclose(
        original_points(rotated / size, [w, h], turns, mirrored), expected, atol=1e-10
    )


def test_reference_export_validation_and_cache_partition(tmp_path, monkeypatch):
    client, previews, _, digest = client_fixture(tmp_path, monkeypatch)
    previews._identity = {"test": True}
    ref = NativeReference(previews.character, sha256(previews.character))
    face = Orientation(30, 0, 0)
    first = with_body(ref, face, Orientation())
    second = with_body(ref, face, Orientation(10, 0, 0))
    assert previews.identity(first, "bust") == previews.identity(ref, "bust")
    assert previews.identity(first, "bust", face) != previews.identity(
        second, "bust", face
    )
    assert previews.identity(first, "bust", face) != previews.identity(
        ref, "bust", face
    )
    spec = dict(
        scope="bust",
        content_hash=ref.content_hash,
        yaw=30,
        pitch=0,
        roll=0,
        body_yaw=0,
        body_pitch=0,
        body_roll=0,
    )
    calls = []
    monkeypatch.setattr(
        previews,
        "status",
        lambda pose, *a, **kw: calls.append(pose.metadata) or {"status": "missing"},
    )
    endpoint = "/api/head/reference/oriented"
    assert (
        client.post(endpoint, json=spec, headers={"X-Pose-Review": "1"}).status_code
        == 200
    )
    assert calls[-1]["bust_body"] == dict(yaw=0, pitch=0, roll=0)
    assert client.get(endpoint, params=spec).status_code == 200
    for mutation in (
        {"body_pitch": None},
        {"scope": "head"},
        {"yaw": 100},
        {"body_yaw": 181},
    ):
        bad = {**spec, **mutation}
        assert (
            client.post(endpoint, json=bad, headers={"X-Pose-Review": "1"}).status_code
            == 422
        )
        bad = {k: v for k, v in bad.items() if v is not None}
        assert client.get(endpoint, params=bad).status_code == 422
    previews.close()


def test_manual_back_head_review_and_body_preview_binding(tmp_path, monkeypatch):
    client, previews, _, digest = client_fixture(tmp_path, monkeypatch)
    ref = sha256(previews.character)
    endpoint = "/api/head/queries/rough_test/angle-reviews"
    headers = {"X-Pose-Review": "1"}
    body = dict(yaw=150, pitch=0, roll=0)
    spec = dict(
        content_hash=digest,
        reference_hash=ref,
        scope="bust",
        region=[50, 50, 250, 300],
        angles=dict(yaw=180, pitch=0, roll=0),
        body_angles=body,
        status="accepted",
        preview_version="f" * 64,
        visual_confirmed=True,
    )

    def status(pose, *a, **kw):
        return {
            "status": "ready",
            "version": "f" * 64 if pose.metadata.get("bust_body") == body else "e" * 64,
        }

    monkeypatch.setattr(previews, "status", status)
    assert (
        client.post(
            endpoint,
            json={**spec, "body_angles": dict(yaw=140, pitch=0, roll=0)},
            headers=headers,
        ).status_code
        == 409
    )
    saved = client.post(endpoint, json=spec, headers=headers)
    assert saved.status_code == 200
    row = saved.json()
    assert row["target"].startswith("region:") and row["body_angles"] == body
    assert row["provenance"]["original_orientation"] is None
    assert client.get(endpoint).json()["items"][0]["current"]
    assert (
        client.post(
            endpoint, json={**spec, "scope": "head"}, headers=headers
        ).status_code
        == 422
    )
    assert (
        client.post(
            endpoint, json={**spec, "points": [[50, 50]] * 6}, headers=headers
        ).status_code
        == 422
    )
    assert (
        client.post(
            endpoint, json={**spec, "region": [0, 0, 10, 10]}, headers=headers
        ).status_code
        == 409
    )
    previews.close()


def test_shoulder_route_binds_current_local_image(tmp_path, monkeypatch):
    client, previews, _, digest = client_fixture(tmp_path, monkeypatch)
    endpoint = "/api/head/shoulder-roll"
    headers = {"X-Pose-Review": "1"}
    spec = dict(
        query="rough_test",
        content_hash=digest,
        points=[[100, 100], [300, 200]],
        body_angles=dict(yaw=20, pitch=5, roll=0),
    )
    assert client.post(endpoint, json=spec).status_code == 403
    assert (
        client.post(
            endpoint, json=spec, headers={**headers, "Origin": "https://other.test"}
        ).status_code
        == 403
    )
    result = client.post(endpoint, json=spec, headers=headers)
    assert result.status_code == 200 and result.json()["body_angles"]["yaw"] == 20
    assert result.json()["body_angles"]["pitch"] == 5
    review = dict(
        content_hash=digest,
        reference_hash=sha256(previews.character),
        scope="bust",
        region=[50, 50, 250, 300],
        angles=dict(yaw=20, pitch=5, roll=0),
        body_angles=result.json()["body_angles"],
        shoulder_points=spec["points"],
        status="hold",
    )
    reviews = "/api/head/queries/rough_test/angle-reviews"
    saved = client.post(reviews, json=review, headers=headers)
    assert (
        saved.status_code == 200 and saved.json()["shoulder_points"] == spec["points"]
    )
    changed = {**review, "body_angles": dict(yaw=20, pitch=5, roll=0)}
    assert client.post(reviews, json=changed, headers=headers).status_code == 409
    assert (
        client.post(
            endpoint, json={**spec, "content_hash": "0" * 64}, headers=headers
        ).status_code
        == 409
    )
    assert (
        client.post(
            endpoint, json={**spec, "points": [[0, 0], [512, 0]]}, headers=headers
        ).status_code
        == 409
    )
    previews.close()
