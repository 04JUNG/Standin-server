"""Face review persistence requires current observations and an inspected export."""

import sqlite3

import pytest

from pose_curation.head.reviews import HeadReviewStore
from pose_curation.scoped.matching import projection
from pose_curation.orientation import Orientation
from pose_curation.storage import sha256
from tests.test_pose_head import FACE, candidate_fixture, client_fixture


@pytest.fixture
def review_env(tmp_path, monkeypatch):
    client, previews, image, digest = client_fixture(tmp_path, monkeypatch)
    raw = candidate_fixture(tmp_path, digest)
    spec = {
        "content_hash": digest,
        "reference_hash": sha256(previews.character),
        "scope": "head",
        "candidate": "a" * 24,
        "angles": {"yaw": 31.5, "pitch": 18, "roll": -9},
        "status": "hold",
        "note": "look slightly further left",
    }
    yield client, previews, image, raw, spec
    previews.close()


ENDPOINT = "/api/head/queries/rough_test/angle-reviews"
HEADERS = {"X-Pose-Review": "1"}


def test_review_history_scope_separation_and_concurrent_write(review_env, tmp_path):
    client, _, _, _, spec = review_env
    first = client.post(ENDPOINT, json=spec, headers=HEADERS)
    assert first.status_code == 200
    first = first.json()
    assert first["provenance"]["original_orientation"]["yaw"] == 30
    assert first["angles"]["yaw"] == 31.5
    assert client.post(ENDPOINT, json=spec, headers=HEADERS).status_code == 409
    second = client.post(
        ENDPOINT,
        json={**spec, "expected_revision": first["revision"], "status": "rejected"},
        headers=HEADERS,
    )
    assert second.status_code == 200
    assert (
        client.post(
            ENDPOINT, json={**spec, "scope": "bust"}, headers=HEADERS
        ).status_code
        == 200
    )
    rows = client.get(ENDPOINT).json()["items"]
    assert len(rows) == 2 and all(row["current"] for row in rows)
    path = tmp_path / "head-direction/angle-reviews.sqlite"
    assert len(HeadReviewStore(path).list(spec["content_hash"])) == 2
    with sqlite3.connect(path) as con:
        assert con.execute("SELECT count(*) FROM head_angle_events").fetchone()[0] == 3


def test_accept_requires_confirmation_exact_render_and_current_reference(
    review_env, monkeypatch
):
    client, previews, _, _, spec = review_env
    accepted = {
        **spec,
        "status": "accepted",
        "preview_version": "f" * 64,
        "visual_confirmed": True,
    }
    assert client.post(ENDPOINT, json=accepted, headers=HEADERS).status_code == 409
    checked = []

    def ready(pose, scope, orientation):
        checked.append((scope, orientation))
        return {"status": "ready", "version": "f" * 64}

    monkeypatch.setattr(previews, "status", ready)
    for mutation in (
        {"visual_confirmed": False},
        {"preview_version": "e" * 64},
        {"reference_hash": "0" * 64},
    ):
        assert (
            client.post(
                ENDPOINT, json={**accepted, **mutation}, headers=HEADERS
            ).status_code
            == 409
        )
    assert client.post(ENDPOINT, json=accepted, headers=HEADERS).status_code == 200
    assert checked[-1] == ("head", Orientation(31.5, 18, -9))
    assert client.get(ENDPOINT).json()["items"][0]["current"]
    monkeypatch.setattr(
        previews, "status", lambda *a, **kw: {"status": "ready", "version": "e" * 64}
    )
    old = client.get(ENDPOINT).json()["items"][0]
    assert not old["current"] and old["status"] == "accepted"


def test_review_rejects_cross_origin_missing_target_and_stale_sources(review_env):
    client, _, image, raw, spec = review_env
    assert client.post(ENDPOINT, json=spec).status_code == 403
    assert (
        client.post(
            ENDPOINT, json=spec, headers={**HEADERS, "Origin": "https://other.test"}
        ).status_code
        == 403
    )
    no_target = {k: v for k, v in spec.items() if k != "candidate"}
    assert client.post(ENDPOINT, json=no_target, headers=HEADERS).status_code == 422
    assert (
        client.post(
            ENDPOINT, json={**spec, "candidate": "b" * 24}, headers=HEADERS
        ).status_code
        == 409
    )
    for mutation in (
        {"scope": "full"},
        {"angles": {"yaw": 181, "pitch": 0, "roll": 0}},
    ):
        assert (
            client.post(
                ENDPOINT, json={**spec, **mutation}, headers=HEADERS
            ).status_code
            == 422
        )
    assert client.post(ENDPOINT, json=spec, headers=HEADERS).status_code == 200
    raw.write_text('{"changed":true}', encoding="utf-8")
    assert not client.get(ENDPOINT).json()["items"][0]["current"]
    assert client.post(ENDPOINT, json=spec, headers=HEADERS).status_code == 409
    image.write_bytes(b"new image")
    assert client.get(ENDPOINT).status_code == 409


def test_exclusion_keeps_review_but_prevents_reuse(review_env):
    client, _, _, _, spec = review_env
    assert client.post(ENDPOINT, json=spec, headers=HEADERS).status_code == 200
    assert (
        client.post(
            "/api/head/queries/rough_test/review",
            json={"content_hash": spec["content_hash"], "excluded": True},
            headers=HEADERS,
        ).status_code
        == 200
    )
    rows = client.get(ENDPOINT).json()["items"]
    assert len(rows) == 1 and not rows[0]["current"]
    assert client.post(ENDPOINT, json=spec, headers=HEADERS).status_code == 409


def test_manual_target_is_bound_to_actual_input_points(review_env, monkeypatch):
    import pose_curation.review.head_review_routes as routes

    client, _, _, _, spec = review_env
    monkeypatch.setattr(routes, "canonical", lambda *args: FACE)
    manual = {k: v for k, v in spec.items() if k != "candidate"}
    manual["points"] = (
        projection(FACE, Orientation(30, 20, -10)) * 12 + [230, 250]
    ).tolist()
    saved = client.post(ENDPOINT, json=manual, headers=HEADERS)
    assert saved.status_code == 200
    row = saved.json()
    assert row["target"].startswith("manual:")
    assert row["provenance"]["points"] == manual["points"]
    assert row["provenance"]["original_orientation"]["yaw"] == 30
    assert (
        client.post(
            ENDPOINT, json={**manual, "candidate": "a" * 24}, headers=HEADERS
        ).status_code
        == 422
    )
    assert (
        client.post(
            ENDPOINT, json={**manual, "points": [[0, 0]] * 6}, headers=HEADERS
        ).status_code
        == 409
    )


def test_changed_fitter_makes_old_decision_stale(review_env, monkeypatch):
    import pose_curation.review.head_review_routes as routes

    client, _, _, _, spec = review_env
    client.post(ENDPOINT, json=spec, headers=HEADERS)
    monkeypatch.setattr(routes, "code_revision", lambda: "new fitter")
    row = client.get(ENDPOINT).json()["items"][0]
    assert not row["current"] and "코드" in row["stale_reason"]
