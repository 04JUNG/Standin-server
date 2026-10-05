"""QA gates on synthetic fixtures; no Blender, AWS, or dataset download."""

from pathlib import Path
import sqlite3

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from tests.test_pose_curation import library
from pose_curation.motion import Motion
from pose_curation.hands.bvh import augment
from pose_curation.candidates import build_candidates
from pose_curation.review.catalog import Catalog
from pose_curation.review.store import ReviewStore
from pose_curation.review.selection import decision
from pose_curation.review.app import create_app
from pose_curation.qa import policy
from pose_curation.qa.checks import inspect_pose, preview_issues
from pose_curation.qa.workflow import run
from pose_curation.publication import run as publish
from pose_curation.storage import sha256, write_json


@pytest.fixture
def candidate(library, tmp_path):
    data, raw = library
    curation = data / "curation"
    batch = curation / "batches/test"
    body = tmp_path / "body.bvh"
    Motion.load(raw).export(0, body)
    bvh = batch / "hands/preset/new.bvh"
    hands = augment(body, bvh)
    hands["captured_from_source"] = False
    digest = sha256(bvh)
    record = {
        "pose_id": "new",
        "bvh": bvh.relative_to(batch).as_posix(),
        "bvh_sha256": digest,
        "preview_kind": "character",
        "source": "fixture",
        "author": "Test fixture",
        "license": "CC0-1.0",
        "source_url": "https://example.org/test",
        "source_sha256": sha256(raw),
        "movement": "ID",
        "source_frame_0based": 0,
        "hand_augmentation": hands,
        "thumbnails": {},
        "anatomy_check": {
            "bvh_sha256": digest,
            "policy_fingerprint": policy.fingerprint(),
            "flags": [],
            "torso_segment_rotation_degrees": {"Spine": 0, "Spine1": 0, "Spine2": 0},
            "hinge_mismatch_degrees": {"Left": None, "Right": None},
            "skin_intersections": {"Left": 0, "Right": 0},
        },
    }
    for view in policy.VIEWS:
        path = batch / f"{view}.jpg"
        Image.new("RGB", (32, 32), "gray").save(path)
        record["thumbnails"][view] = {"path": path.name, "sha256": sha256(path)}
    record["preview"] = {
        "character_sha256": policy.CHARACTER_SHA256,
        "fingerprint": "fixture-render",
        "report": "render-results/fixture.json",
    }
    write_json(
        batch / "render-results/fixture.json",
        {
            "ok": True,
            "bvh_sha256": digest,
            "render_fingerprint": "fixture-render",
            "fingers": {"applied_joints": 30},
            "thumbnails": record["thumbnails"],
        },
    )
    manifest = {
        "schema_version": 1,
        "batch_id": "test",
        "status": "complete",
        "poses": [record],
    }

    def save():
        write_json(batch / "manifest.json", manifest)

    def pose():
        return next(p for p in Catalog(data, curation).all() if p.group == "new")

    save()
    build_candidates([record], batch, "test")
    return data, curation, batch, record, save, pose


def evidence(pose):
    return {
        "policy_fingerprint": policy.fingerprint(),
        "bvh_sha256": pose.content_hash,
        "thumbnail_sha256": pose.metadata["thumbnail_versions"],
        "visual_checks": list(policy.VISUAL_CHECKS),
        "resolved_findings": [],
    }


def test_valid_candidate_is_never_auto_approved_and_nested_hands_path_works(
    candidate, tmp_path
):
    data, curation, batch, record, save, get = candidate
    pose = get()
    assert preview_issues(pose) == []
    result = inspect_pose(pose, {})
    assert result["automatic_ready"] and not result["publishable"]
    baseline = sha256(data / "poses.db")
    bvh = sha256(pose.bvh)
    report = run(data, curation, tmp_path / "qa", batch="test")
    assert report["counts"] == {"visual_review": 1}
    assert ReviewStore(curation / "reviews.sqlite").all() == {}
    assert sha256(pose.bvh) == bvh and sha256(data / "poses.db") == baseline
    assert (tmp_path / "qa/report.html").exists() and len(report["sheets"]) == 1


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("finger", "bvh.invalid"),
        ("trailing", "bvh.invalid"),
        ("hash", "bvh.invalid"),
        ("source", "source.author"),
        ("spine", "mesh.torso_segment_rotation_degrees.Spine"),
        ("skin", "mesh.skin_intersections.Left"),
        ("nan", "mesh.missing"),
        ("policy", "mesh.stale"),
        ("preview", "preview.invalid"),
    ],
)
def test_fail_closed_for_corrupt_data_and_quiet_bad_metrics(candidate, mutation, code):
    _, _, batch, r, save, get = candidate
    if mutation in {"finger", "trailing"}:
        path = batch / r["bvh"]
        text = path.read_text()
        text = (
            text.replace("LeftHandThumb2", "MissingThumb")
            if mutation == "finger"
            else text + " 123\n"
        )
        path.write_text(text)
        r["bvh_sha256"] = sha256(path)
    elif mutation == "hash":
        r["bvh_sha256"] = "0" * 64
    elif mutation == "source":
        r.pop("author")
    elif mutation == "spine":
        r["anatomy_check"]["torso_segment_rotation_degrees"]["Spine"] = 90
    elif mutation == "skin":
        r["anatomy_check"]["skin_intersections"]["Left"] = 1
    elif mutation == "nan":
        r["anatomy_check"]["skin_intersections"]["Left"] = float("nan")
    elif mutation == "policy":
        r["anatomy_check"]["policy_fingerprint"] = "old"
    elif mutation == "preview":
        (batch / "front.jpg").write_bytes(b"changed")
    save()
    result = inspect_pose(get(), {})
    assert not result["automatic_ready"]
    assert code in {f["code"] for f in result["findings"]}


def test_exclusion_survives_revision_until_explicit_restore(candidate):
    _, curation, _, r, save, get = candidate
    store = ReviewStore(curation / "reviews.sqlite")
    p = get()
    store.save(p.key, p.content_hash, "accepted", "earlier accepted revision")
    store.save(p.key, "old-hash", "rejected", "bad pose")
    assert decision(p, store.all())["inherited_exclusion"]
    store.save(p.key, p.content_hash, "pending", "explicit restoration")
    assert decision(p, store.all())["status"] == "pending"


def test_review_api_requires_checklist_then_binds_approval_to_pixels(candidate):
    data, curation, batch, r, save, get = candidate
    p = get()
    body = {
        "status": "accepted",
        "content_hash": p.content_hash,
        "note": "All views checked",
    }
    headers = {"X-Pose-Review": "1"}
    with TestClient(create_app(data, curation), base_url="http://127.0.0.1") as client:
        url = f"/api/poses/{p.key}/review"
        assert client.put(url, json=body, headers=headers).status_code == 422
        response = client.put(
            url,
            json={**body, "visual_checks": list(policy.VISUAL_CHECKS)},
            headers=headers,
        )
        assert response.status_code == 200, response.text
        assert (
            response.json()["evidence"]["thumbnail_sha256"]
            == p.metadata["thumbnail_versions"]
        )
        assert client.get(f"/api/poses/{p.key}/qa").json()["status"] == "approved"
        r["anatomy_check"]["skin_intersections"]["Left"] = 1
        save()
        assert (
            client.put(
                url,
                json={**body, "visual_checks": list(policy.VISUAL_CHECKS)},
                headers=headers,
            ).status_code
            == 409
        )


def test_warning_needs_explicit_resolution_and_publication_keeps_last_good(candidate):
    data, curation, batch, r, save, get = candidate
    p = get()
    store = ReviewStore(curation / "reviews.sqlite")
    store.save(p.key, p.content_hash, "accepted", "Four views", evidence=evidence(p))
    assert publish(data, curation)["new"] == 1
    before = sha256(curation / "library/poses.db")
    r["near_duplicate"] = True
    save()
    with pytest.raises(ValueError, match="QA gate failed"):
        publish(data, curation)
    assert sha256(curation / "library/poses.db") == before
    updated = evidence(get())
    updated["resolved_findings"] = ["duplicate"]
    store.save(
        p.key,
        p.content_hash,
        "accepted",
        "Different direction adds coverage",
        evidence=updated,
    )
    assert publish(data, curation)["new"] == 1
    Image.new("RGB", (32, 32), "white").save(batch / "front.jpg")
    with pytest.raises(ValueError, match="preview changed"):
        publish(data, curation)


def test_changed_composition_base_is_blocked(candidate):
    _, _, _, r, save, get = candidate
    r["composition_variant"] = {"base_pose_id": "base", "base_sha256": "old"}
    save()
    result = inspect_pose(get(), {}, bases={})
    assert "composition.stale" in {f["code"] for f in result["findings"]}


def test_visual_evidence_cannot_be_reused_after_policy_change(candidate):
    _, _, _, r, save, get = candidate
    p = get()
    ev = evidence(p)
    ev["policy_fingerprint"] = "old"
    result = inspect_pose(
        p,
        {
            (p.key, p.content_hash): {
                "status": "accepted",
                "note": "checked",
                "evidence": ev,
            }
        },
    )
    assert result["automatic_ready"] and not result["publishable"]


def test_prepare_reuses_nested_previews_and_does_not_promote_hold(
    candidate, tmp_path, monkeypatch
):
    from pose_curation.rendering import batch as renderer, diagnostics

    data, curation, _, record, _, get = candidate
    pose = get()
    store = ReviewStore(curation / "reviews.sqlite")
    store.save(pose.key, pose.content_hash, "hold", "Needs silhouette review")
    before = store.all()

    def no_render(*args, **kwargs):
        pytest.fail("Existing BVH-bound preview evidence must be reused")

    def diagnostic_result(*args, **kwargs):
        return {
            "poses": [
                {
                    "pose_id": pose.pose_id,
                    "bvh_sha256": pose.content_hash,
                    "ok": True,
                    "fingers": {"applied_joints": 30},
                    "anatomy": record["anatomy_check"],
                }
            ],
            "fingerprint": "test-diagnostics",
        }

    monkeypatch.setattr(renderer, "run", no_render)
    monkeypatch.setattr(diagnostics, "run", diagnostic_result)
    result = run(
        data, curation, tmp_path / "prepared", batch="test", prepare_evidence=True
    )
    assert result["counts"] == {"visual_review": 1}
    assert store.all() == before
    assert get().metadata["thumbnail_versions"] == pose.metadata["thumbnail_versions"]
