"""Offline contract tests using an original procedural skeleton, no dataset files."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pose_curation.motion import Motion, normalized_body
from pose_curation.pipeline import run
from pose_curation.review.app import create_app
from pose_curation.review.store import ReviewStore
from pose_curation.selection import select_frames
from pose_curation.sources.style100 import parse_catalog
from pose_curation.storage import contained_path, sha256, write_json
from src.bvh import parse_bvh, load_coco17
from src.library import build_entries_from_pose
from src.repo import build_db, load_entries


def make_motion(path: Path) -> None:
    """Simple T-rest rig with moving elbows, translated and yawed root."""
    hierarchy, names = [], []

    def joint(name, offset, children=(), root=False, depth=0):
        indent = "  " * depth
        names.append(name)
        hierarchy.extend([f"{indent}{'ROOT' if root else 'JOINT'} {name}", indent + "{",
                          f"{indent} OFFSET {' '.join(map(str, offset))}",
                          f"{indent} CHANNELS {'6 Xposition Yposition Zposition' if root else '3'} Yrotation Xrotation Zrotation"])
        for child in children:
            joint(*child, depth=depth + 1)
        if not children:
            tip = "12 0 0" if name == "LeftWrist" else "-12 0 0" if name == "RightWrist" else "0 3 0"
            hierarchy.extend([indent + " End Site", indent + " {", indent + " OFFSET " + tip, indent + " }"])
        hierarchy.append(indent + "}")

    arms = []
    legs = []
    for side, sign in (("Left", 1), ("Right", -1)):
        arms.append((side + "Collar", (sign * 5, 3, 0), [(side + "Shoulder", (sign * 8, 0, 0),
                     [(side + "Elbow", (sign * 15, 0, 0), [(side + "Wrist", (sign * 15, 0, 0))])])]))
        legs.append((side + "Hip", (sign * 7, -5, 0), [(side + "Knee", (0, -40, 0),
                    [(side + "Ankle", (0, -40, 0), [(side + "Toe", (0, 0, 8))])])]))
    spine = ("Chest", (0, 12, 0), [("Chest2", (0, 10, 0), [("Chest3", (0, 10, 0),
             [("Chest4", (0, 10, 0), [("Neck", (0, 8, 0), [("Head", (0, 10, 0))]), *arms])])])])
    joint("Hips", (0, 0, 0), [spine, *legs], root=True)
    frames = np.zeros((240, 3 + len(names) * 3))
    frames[:, :6] = [12, 100, 25, 45, 7, 2]
    elbow = 6 + (names.index("LeftElbow") - 1) * 3
    frames[:, elbow + 2] = 75 * np.sin(np.arange(240) / 25)
    text = "HIERARCHY\n" + "\n".join(hierarchy) + "\nMOTION\nFrames: 240\nFrame Time: 0.016667\n"
    text += "\n".join(" ".join(f"{value:.6f}" for value in frame) for frame in frames) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def library(tmp_path):
    data = tmp_path / "data"
    raw = tmp_path / "motion.bvh"
    make_motion(raw)
    motion = Motion.load(raw)
    bvh = data / "bvh" / "baseline.bvh"
    motion.export(90, bvh)
    kp, scores = load_coco17(str(bvh))
    entries = build_entries_from_pose("baseline", kp, {"shot": "full_half", "action": "other", "relationship": "solo"},
                                      "data/bvh/baseline.bvh", scores)
    build_db(entries, str(data / "poses.db"))
    return data, raw


def test_motion_export_preserves_height_tilt_and_hierarchy(library, tmp_path):
    _, raw = library
    motion = Motion.load(raw)
    target = tmp_path / "pose.bvh"
    motion.export(80, target)
    joints, frames = parse_bvh(str(target))
    assert frames.shape == (1, motion.frames.shape[1])
    assert np.allclose(frames[0, :6], [0, 100, 0, 0, 7, 2])
    assert np.allclose(frames[0, 6:], motion.frames[80, 6:])
    assert [j[0] for j in joints] == [j[0] for j in motion.joints]
    assert raw.read_text().split("MOTION")[0] == target.read_text().split("MOTION")[0]
    indices, points = motion.sample(15, 220, 8)
    selected = select_frames(points, indices * motion.frame_time, count=3, minimum_distance=.12, minimum_seconds=.3)
    assert 2 <= len(selected) <= 3
    for candidate in selected:
        assert 15 < indices[candidate.sample_index] < 220
    assert selected == select_frames(points, indices * motion.frame_time, count=3, minimum_distance=.12, minimum_seconds=.3)


def test_static_clip_does_not_generate_duplicate_poses(library):
    _, raw = library
    motion = Motion.load(raw)
    kp, _ = motion.keypoints(motion.canonical_frame(90))
    points = np.repeat(kp[None], 30, axis=0)
    assert len(select_frames(points, np.arange(30) / 8, count=8, minimum_distance=.12, minimum_seconds=.3)) == 1
    assert np.allclose(normalized_body(kp), normalized_body(kp * 10 + [200, 20, -50]))


def test_catalog_requires_source_and_trim_contract():
    html = '<a href="https://drive.google.com/uc?id=a">Angry_ID.bvh</a>'
    cuts = "STYLE_NAME,ID_START,ID_STOP\nAngry,10,200\n"
    clips = parse_catalog(html, cuts)
    assert (clips[0].style, clips[0].movement, clips[0].start, clips[0].stop) == ("Angry", "ID", 10, 200)
    with pytest.raises(ValueError):
        parse_catalog(html.replace("drive.google.com", "unexpected.example"), cuts)
    with pytest.raises(ValueError):
        parse_catalog(html, cuts.replace("10,200", "N/A,N/A"))


def test_offline_build_resume_provenance_and_repair(library, tmp_path):
    data, raw = library
    curation = data / "curation"
    source = curation / "sources" / "100style"
    source.mkdir(parents=True)
    url = "https://drive.google.com/uc?id=fixture"
    (source / "catalog.html").write_text(f'<a href="{url}">Angry_ID.bvh</a>')
    (source / "Frame_Cuts.csv").write_text("STYLE_NAME,ID_START,ID_STOP\nAngry,10,230\n")
    cached = source / "raw" / "Angry_ID.bvh"
    cached.parent.mkdir()
    cached.write_bytes(raw.read_bytes())
    write_json(cached.with_suffix(".bvh.source.json"), {"url": url, "sha256": sha256(cached)})
    config = tmp_path / "config.json"
    write_json(config, {"schema_version": 1, "source": "100style", "styles": ["Angry"], "movements": ["ID"],
                        "poses_per_clip": 3, "sample_hz": 8, "minimum_separation_seconds": .3,
                        "minimum_pose_distance": .12, "near_duplicate_distance": .16})
    baseline = sha256(data / "poses.db")
    first = run(config, data, curation, "test", offline=True)
    assert first["status"] == "complete" and 1 <= len(first["poses"]) <= 3
    pose = first["poses"][0]
    assert pose["license"] == "CC-BY-4.0" and pose["source_sha256"] == sha256(cached)
    batch = curation / "batches" / "test"
    bvh = batch / pose["bvh"]
    timestamp = bvh.stat().st_mtime_ns
    again = run(config, data, curation, "test", offline=True)
    assert again["poses"] == first["poses"] and bvh.stat().st_mtime_ns == timestamp
    bvh.write_text("broken")
    repaired = run(config, data, curation, "test", offline=True)
    assert sha256(bvh) == repaired["poses"][0]["bvh_sha256"]
    entries = load_entries(str(batch / "candidates.db"))
    assert len(entries) == 4 * len(first["poses"])
    assert all(entry.meta["source"] == "100style" for entry in entries)
    assert sha256(data / "poses.db") == baseline
    # Existing + new are independently addressable; review is not publication.
    with TestClient(create_app(data, curation), base_url="http://127.0.0.1") as client:
        assert client.get("/api/summary").json()["groups"] == {"new": len(first["poses"]), "existing": 1}
        candidates = client.get("/api/poses?group=new&limit=1").json()
        item = candidates["items"][0]
        assert candidates["total"] == len(first["poses"])
        assert item["preview_kind"] == "pending" and item["views"] == []
        assert client.get(f"/api/poses/{item['key']}/thumbnail?view=front").status_code == 404
        assert client.get(f"/api/poses/{item['key']}/skeleton").json()["frames"] == 1


def test_review_is_durable_revision_bound_and_requires_local_header(library):
    data, _ = library
    curation = data / "curation"
    app = create_app(data, curation)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        item = client.get("/api/poses").json()["items"][0]
        url = f"/api/poses/{item['key']}/review"
        body = {"status": "hold", "note": "Check ankle", "content_hash": item["content_hash"]}
        assert client.put(url, json=body).status_code == 403
        assert client.put(url, json=body, headers={"X-Pose-Review": "1", "Origin": "https://untrusted.example"}).status_code == 403
        assert client.put(url, json={**body, "content_hash": "0" * 64}, headers={"X-Pose-Review": "1"}).status_code == 409
        assert client.put(url, json=body, headers={"X-Pose-Review": "1"}).status_code == 200
        assert client.get("/api/poses?status=hold").json()["total"] == 1
        assert client.get("/api/poses?group=new").json()["total"] == 0
        assert client.get("/api/poses?limit=10000").status_code == 422
        assert client.get(f"/api/poses/{item['key']}/thumbnail?view=../../poses.db").status_code == 422
        assert client.get("/api/poses/unknown/bvh").status_code == 404
    with TestClient(create_app(data, curation), base_url="http://127.0.0.1") as client:
        assert client.get(f"/api/poses/{item['key']}").json()["review"]["note"] == "Check ankle"
    store = ReviewStore(curation / "reviews.sqlite")
    assert (item["key"], "changed-content") not in store.all()
    with sqlite3.connect(curation / "reviews.sqlite") as con:
        assert con.execute("SELECT COUNT(*) FROM review_events").fetchone()[0] == 1


def test_paths_cannot_escape_data_directory(tmp_path):
    with pytest.raises(ValueError):
        contained_path(tmp_path, "../secrets")


def test_framing_endpoint_checks_scope_origin_and_serves_images(library, monkeypatch):
    from pose_curation.review.framing import FramedPreviews
    data, _ = library
    calls = []

    def status(self, pose, scope, *, start=False):
        calls.append((pose.key, scope, start))
        return {"status": "rendering", "scope": scope}

    monkeypatch.setattr(FramedPreviews, 'status', status)
    with TestClient(create_app(data, data / 'curation'), base_url='http://127.0.0.1') as client:
        key = client.get('/api/poses').json()['items'][0]['key']
        endpoint = f'/api/poses/{key}/framing'
        assert client.post(endpoint + '?scope=half').status_code == 403
        headers = {'X-Pose-Review': '1'}
        assert client.post(endpoint + '?scope=invalid', headers=headers).status_code == 422
        assert client.post(endpoint + '?scope=head', headers={**headers, 'Origin': 'https://other.test'}).status_code == 403
        assert calls == []
        assert client.post(endpoint + '?scope=half', headers=headers).json()['status'] == 'rendering'
        assert calls[-1] == (key, 'half', True)
        assert client.get(endpoint + '?scope=half').status_code == 200
        assert calls[-1] == (key, 'half', False)


def test_oriented_export_rejects_stale_revision_and_invalid_angles(library, monkeypatch):
    from pose_curation.review.framing import FramedPreviews
    data, _ = library
    calls = []
    def status(self, pose, scope, *, start=False, orientation=None):
        calls.append((scope, start, orientation))
        return {'status': 'ready', 'scope': scope, 'version': 'test'}
    monkeypatch.setattr(FramedPreviews, 'status', status)
    monkeypatch.setattr(FramedPreviews, 'artifact', lambda *args: None)
    with TestClient(create_app(data, data / 'curation'), base_url='http://127.0.0.1') as client:
        pose = client.get('/api/poses').json()['items'][0]
        path = f"/api/poses/{pose['key']}/oriented"
        spec = {'scope': 'full', 'yaw': 37, 'pitch': 25, 'roll': -18, 'content_hash': pose['content_hash']}
        headers = {'X-Pose-Review': '1'}
        assert client.post(path, json=spec).status_code == 403
        assert client.post(path, json=spec, headers={**headers, 'Origin': 'https://other.test'}).status_code == 403
        assert client.post(path, json={**spec, 'yaw': 181}, headers=headers).status_code == 422
        assert client.get(path, params={**spec, 'pitch': 'nan'}).status_code == 422
        assert client.post(path, json={**spec, 'content_hash': '0' * 64}, headers=headers).status_code == 409
        assert calls == []
        result = client.post(path, json=spec, headers=headers).json()
        assert result['orientation']['yaw'] == 37
        assert calls[-1][0:2] == ('full', True)
        assert client.get(path, params=spec).status_code == 200
        assert calls[-1][1] is False
        assert client.get(path + '/fbx', params={**spec, 'v': 'wrong'}).status_code == 409
        assert client.get(path + '/bvh', params={**spec, 'v': 'test'}).status_code == 422
        assert client.get(f"/api/poses/{pose['key']}/bvh").status_code == 200


def test_angle_triage_uses_palm_base_not_curled_fingertips(library, tmp_path):
    from pose_curation.audit import inspect_bvh
    from pose_curation.hands.bvh import augment
    from src.bvh import rotation_channel_indices, write_single_frame_bvh

    _, raw = library
    body = tmp_path / "body.bvh"
    Motion.load(raw).export(90, body)
    outputs = []
    for preset in ("open", "fist"):
        out = tmp_path / f"{preset}.bvh"
        augment(body, out, left=preset, right=preset)
        outputs.append(inspect_bvh(out))
    assert outputs[0]["angles"] == outputs[1]["angles"]
    assert outputs[1]["finger_joints"] == 30
    assert outputs[1]["flags"] == []
    joints, frames = parse_bvh(str(body))
    wrist = next(i for i, joint in enumerate(joints) if joint[0] == "LeftWrist")
    frames[0, rotation_channel_indices(joints, wrist)[-1]] = 100
    bent = tmp_path / "bent.bvh"
    write_single_frame_bvh(str(body), frames[0], str(bent))
    augment(bent, tmp_path / "bent-hands.bvh")
    assert any("LeftWrist" in flag for flag in inspect_bvh(tmp_path / "bent-hands.bvh")["flags"])


def test_exclusion_removes_pose_from_default_library_and_export_but_is_restorable(library):
    from pose_curation.publication import run as publish

    data, _ = library
    baseline_hash = sha256(data / "poses.db")
    curation = data / "curation"
    with TestClient(create_app(data, curation), base_url="http://127.0.0.1") as client:
        item = client.get("/api/poses").json()["items"][0]
        body = {"status": "rejected", "note": "Malformed knee", "content_hash": item["content_hash"]}
        client.put(f"/api/poses/{item['key']}/review", json=body, headers={"X-Pose-Review": "1"}).raise_for_status()
        assert client.get("/api/poses").json()["total"] == 0
        assert client.get("/api/poses?group=existing").json()["total"] == 0
        assert client.get("/api/summary").json()["excluded"] == 1
        assert client.get("/api/poses?group=excluded").json()["items"][0]["excluded"] is True
        assert client.get("/api/poses?status=rejected").json()["total"] == 1
        assert client.get(f"/api/poses/{item['key']}/bvh").status_code == 200
        result = publish(data, curation)
        assert result["poses"] == 0 and len(result["excluded"]) == 1
        assert load_entries(str(curation / "library/poses.db")) == []
        client.put(f"/api/poses/{item['key']}/review", json={**body, "status": "pending"}, headers={"X-Pose-Review": "1"}).raise_for_status()
        assert client.get("/api/poses").json()["total"] == 1
        assert client.get("/api/poses?group=excluded").json()["total"] == 0
        publish(data, curation)
    original = load_entries(str(data / "poses.db"))
    restored = load_entries(str(curation / "library/poses.db"))
    assert len(restored) == len(original) == 4
    assert all(np.array_equal(a.feature, b.feature) for a, b in zip(original, restored))
    assert all(Path(p.bvh_path).is_file() for p in restored)
    assert sha256(data / "poses.db") == baseline_hash


def test_publication_can_write_an_alternate_snapshot(library):
    from pose_curation.publication import run as publish

    data, _ = library
    curation = data / "curation"
    destination = curation / "library-next"
    result = publish(data, curation, destination_dir=destination)
    assert result["poses"] == 1
    assert (destination / "poses.db").is_file()
    assert not (curation / "library/poses.db").exists()


def test_publication_requires_current_review_and_previews_and_preserves_last_good_db(library):
    from pose_curation.publication import run as publish
    from pose_curation.review.catalog import Catalog
    from PIL import Image

    data, raw = library
    curation = data / "curation"
    batch = curation / "batches/test"
    bvh = batch / "bvh/new.bvh"
    Motion.load(raw).export(60, bvh)
    points, scores = load_coco17(str(bvh))
    entries = build_entries_from_pose("new", points, {"shot": "full_half", "action": "running", "relationship": "solo"}, str(bvh), scores)
    build_db(entries, str(batch / "candidates.db"))
    record = {"pose_id": "new", "bvh": "bvh/new.bvh", "bvh_sha256": sha256(bvh), "thumbnails": {}, "preview_kind": "pending"}
    manifest = {"schema_version": 1, "batch_id": "test", "status": "complete", "poses": [record]}
    write_json(batch / "manifest.json", manifest)
    assert publish(data, curation)["new"] == 0
    pose = next(p for p in Catalog(data, curation).all() if p.group == "new")
    store = ReviewStore(curation / "reviews.sqlite")
    store.save(pose.key, "0" * 64, "accepted", "old revision")
    assert publish(data, curation)["new"] == 0
    store.save(pose.key, pose.content_hash, "accepted", "visual review")
    last_good = sha256(curation / "library/poses.db")
    with pytest.raises(ValueError, match="four character previews"):
        publish(data, curation)
    assert sha256(curation / "library/poses.db") == last_good
    for view in ("front", "side", "back", "three_quarter"):
        path = batch / f"{view}.jpg"
        Image.new("RGB", (16, 16), "gray").save(path)
        record["thumbnails"][view] = {"path": path.name, "sha256": sha256(path)}
    record["preview_kind"] = "character"
    for anatomy in (None, {"bvh_sha256": "0" * 64, "flags": []},
                    {"bvh_sha256": pose.content_hash, "flags": ["arm intersects torso"]}):
        record["anatomy_check"] = anatomy
        write_json(batch / "manifest.json", manifest)
        with pytest.raises(ValueError, match="anatomy checks"):
            publish(data, curation)
        assert sha256(curation / "library/poses.db") == last_good
    record["anatomy_check"] = {"bvh_sha256": pose.content_hash, "flags": []}
    write_json(batch / "manifest.json", manifest)
    # A quiet legacy flags array alone is not full QA evidence. The valid
    # end-to-end publication fixture is covered in test_pose_qa.py.
    with pytest.raises(ValueError, match="QA gate failed"):
        publish(data, curation)
    assert sha256(curation / "library/poses.db") == last_good


def test_character_preview_publication_is_complete_and_revision_bound(library, tmp_path):
    from PIL import Image
    from pose_curation.rendering.batch import apply_result, VIEWS
    from pose_curation.rendering.profiles import STYLE100
    from converter.bone_map import CANONICAL_BONES

    assert set(STYLE100) == set(CANONICAL_BONES)
    data, raw = library
    batch = data / "curation" / "batches" / "render-test"
    bvh = batch / "bvh" / "candidate.bvh"
    Motion.load(raw).export(90, bvh)
    pose = {"pose_id": "candidate", "bvh": "bvh/candidate.bvh", "bvh_sha256": sha256(bvh), "thumbnails": {}}
    result = {"ok": True, "bvh_sha256": sha256(bvh), "render_fingerprint": "render-v1",
              "blender_version": "5.2.0 LTS", "thumbnails": {}}
    for view in VIEWS:
        path = batch / "character-thumbs" / f"candidate__{view}.jpg"
        path.parent.mkdir(exist_ok=True)
        Image.new("RGB", (256, 256), (153, 153, 153)).save(path)
        result["thumbnails"][view] = {"path": str(path), "sha256": sha256(path)}
    identity = {"renderer": "test", "character": "standin-master-v2", "character_sha256": "a" * 64}
    assert not apply_result(batch, pose, {**result, "ok": False}, "render-v1", identity)
    assert not apply_result(batch, pose, result, "wrong-render-version", identity)
    assert not apply_result(batch, pose, {**result, "bvh_sha256": "wrong-pose"}, "render-v1", identity)
    broken = {**result, "thumbnails": {v: t for v, t in result["thumbnails"].items() if v != "back"}}
    assert not apply_result(batch, pose, broken, "render-v1", identity)
    assert pose["thumbnails"] == {}
    assert apply_result(batch, pose, result, "render-v1", identity)
    write_json(batch / "manifest.json", {"schema_version": 1, "batch_id": "render-test", "status": "complete", "poses": [pose]})
    with TestClient(create_app(data, data / "curation"), base_url="http://127.0.0.1") as client:
        item = client.get("/api/poses?group=new").json()["items"][0]
        assert item["preview_kind"] == "character"
        assert item["thumbnail_versions"]["front"] == result["thumbnails"]["front"]["sha256"]
        assert client.get(f"/api/poses/{item['key']}/thumbnail?view=front").headers["content-type"] == "image/jpeg"
        # Rerendering leaves the BVH unchanged but must invalidate image caches.
        front = Path(result["thumbnails"]["front"]["path"])
        Image.new("RGB", (256, 256), (210, 210, 210)).save(front)
        result["thumbnails"]["front"]["sha256"] = sha256(front)
        assert apply_result(batch, pose, result, "render-v1", identity)
        write_json(batch / "manifest.json", {"schema_version": 1, "batch_id": "render-test", "status": "complete", "poses": [pose]})
        refreshed = client.get(f"/api/poses/{item['key']}").json()
        assert refreshed["content_hash"] == item["content_hash"]
        assert refreshed["thumbnail_versions"]["front"] != item["thumbnail_versions"]["front"]
    Path(result["thumbnails"]["front"]["path"]).write_bytes(b"corrupt")
    assert not apply_result(batch, pose, result, "render-v1", identity)
