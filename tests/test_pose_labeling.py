"""Presentation relabeling must not rewrite a reviewed motion or its provenance."""

import pytest

from pose_curation import labeling
from pose_curation.storage import read_json, write_json


def test_style100_labels_change_only_presentation_fields(tmp_path, monkeypatch):
    batch = tmp_path / "styles"
    batch.mkdir()
    pose = {
        "pose_id": "style100_Balance_ID_f000001_test",
        "clip": "Balance_ID",
        "source": "100style",
        "source_sha256": "source-hash",
        "bvh_sha256": "motion-hash",
        "thumbnails": {"front": {"sha256": "preview-hash"}},
        "review_decision": "accepted",
    }
    write_json(batch / "manifest.json", {"poses": [pose]})
    labels = tmp_path / "labels.json"
    label = {"style": "balance", "movement": "sports", "category": "sports", "category_label": "스포츠"}
    write_json(labels, {"Balance": label})
    rebuilt = []
    monkeypatch.setattr(labeling, "build_candidates", lambda poses, path, name: rebuilt.append((poses, path, name)))

    assert labeling.apply(batch, labels) == {"poses": 1, "styles": 1}
    saved = read_json(batch / "manifest.json")["poses"][0]
    assert {key: saved[key] for key in label} == label
    assert {key: saved[key] for key in pose} == pose
    assert len(rebuilt) == 1


def test_style100_labels_reject_incomplete_or_extra_fields(tmp_path, monkeypatch):
    batch = tmp_path / "styles"
    batch.mkdir()
    write_json(batch / "manifest.json", {"poses": [{"clip": "Balance_ID", "source": "100style"}]})
    labels = tmp_path / "labels.json"
    monkeypatch.setattr(labeling, "build_candidates", lambda *args: pytest.fail("must not rebuild invalid labels"))

    write_json(labels, {"Balance": {"style": "balance"}})
    with pytest.raises(ValueError, match="requires only"):
        labeling.apply(batch, labels)
    write_json(labels, {"Balance": {"style": "balance", "movement": "sports", "category": "sports", "category_label": "스포츠", "source": "tampered"}})
    with pytest.raises(ValueError, match="requires only"):
        labeling.apply(batch, labels)
