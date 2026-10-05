"""Cleanup preserves meaningful differences and never prunes transitively."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from pose_curation.cleanup.features import VERSION, LIMITS, compare, extract
from pose_curation.cleanup.plan import representative_groups
from pose_curation.cleanup.apply import validate, VIEWS
from pose_curation.review.catalog import Pose
from pose_curation.review.store import ReviewStore
from pose_curation.storage import sha256
from tests.test_pose_curation import make_motion
from pose_curation.motion import Motion
from pose_curation.hands.bvh import augment
from src.bvh import parse_bvh, channel_starts, hierarchy_text


def fixture_pose(tmp_path):
    raw, body, path = [tmp_path / n for n in ('motion.bvh', 'body.bvh', 'hands.bvh')]
    make_motion(raw)
    Motion.load(raw).export(90, body)
    augment(body, path, left='fist', right='fist')
    return path


def changed_frame(source, target, joint, angles):
    joints, frames = parse_bvh(str(source))
    i = next(i for i, row in enumerate(joints) if row[0] == joint)
    start = channel_starts(joints)[i]
    for n, channel in enumerate(joints[i][3]):
        if channel in angles:
            frames[0, start + n] += angles[channel]
    target.write_text(hierarchy_text(str(source)) + '\nMOTION\nFrames: 1\nFrame Time: 0.0333333\n' +
                      ' '.join(str(v) for v in frames[0]) + '\n', encoding='utf-8')


def test_heading_and_translation_match_but_inversion_does_not(tmp_path):
    path = fixture_pose(tmp_path)
    moved, inverted = tmp_path / 'moved.bvh', tmp_path / 'inverted.bvh'
    changed_frame(path, moved, 'Hips', {'Yrotation': 77, 'Xposition': 900, 'Zposition': -400})
    changed_frame(path, inverted, 'Hips', {'Zrotation': 165})
    assert compare(extract(path), extract(moved))[1]
    assert not compare(extract(path), extract(inverted))[1]


@pytest.mark.parametrize('joint,channel', [('Head', 'Yrotation'), ('LeftHandIndex1', 'Zrotation'), ('LeftWrist', 'Xrotation')])
def test_body_similarity_cannot_hide_head_hand_or_finger_changes(tmp_path, joint, channel):
    path = fixture_pose(tmp_path)
    other = tmp_path / 'other.bvh'
    changed_frame(path, other, joint, {channel: 80})
    a, b = extract(path), extract(other)
    assert np.allclose(a['body'], b['body'])
    assert not compare(a, b)[1]


def test_missing_or_partial_fingers_are_not_automatically_equated(tmp_path):
    a = extract(fixture_pose(tmp_path))
    for count in (0, 2):
        b = deepcopy(a)
        b['finger_names'] = b['finger_names'][:count]
        b['fingers'] = b['fingers'][:count]
        assert not compare(b, b)[1]
        assert not compare(a, b)[1]


def test_direct_representative_prevents_transitive_loss():
    groups = representative_groups(['a', 'b', 'c'], [('a', 'b', {}), ('b', 'c', {})],
                                   {'a': 0, 'b': 1, 'c': 2})
    assert groups == [{'keeper': 'a', 'members': [{'key': 'b', 'metrics': {}}]}]


def validation_fixture(tmp_path):
    path = fixture_pose(tmp_path)
    thumbs = {}
    for view in VIEWS:
        thumbs[view] = tmp_path / (view + '.jpg')
        thumbs[view].write_bytes(view.encode())
    poses = {key: Pose(key, key, 'existing', 's3-v1', path, sha256(path), thumbs)
             for key in ('a', 'b', 'c')}
    plan = {'feature_version': VERSION, 'limits': LIMITS, 'records': {
        key: {'content_hash': p.content_hash, 'review': {'status': 'pending', 'note': ''},
              'thumbnail_sha256': {v: sha256(f) for v, f in thumbs.items()}}
        for key, p in poses.items()}}
    choices = {'reviewer': 'test reviewer', 'decisions': [
        {'key': 'b', 'keeper': 'a', 'kind': 'duplicate', 'reason': 'same four views', 'reviewed_views': list(VIEWS)}]}
    return plan, choices, poses


def test_stale_preview_and_excluded_representative_fail_closed(tmp_path):
    plan, choices, poses = validation_fixture(tmp_path)
    assert len(validate(plan, choices, poses, {})) == 1
    choices['decisions'].append({**choices['decisions'][0], 'key': 'a', 'keeper': 'c'})
    with pytest.raises(ValueError, match='retained direct'):
        validate(plan, choices, poses, {})
    choices['decisions'].pop()
    poses['a'].thumbnails['front'].write_bytes(b'changed')
    with pytest.raises(ValueError, match='preview changed'):
        validate(plan, choices, poses, {})


def test_missing_four_view_review_does_not_mutate_sources(tmp_path):
    plan, choices, poses = validation_fixture(tmp_path)
    choices['decisions'][0]['reviewed_views'] = ['front']
    with pytest.raises(ValueError, match='four-view'):
        validate(plan, choices, poses, {})
    assert sha256(poses['a'].bvh) == poses['a'].content_hash


def test_batch_exclusion_detects_concurrent_review_and_preserves_events(tmp_path):
    store = ReviewStore(tmp_path / 'reviews.sqlite')
    store.save('a', 'a' * 64, 'accepted', 'keep', evidence={'test': True})
    snapshot = store.all()
    store.save('b', 'b' * 64, 'hold', 'user changed this')
    rows = [{'key': 'a', 'content_hash': 'a' * 64, 'note': 'duplicate'},
            {'key': 'b', 'content_hash': 'b' * 64, 'note': 'duplicate'}]
    with pytest.raises(ValueError, match='reviews changed'):
        store.reject_batch(rows, expected_reviews=snapshot)
    assert store.all()[('a', 'a' * 64)]['status'] == 'accepted'
    store.reject_batch(rows, expected_reviews=store.all())
    assert store.all()[('a', 'a' * 64)]['status'] == 'rejected'
    assert store.all()[('a', 'a' * 64)]['evidence'] == {'test': True}
    with store.connection() as con:
        assert con.execute('SELECT count(*) FROM review_events').fetchone()[0] == 4


def test_g1_optional_spine_and_partial_legacy_hands(monkeypatch):
    from types import SimpleNamespace
    from pose_curation.rendering import diagnostics
    names = ['Hips', 'Spine', 'Spine1', 'LeftArm', 'RightArm', 'LeftHandIndex1']
    monkeypatch.setattr(diagnostics, 'parse_bvh', lambda _: ([(n,) for n in names], []))
    pose = SimpleNamespace(metadata={}, bvh=Path('fixture.bvh'), group='existing')
    assert diagnostics.rig_profile(pose) == 'mixamo_noprefix'
    assert diagnostics.finger_mode(pose) == 'legacy_body_only'
    pose.group = 'new'
    assert diagnostics.finger_mode(pose) == 'strict'


def test_coverage_prevents_losing_useful_root_directions():
    from pose_curation.cleanup.coverage import top_hit_counts, compare_reports
    person = {'quantitative_eligible': True, 'keypoints': [[0, 1]], 'scores': [1],
              'hits': [{'pose_id': 'useful-view', 'distance': .3}]}
    before = {'model': 'fixed', 'method': 'unchanged', 'images': [
        {'id': 'anonymous', 'image_sha256': 'hash', 'people': [person]}]}
    after = deepcopy(before)
    after['images'][0]['people'][0]['hits'] = [{'pose_id': 'other-view', 'distance': .4}]
    counts = top_hit_counts(before)
    assert counts['useful-view'] == 1 and counts['other-view'] == 0
    assert compare_reports(before, after)['summary']['max_increase'] == pytest.approx(.1)
    groups = representative_groups(['other-view', 'useful-view'],
                                   [('other-view', 'useful-view', {})],
                                   {key: -counts[key] for key in ['other-view', 'useful-view']})
    assert groups[0]['keeper'] == 'useful-view'
    after['images'][0]['people'][0]['keypoints'][0][0] = .5
    with pytest.raises(ValueError, match='query joints'):
        compare_reports(before, after)


def test_atomic_batch_rolls_back_on_write_failure(tmp_path, monkeypatch):
    store = ReviewStore(tmp_path / 'reviews.sqlite')
    store.save('a', 'a' * 64, 'accepted', 'before')
    store.save('b', 'b' * 64, 'accepted', 'before')
    before = store.all()
    write = store._write

    def fail_second(con, record, evidence):
        if record['pose_key'] == 'b':
            raise RuntimeError('simulated write failure')
        write(con, record, evidence)

    monkeypatch.setattr(store, '_write', fail_second)
    with pytest.raises(RuntimeError):
        store.reject_batch([{'key': key, 'content_hash': key * 64, 'note': 'duplicate'}
                            for key in ('a', 'b')], expected_reviews=before)
    assert store.all() == before
    with store.connection() as con:
        assert con.execute('SELECT count(*) FROM review_events').fetchone()[0] == 2


def test_report_escapes_source_labels():
    from pose_curation.cleanup.report import render
    report = {'before': {'published': 10}, 'after': {'published': 9}, 'removals': [
        {'key': 'old', 'pose_id': '<script>bad</script>', 'group': 'existing',
         'kind': 'quality', 'reason': 'bad <angle>'}]}
    page = render(report)
    assert '<script>bad</script>' not in page
    assert '&lt;script&gt;bad&lt;/script&gt;' in page
