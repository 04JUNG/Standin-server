"""Preview cache must never serve a different BVH revision or partial job."""
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

from pose_curation.review.framing import FramedPreviews, VIEWS
from pose_curation.storage import sha256, write_json
from pose_curation.orientation import Orientation


def test_cache_requires_all_views_and_current_revision(tmp_path):
    service = FramedPreviews(tmp_path)
    service._identity = {'renderer': 'test'}
    pose = SimpleNamespace(content_hash='a' * 64, metadata={}, group='existing')
    fingerprint = service.identity(pose, 'half')
    directory = service.root / fingerprint
    directory.mkdir(parents=True)
    thumbnails = {}
    for view in VIEWS:
        path = directory / f'preview__{view}.jpg'
        path.write_bytes(b'test-image')
        thumbnails[view] = {'sha256': sha256(path)}
    result = {'ok': True, 'fingerprint': fingerprint, 'thumbnails': thumbnails}
    write_json(directory / 'result.json', result)
    assert service.status(pose, 'half')['status'] == 'ready'
    assert service.image(pose, 'half', 'front', fingerprint).is_file()
    assert service.image(pose, 'head', 'front', fingerprint) is None
    pose.content_hash = 'b' * 64
    assert service.image(pose, 'half', 'front', fingerprint) is None
    pose.content_hash = 'a' * 64
    (directory / 'preview__back.jpg').write_bytes(b'corrupt')
    assert service.status(pose, 'half')['status'] == 'missing'
    assert service.image(pose, 'half', 'front', fingerprint) is None
    service.close()


def test_oriented_artifact_bound_to_angles_and_all_file_hashes(tmp_path):
    from pose_curation.review.framing import ORIENTED_FILES
    service = FramedPreviews(tmp_path)
    service._identity = {'renderer': 'test'}
    bvh = tmp_path / 'pose.bvh'
    bvh.write_bytes(b'original')
    pose = SimpleNamespace(content_hash=sha256(bvh), bvh=bvh, metadata={}, group='existing')
    angles = Orientation(37, 25, -18)
    fingerprint = service.identity(pose, 'half', angles)
    directory = service.root / fingerprint
    directory.mkdir(parents=True)
    files = {}
    for kind, filename in ORIENTED_FILES.items():
        path = directory / filename
        path.write_bytes(kind.encode())
        files[kind] = {'file': filename, 'sha256': sha256(path)}
    write_json(directory / 'result.json', {'ok': True, 'kind': 'orientation',
               'fingerprint': fingerprint, 'files': files})
    assert service.status(pose, 'half', orientation=angles)['status'] == 'ready'
    assert service.artifact(pose, 'half', angles, 'fbx', fingerprint).is_file()
    assert service.artifact(pose, 'half', Orientation(38,25,-18), 'fbx', fingerprint) is None
    assert service.artifact(pose, 'head', angles, 'bvh', fingerprint) is None
    assert service.artifact(pose, 'half', angles, 'bvh', fingerprint) is None
    (directory / ORIENTED_FILES['fbx']).write_bytes(b'corrupt')
    assert service.artifact(pose, 'half', angles, 'fbx', fingerprint) is None
    service.close()


def test_same_request_joins_job_and_failed_job_can_retry(tmp_path, monkeypatch):
    service = FramedPreviews(tmp_path)
    service._identity = {'renderer': 'test'}
    pose = SimpleNamespace(content_hash='a' * 64, metadata={}, group='new')
    jobs = []

    def submit(*args):
        future = Future()
        jobs.append(future)
        return future

    monkeypatch.setattr(service._executor, 'submit', submit)
    assert service.status(pose, 'head', start=True)['status'] == 'rendering'
    assert service.status(pose, 'head', start=True)['status'] == 'rendering'
    assert len(jobs) == 1
    jobs[0].set_exception(ValueError('bad crop'))
    assert service.status(pose, 'head')['error'] == 'bad crop'
    assert service.status(pose, 'head', start=True)['status'] == 'rendering'
    assert len(jobs) == 2
    jobs[1].cancel()
    assert service.status(pose, 'head')['status'] == 'failed'
    service.close()
