import hashlib
import json
import math

import pytest

from src.rough_posters import native_view, rough_poster
from tests.test_directional_posters import asset


@pytest.mark.parametrize('yaw', range(0, 360, 45))
def test_physical_native_camera_has_same_screen_axes_as_search(yaw):
    view = native_view(f'yaw_{yaw:03d}')
    native_yaw = math.radians((180 - int(view[4:])) % 360)
    y = math.radians(yaw)
    # Camera construction from render_human_mesh_review.stage_match. Independent
    # asymmetric points detect left/right reflection and a reversed yaw sign.
    for x, z in [(3, 1), (-2, 5), (1, -4)]:
        screen_x = -math.cos(native_yaw) * x - math.sin(native_yaw) * z
        assert screen_x == pytest.approx(math.cos(y) * x + math.sin(y) * z)


def test_serves_exact_selected_bytes_and_declares_missing_angle(tmp_path):
    source, front, *_ = asset(tmp_path)
    _, angled, *_ = asset(tmp_path, view='yaw_315')
    image, h = rough_poster(tmp_path, source, 'full', 'yaw_045')
    assert image == angled
    assert h['X-Standin-Preview-View'] == 'yaw_315'
    assert h['X-Standin-Poster-Yaw'] == '45'
    assert h['X-Standin-Poster-Fallback'] == 'none'
    image, h = rough_poster(tmp_path, source, 'full', 'yaw_090')
    assert image == front
    assert h['X-Standin-Poster-Yaw'] == '0'
    assert h['X-Standin-Poster-Fallback'] == 'missing_angle'


def test_corruption_never_falls_back_to_valid_front(tmp_path):
    source, *_ = asset(tmp_path)
    _, _, meta, _, path = asset(tmp_path, view='yaw_315')
    path.write_text(json.dumps({**meta, 'image_sha256': 'f' * 64}))
    with pytest.raises(ValueError):
        rough_poster(tmp_path, source, 'full', 'yaw_045')


def test_http_preserves_source_guard_and_old_png_contract(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from fastapi.responses import FileResponse
    import api.app as app
    from src import pose_posters
    raw = b'public registered BVH'
    source = hashlib.sha256(raw).hexdigest()
    bvh = tmp_path / 'source.bvh'
    bvh.write_bytes(raw)
    monkeypatch.setattr(app, 'get_pose_bvh', lambda _: FileResponse(bvh))
    monkeypatch.setattr(app.CFG, 'data_dir', str(tmp_path))
    old = b'old PNG'
    monkeypatch.setattr(pose_posters, 'poster', lambda *args: (old, {
        **pose_posters.identity(source), 'kind': 'default-body',
        'image_sha256': hashlib.sha256(old).hexdigest()}))
    client = TestClient(app.app)
    url = f'/pose/pose/poster?source_sha256={source}'
    assert client.get(url).content == old
    # An old bundle takes the legacy PNG path. Once advertised, the new bundle
    # uses its own native PNG adapter on old requests and our opt-in JPEG path.
    _, jpg, *_ = asset(tmp_path, view='yaw_315', source=raw)
    r = client.get(url + '&variant=rough-yaw8-v1&view=yaw_045')
    assert r.status_code == 200 and r.content == jpg
    assert r.headers['x-standin-poster-yaw'] == '45'
    assert client.get(url.replace(source, 'f' * 64) + '&variant=rough-yaw8-v1').status_code == 409
    assert client.get(url + '&variant=rough-yaw8-v1&view=yaw_001').status_code == 400
    assert client.get(url + '&view=yaw_045').status_code == 400
    # Missing a scope/source never borrows another asset or starts conversion.
    assert client.get(url + '&variant=rough-yaw8-v1&scope=head').status_code == 404
