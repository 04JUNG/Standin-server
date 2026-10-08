"""Preview packaging/identity checks: the style must survive container deployment."""
from pathlib import Path
from converter.preview_style import studio_parameters


def test_preview_studio_has_finite_soft_diffuse_lighting():
    import math
    parameters = studio_parameters()
    assert len(parameters['lights']) == 2
    assert all(0.2 <= value <= 0.4 for value in parameters['ambient'])
    for light in parameters['lights']:
        assert all(math.isfinite(value) for value in light['direction'])
        assert abs(sum(value * value for value in light['direction']) - 1) < 1e-5
        assert 0 < light['wrap'] <= 1


def test_container_includes_style_code_and_studio_asset():
    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / 'Dockerfile.converter').read_text()
    for name in ('preview_style.py', 'preview_studio.sl'):
        assert f'COPY converter/{name} /app/converter/{name}' in dockerfile


def test_review_identity_tracks_studio_asset():
    from pose_curation.rendering.batch import render_identity
    from unittest.mock import patch
    from pose_curation.qa.policy import CHARACTER_SHA256
    root = Path(__file__).resolve().parents[2]
    with patch('pose_curation.rendering.batch.sha256', return_value=CHARACTER_SHA256):
        identity = render_identity(root / 'character.fbx', root / 'blender.exe')
    assert 'converter/preview_studio.sl' in identity['code']
    assert 'converter/preview_style.py' in identity['code']
