"""GPU regression: a smooth plane must not acquire depth-quantization stripes."""
import numpy as np
import pytest


def test_smooth_tilted_surface_has_no_internal_outline():
    pytest.importorskip('moderngl')
    from pose_curation.thumbnail_style.renderer import Renderer
    try:
        renderer = Renderer()
    except Exception as error:
        pytest.skip(f'OpenGL 3.3 context unavailable: {error}')
    try:
        positions = np.array([[-1, -1, -1], [1, -1, 1],
                              [1, 1, 1], [-1, 1, -1]], dtype='f4')
        normals = np.tile(np.array([-1, 0, 1]) / np.sqrt(2), (4, 1)).astype('f4')
        indices = np.array([0, 1, 2, 0, 2, 3], dtype='u4')
        renderer.load([(positions, normals, indices)])
        image = np.asarray(renderer.render(0))
        # Stay away from the real silhouette and its antialiasing pixels.
        interior = image[100:400, 100:400].astype(int)
        assert np.max(np.ptp(interior, axis=(0, 1))) <= 1
        # The surface still has a visible silhouette against its white surround.
        assert np.max(image[256, 256]) < np.min(image[0, 0])
    finally:
        renderer.ctx.release()
