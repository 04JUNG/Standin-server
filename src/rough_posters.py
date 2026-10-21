"""Read prepared yaw8 assets in the search frame. Never render on a card request."""
from pathlib import Path

from .directional_posters import VIEWS, SCOPES, read_directional_poster
from .pose_posters import read_poster, headers

VARIANT = 'rough-yaw8-v1'


def native_view(search_view: str) -> str:
    """Native camera right=-Ry(180-A)[0]; search right=Ry(Y)[0], so A=-Y.

    Only choose a physically rendered view; never reflect a raster or a pose.
    Both paths use the original BVH frame, not an anatomical facing reset.
    """
    if search_view not in VIEWS:
        raise ValueError('invalid search view')
    return f'yaw_{(-int(search_view[4:])) % 360:03d}'


def rough_poster(data_dir: Path, source: str, scope: str, view: str):
    native = native_view(view)
    fallback = 'none'
    if scope not in SCOPES:
        image, meta = read_poster(data_dir / 'pose-posters', source, scope)
        result = headers(meta)
        result.update({
            'X-Standin-Poster-Variant': VARIANT,
            'X-Standin-Poster-Yaw': '0',
            'X-Standin-Poster-Fallback': 'unsupported_scope',
            'X-Standin-Projection-Matches-Search': 'false',
        })
        return image, result
    try:
        image, result = read_directional_poster(data_dir, source, scope, native)
        served = int(view[4:])
    except FileNotFoundError:
        # Absence permits a same-source/scope reference. Corruption never does.
        try:
            image, result = read_directional_poster(data_dir, source, scope, 'yaw_000')
        except FileNotFoundError:
            image, meta = read_poster(data_dir / 'pose-posters', source, scope)
            result = headers(meta)
        served, fallback = 0, 'missing_angle' if view != 'yaw_000' else 'missing_directional_asset'
    result.update({
        'X-Standin-Poster-Variant': VARIANT,
        'X-Standin-Poster-Yaw': str(served),
        'X-Standin-Poster-Fallback': fallback,
        'X-Standin-Projection-Matches-Search': 'false',
    })
    return image, result
