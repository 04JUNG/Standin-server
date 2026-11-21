"""Health-only source verification; export/shape execution keeps its full gates."""
from pathlib import Path
from converter.shape_spec import load_registry, VERSION_RE, ShapeError
from converter_api.verified_files import asset_digests


def health_shape_registry():
    registry = load_registry(verify_files=False)
    if registry is None:
        return None
    for rules, source in registry['sources'].items():
        for key in ('blend', 'controller'):
            path = Path(source[key])
            if path.is_symlink() or asset_digests.digest(path) != source[key + '_sha256']:
                raise ShapeError('shape source checksum mismatch')
        if VERSION_RE.findall(Path(source['controller']).read_text(encoding='utf-8')) != [rules]:
            raise ShapeError('controller VERSION does not match shape rules')
    return registry
