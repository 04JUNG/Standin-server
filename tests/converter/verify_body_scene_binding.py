"""Real codec equivalence and rollback with synthetic CI fixture only."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import bpy
from io_scene_fbx import encode_bin, parse_fbx
from converter import retarget as rt
from converter.shape_bake import import_full_character
from converter.body_scene_transfer import canonical_frames

assets, out = map(Path, sys.argv[sys.argv.index('--')+1:])
out.mkdir(parents=True, exist_ok=False)
rt.import_character = import_full_character
rt.reset_scene()
arm, meshes = import_full_character(str(assets/'character_mixamo.fbx'))
before_data = [(obj, obj.data) for obj in meshes]
original_write = encode_bin.write
before_open = [(module, vars(module).get('open'), 'open' in vars(module)) for module in (encode_bin, parse_fbx)]


def broken_write(*args, **kwargs):
    raise RuntimeError('synthetic codec failure')


encode_bin.write = broken_write
try:
    try:
        canonical_frames(out/'failure.fbx')
        raise AssertionError('codec failure was not propagated')
    except RuntimeError as exc:
        assert str(exc) == 'synthetic codec failure'
finally:
    encode_bin.write = original_write
assert all(obj.data == data for obj, data in before_data)
assert all(('open' in vars(module)) == present and vars(module).get('open') is value
           for module, value, present in before_open)
assert not list(out.glob('*.memory-rig.fbx'))

# Successful export immediately after failure also proves stale exporter wrapper
# state cannot poison the original disk recovery path.
baseline = out/'baseline.fbx'
rt.export_fbx(str(baseline), embed_textures=False, bake_anim=False)
memory, size = canonical_frames(out/'candidate.fbx')
rt.reset_scene()
arm, _ = import_full_character(str(baseline))
disk = {b.name: arm.matrix_world @ b.matrix_local for b in arm.data.bones}
assert set(memory) == set(disk)
error = max(abs(memory[name][r][c]-disk[name][r][c]) for name in disk for r in range(4) for c in range(4))
assert error == 0, error
assert not list(out.glob('*.memory-rig.fbx'))
assert all(('open' in vars(module)) == present and vars(module).get('open') is value
           for module, value, present in before_open)
(out/'verification.json').write_text(json.dumps(dict(ok=True, bones=len(disk), max_matrix_error=error,
    memory_bytes=size, codec_failure_restored_mesh_data=True, codec_hooks_restored=True,
    recovery_disk_export_succeeded=True), indent=2))
print('PASS memory binding and codec failure recovery', flush=True)
