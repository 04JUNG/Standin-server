"""Opt-in Blender integration QA: -- character.fbx output-dir pose.bvh [...].

Run with the pinned Blender --background --python-exit-code 1 --python script.
Outputs FBX/PNG plus aggregate checks; never reads user images or publishes files.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from converter.convert import convert
from converter.framing import process_output
from converter import retarget as rt


def inspect(path):
    rt.reset_scene()
    arm, meshes = rt.import_character(str(path))
    bones = {b.name: {"parent": b.parent.name if b.parent else None,
                      "matrix": [v for row in b.matrix_local for v in row]}
             for b in arm.data.bones}
    return bones, sum(len(m.data.vertices) for m in meshes), sum(len(m.data.uv_layers) for m in meshes)


def main():
    character, output, *poses = sys.argv[sys.argv.index('--')+1:]
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for pose in poses:
        name = Path(pose).stem
        full = root/f'{name}-source.fbx'
        report = convert(bvh_path=str(Path(pose).resolve()), character_fbx=str(Path(character).resolve()),
                         out_path=str(full), output_mode='rigged_rest')
        assert report.ok, report
        source_bones, source_vertices, source_uv = inspect(full)
        for scope in ('full', 'half', 'bust', 'head'):
            dest = root/f'{name}-{scope}.fbx'
            shutil.copyfile(full, dest)
            before = hashlib.sha256(dest.read_bytes()).hexdigest()
            details = process_output(dest, scope, preview_path=root/f'{name}-{scope}.png')
            if scope == 'full':
                assert hashlib.sha256(dest.read_bytes()).hexdigest() == before
            bones, vertices, uv = inspect(dest)
            assert set(bones) == set(source_bones)
            for key, bone in bones.items():
                assert bone['parent'] == source_bones[key]['parent']
                assert max(abs(a-b) for a,b in zip(bone['matrix'], source_bones[key]['matrix'])) < 1e-4
            assert uv == source_uv
            assert vertices == source_vertices if scope == 'full' else 0 < vertices < source_vertices
            results.append({"pose": name, "scope": scope, **details, "roundtrip_bones": len(bones), "uv_layers": uv})
    (root/'verification.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('FRAMING_QA_OK', len(results))


if __name__ == '__main__':
    main()
