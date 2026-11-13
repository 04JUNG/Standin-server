"""Actual-model shape comparison with identical geometric-normal diagnostics.

All three columns use recomputed geometric normals to isolate geometry. They
are not production shading exports or CSP acceptance artifacts.
"""
import json, sys, hashlib, argparse
from pathlib import Path
import bpy, numpy as np
from mathutils import Vector
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from converter.local import inspection, surface
from converter import retarget as rt
from scripts.local_converter.render_followup import render

def main(base, motion, finish, out):
    out.mkdir(parents=True, exist_ok=False)
    records = json.loads((finish / 'finish-results.json').read_text())
    evidence = []
    for record in records:
        mi, pi = (record['model'], record['pose'])
        data = np.load(base / f'stages-{mi}-{pi}.npz')
        a = np.load(motion / f"{mi}-{pi}-{record['stage2Candidate']}.npz")['points']
        b = np.load(finish / f'{mi}-{pi}-1.0.npz')['points']
        rows = {'v10': data['corrected'], 'motion-rejected': a, 'finish-rejected': b}
        allp = np.concatenate(list(rows.values()))
        center = (allp.min(0) + allp.max(0)) / 2
        scale = float(np.ptp(allp, axis=0).max())
        row = []
        for label, points in rows.items():
            bpy.ops.wm.read_factory_settings(use_empty=True)
            arm, meshes = inspection.import_fbx(base / f'stage-{mi}-{pi}-before-surface.fbx')
            obj = meshes[0]
            weights = [sorted(((obj.vertex_groups[g.group].name, g.weight) for g in v.groups)) for v in obj.data.vertices]
            hierarchy = {x.name: x.parent.name if x.parent else None for x in arm.data.bones}
            inv = obj.matrix_world.inverted()
            for v, p in zip(obj.data.vertices, points):
                v.co = inv @ Vector(p)
            obj.data.update()
            obj.data.calc_loop_triangles()
            f = np.array([t.vertices[:] for t in obj.data.loop_triangles])
            normals = surface.geometric_normals(np.array([v.co[:] for v in obj.data.vertices]), f)
            obj.data.normals_split_custom_set_from_vertices(normals.tolist())
            obj.data.update()
            path = out / f'{mi}-{pi}-{label}.fbx'
            rt.export_fbx(str(path), embed_textures=False)
            bpy.ops.wm.read_factory_settings(use_empty=True)
            check, cm = inspection.import_fbx(path)
            new = np.asarray(inspection.surface(cm)[cm[0].name])
            error = float(np.linalg.norm(new - points, axis=1).max())
            assert error < 2e-05
            assert weights == [sorted(((cm[0].vertex_groups[g.group].name, g.weight) for g in v.groups)) for v in cm[0].data.vertices]
            assert all((check.data.bones.get(n) is not None and (check.data.bones[n].parent.name if check.data.bones[n].parent else None) == p for n, p in hierarchy.items()))
            row.append(dict(label=label, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), maxReimportPositionErrorM=error, originalWeightsHierarchyPreserved=True))
            for back in (False, True):
                render(path, out / f'{mi}-{pi}-{label}-{back}.png', center, scale, back)
        evidence.append(dict(model=mi, pose=pi, normalPolicy='geometric for every column, diagnostic only; not production appearance', files=row))
        (out / 'render-evidence.json').write_text(json.dumps(evidence, indent=2))
        print('VISUAL_STAGE', mi, pi, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--motion', type=Path, required=True)
    p.add_argument('--finish', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    main(a.base, a.motion, a.finish, a.out)
