"""Blender-only: reload delivered FBXs and compare exact geometry, bones, weights.

Runs outside timed benchmark sections. Does not change artifact bytes.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from converter import retarget as rt
import bpy

def fingerprint(path):
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    rt.reset_scene();arm,meshes=rt.import_character(str(path))
    h=hashlib.sha256();vertex_count=0;triangle_count=0
    def floats(values):
        for v in values:h.update(struct.pack('<d',float(v)))
    def label(v):h.update(str(v).encode());h.update(b'\0')
    for bone in sorted(arm.data.bones,key=lambda b:b.name):
        label(bone.name);label(bone.parent.name if bone.parent else None)
        floats(x for row in bone.matrix_local for x in row)
        floats(bone.head_local);floats(bone.tail_local)
        pose=arm.pose.bones[bone.name]
        floats(x for row in pose.matrix for x in row)
        floats(x for row in pose.matrix_basis for x in row)
    for obj in sorted(meshes,key=lambda m:m.name):
        label(obj.name);floats(x for row in obj.matrix_world for x in row)
        groups={g.index:g.name for g in obj.vertex_groups}
        for vertex in obj.data.vertices:
            floats(vertex.co);label([(groups[g.group],float(g.weight)) for g in vertex.groups])
        for poly in obj.data.polygons:label(tuple(poly.vertices))
        ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh()
        try:
            for vertex in mesh.vertices:floats(ev.matrix_world@vertex.co)
            vertex_count+=len(mesh.vertices);mesh.calc_loop_triangles();triangle_count+=len(mesh.loop_triangles)
        finally:ev.to_mesh_clear()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    return dict(geometry_sha256=h.hexdigest(),vertices=vertex_count,triangles=triangle_count,bones=len(arm.data.bones))

args=sys.argv[sys.argv.index('--')+1:];folder=Path(args[0]).resolve();summary=json.loads((folder/'summary.json').read_text())
assert summary['status']=='timing_complete', 'benchmark did not finish'
assert summary['artifacts'], 'no artifacts to compare'
assert summary['comparisons'] and all(r['business_results_equal'] for r in summary['comparisons']), 'business results changed'
presets={r['preset'] for r in summary['rows']}
expected={preset+'__'+sha for preset in presets for sha in summary['fixtures'].values()}
groups=defaultdict(list)
for row in summary['artifacts']:
    assert hashlib.sha256(Path(row['artifact']).read_bytes()).hexdigest()==row['artifact_sha256'], 'artifact bytes changed'
    fp=fingerprint(Path(row['artifact']));groups[row['case']].append(dict(artifact=row['artifact'],**fp))
assert set(groups)==expected, 'fixture coverage incomplete'
assert all(len(rows)>=2 for rows in groups.values()), 'missing comparison artifacts'
parity={case:len({r['geometry_sha256'] for r in rows})==1 for case,rows in groups.items()}
result=dict(all_equal=all(parity.values()),exact_float_geometry=True,
    artifacts_verified=sum(len(rows) for rows in groups.values()),
    comparison='Reloaded mesh vertices, evaluated skin, topology, bind and posed bone matrices and vertex weights; exporter metadata excluded.',
    parity=parity,cases=dict(groups))
(folder/'geometry-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('GEOMETRY_PARITY',json.dumps(parity),flush=True)
if not result['all_equal']:raise AssertionError('geometry changed')
