"""Blender contract: cached immutable geometry produces identical safety metrics."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from converter import ankle_swing as original
from converter.rest_surface import RestSurface

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
p = argparse.ArgumentParser()
p.add_argument('--rows', type=int, default=24)
p.add_argument('--columns', type=int, default=48)
p.add_argument('--output', type=Path)
a = p.parse_args(args)
vertices = [(.06 * math.sin(j / a.columns * math.pi), -.2 + 1.4 * j / a.columns,
             -.1 + 1.2 * i / a.rows) for i in range(a.rows + 1) for j in range(a.columns + 1)]
faces = []
for i in range(a.rows):
 for j in range(a.columns):
  k = i * (a.columns + 1) + j
  faces.append((k, k + 1, k + a.columns + 2, k + a.columns + 1))
data = bpy.data.meshes.new('rest-safety-test'); data.from_pydata(vertices, [], faces); data.update()
mesh = bpy.data.objects.new('rest-safety-test', data); bpy.context.collection.objects.link(mesh)
for name, value in [('leg', .55), ('foot', .45)]:
 g = mesh.vertex_groups.new(name=name); g.add(list(range(len(vertices))), value, 'REPLACE')
arm = SimpleNamespace(matrix_world=Matrix.Identity(4))
points = {'leg': Vector((0, 0, 1)), 'foot': Vector((0, 0, 0)), 'toe': Vector((0, 1, 0))}
rt = SimpleNamespace(_rest_world=lambda arm, name: SimpleNamespace(translation=points[name]))
table = {f'{name}.L': name for name in points}
policy = json.loads((Path(__file__).resolve().parents[2] / 'converter/airborne_plantar_policy.json').read_text())
rest = [Vector(v) for v in vertices]
weights = original._vertex_weight_rows(mesh)
start = time.monotonic(); cache = RestSurface(original, mesh, arm, rt, table, 'L', rest, policy, weights)
build = time.monotonic() - start
reference_time = optimized_time = 0
cases = {}
for label, positions in [
 ('neutral', [v.copy() for v in rest]),
 ('mild', [Vector((v.x + .01 * math.sin(v.y * 6), v.y, v.z)) for v in rest]),
 ('folded', [Vector((v.x, v.y, -v.z if v.y > .4 else v.z)) for v in rest]),
 ('degenerate', [Vector((0, 0, 0)) for v in rest]),
]:
 start = time.monotonic(); before = original._surface_metrics(mesh, arm, rt, table, 'L', rest, positions, policy)
 reference_time += time.monotonic() - start
 start = time.monotonic(); after = cache.measure(positions); optimized_time += time.monotonic() - start
 assert asdict(before) == asdict(after), (label, asdict(before), asdict(after))
 cases[label] = {'status': after.status, 'metricsExactlyEqual': True}
# Degenerate rig and absent groups keep the same fail-closed reasons.
for label, target_weights, target_points in [('no_weights', [{} for _ in rest], points),
                                          ('zero_rig', weights, {k:Vector() for k in points})]:
 target_rt = SimpleNamespace(_rest_world=lambda arm, name: SimpleNamespace(translation=target_points[name]))
 if label == 'no_weights':
  for g in list(mesh.vertex_groups): mesh.vertex_groups.remove(g)
 before = original._surface_metrics(mesh, arm, target_rt, table, 'L', rest, rest, policy)
 after = RestSurface(original, mesh, arm, target_rt, table, 'L', rest, policy, target_weights).measure(rest)
 assert asdict(before) == asdict(after), label
 cases[label] = {'status': after.status, 'metricsExactlyEqual': True}
result = {'blender': bpy.app.version_string, 'vertices': len(rest), 'cases': cases,
          'buildSeconds': build, 'referenceSeconds': reference_time,
          'cachedMeasurementsSeconds': optimized_time, 'optimizedTotalSeconds': build + optimized_time,
          'note': 'isolated measurements; not a complete export latency benchmark'}
if a.output: a.output.write_text(json.dumps(result, indent=2))
print('REST_SURFACE_VERIFIED ' + json.dumps(result), flush=True)
