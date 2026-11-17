"""Inspect a pinned authoring blend without running embedded scripts."""
import argparse,json,sys
from pathlib import Path
import bpy
p=argparse.ArgumentParser();p.add_argument('--blend');p.add_argument('--out');a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
bpy.ops.wm.open_mainfile(filepath=a.blend)
rows=[]
for o in bpy.context.scene.objects:
 if o.type=='MESH':
  groups={g.index:g.name for g in o.vertex_groups}; counts={}
  for v in o.data.vertices:
   for g in v.groups:
    if g.weight>.05:counts[groups[g.group]]=counts.get(groups[g.group],0)+1
  rows.append(dict(name=o.name,vertices=len(o.data.vertices),faces=len(o.data.polygons),keys=[k.name for k in o.data.shape_keys.key_blocks] if o.data.shape_keys else [],modifiers=[(m.name,m.type) for m in o.modifiers],uv=[u.name for u in o.data.uv_layers],props=list(o.keys()),group_counts=counts))
Path(a.out).write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
