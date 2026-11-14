"""Write scene dependency closure, excluding temporary/orphan source meshes.

All semantic arrays and driver graphs must match after reloading the smaller
container. This is storage cleanup, measured separately from vertex reduction.
"""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import bpy,numpy as np
p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--out',required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);source=Path(a.candidate);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def semantic():
 m=next(o for o in bpy.context.scene.objects if o.get('standin_studio_mesh'));preview=next(o for o in bpy.context.scene.objects if o.get('standin_preview_lod'));arm=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
 h=hashlib.sha256()
 for obj in (m,preview):
  for v in obj.data.vertices:h.update(np.array(v.co,dtype='<f4').tobytes())
  h.update(json.dumps([list(p.vertices) for p in obj.data.polygons]).encode())
  for key in obj.data.shape_keys.key_blocks if obj.data.shape_keys else []:
   arr=np.empty(len(key.data)*3,dtype='<f4');key.data.foreach_get('co',arr);h.update(key.name.encode()+arr.tobytes())
 for obj in (m,arm,preview):h.update(json.dumps({k:str(v) for k,v in obj.items()},sort_keys=True).encode())
 for data in (m.data.shape_keys,arm):
  if data.animation_data:
   for d in data.animation_data.drivers:
    h.update(json.dumps([d.data_path,d.array_index,d.driver.expression,[(v.name,v.type,[(getattr(t.id,'name',None),t.data_path,t.bone_target,t.transform_type,t.transform_space) for t in v.targets]) for v in d.driver.variables]],sort_keys=True).encode())
 return h.hexdigest()
bpy.ops.wm.open_mainfile(filepath=str(source/'model.blend'));before=semantic();bpy.data.libraries.write(str(out/'model.blend'),{bpy.context.scene},fake_user=False,compress=True);bpy.ops.wm.open_mainfile(filepath=str(out/'model.blend'));after=semantic();assert before==after,'packing changed semantic data'
bpy.context.preferences.filepaths.save_version=0;bpy.ops.wm.save_as_mainfile(filepath=str(out/'model.blend'),compress=True)
for item in source.iterdir():
 if item.is_file() and item.name not in ('model.blend','manifest.json'):shutil.copyfile(item,out/item.name)
manifest=json.loads((source/'manifest.json').read_text());manifest['container_cleanup']={'source_blend_sha256':sha(source/'model.blend'),'semantic_sha256':after,'bytes_before':(source/'model.blend').stat().st_size,'bytes_after':(out/'model.blend').stat().st_size};manifest['files']={x.name:sha(x) for x in out.iterdir() if x.is_file()};(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest['container_cleanup']),flush=True)
