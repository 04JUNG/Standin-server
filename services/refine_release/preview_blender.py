"""Bake a display-only GLB from the verified FBX; never edit the FBX or pose."""
import bpy, sys
from pathlib import Path
source,destination=map(Path,sys.argv[sys.argv.index('--')+1:])
if destination.exists(): raise FileExistsError(destination)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.fbx(filepath=str(source))
deps=bpy.context.evaluated_depsgraph_get();objects=[]
for original in list(bpy.context.scene.objects):
    if original.type!='MESH':continue
    mesh=bpy.data.meshes.new_from_object(original.evaluated_get(deps),depsgraph=deps)
    obj=bpy.data.objects.new('verified_'+original.name,mesh)
    obj.matrix_world=original.matrix_world.copy();objects.append(obj)
for original in list(bpy.context.scene.objects):bpy.data.objects.remove(original,do_unlink=True)
for obj in objects:bpy.context.scene.collection.objects.link(obj)
if not objects:raise ValueError('FBX has no display mesh')
bpy.ops.export_scene.gltf(filepath=str(destination),export_format='GLB',export_animations=False,export_yup=True)
