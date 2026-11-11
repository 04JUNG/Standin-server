"""Create small real-FBX culling controls for the exclusive CSP UI owner.

Run in Blender: --python converter/outline_probe.py -- DESTINATION [SOURCE_FBX]
Without a source this creates a small two-bone bent-finger-like control.
These are experiments, never product-ready output.
"""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from converter.outline_worker import run, sha
from converter.outline_spec import VERSION


def material_control(destination,source):
    """Measure whether a Blender backface-transparent graph survives FBX.

    This is an alternate transport probe, not a base-preserving exporter.
    The authoritative candidate uses typed-tree augmentation instead.
    """
    import bpy
    from converter.outline_worker import import_source
    from converter.retarget import export_fbx
    from converter.outline_fbx import PREFIX
    destination.mkdir(parents=True,exist_ok=False)
    arm,meshes=import_source(source)
    for obj in meshes:
        if not obj.name.encode().startswith(PREFIX):continue
        mat=bpy.data.materials.new('Backface transparent graph')
        mat.use_nodes=True;mat.diffuse_color=(0,0,0,1)
        nodes=mat.node_tree.nodes;nodes.clear();links=mat.node_tree.links
        geo=nodes.new('ShaderNodeNewGeometry');mix=nodes.new('ShaderNodeMixShader')
        clear=nodes.new('ShaderNodeBsdfTransparent');black=nodes.new('ShaderNodeBsdfPrincipled')
        black.inputs['Base Color'].default_value=(0,0,0,1)
        black.inputs['Roughness'].default_value=1
        output=nodes.new('ShaderNodeOutputMaterial')
        links.new(geo.outputs['Backfacing'],mix.inputs[0])
        links.new(black.outputs[0],mix.inputs[1]);links.new(clear.outputs[0],mix.inputs[2])
        links.new(mix.outputs[0],output.inputs[0]);obj.data.materials.clear();obj.data.materials.append(mat)
    before=sorted({n.bl_idname for m in meshes if m.name.encode().startswith(PREFIX) for s in m.material_slots for n in s.material.node_tree.nodes})
    path=destination/'backface-shader.fbx';export_fbx(str(path),embed_textures=False,bake_anim=False)
    _,imported=import_source(path)
    after=sorted({n.bl_idname for m in imported if m.name.encode().startswith(PREFIX) for s in m.material_slots for n in s.material.node_tree.nodes})
    (destination/'manifest.json').write_text(json.dumps(dict(source=str(source.resolve()),
        path=str(path.resolve()),sha256=sha(path),before_nodes=before,after_nodes=after,
        backface_graph_preserved='ShaderNodeNewGeometry' in after,
        csp_verified=False,product_candidate=False),indent=2))


def small_source(path):
    import bpy
    from converter.retarget import reset_scene,export_fbx
    reset_scene()
    bpy.ops.object.armature_add()
    arm=bpy.context.object;arm.name='ProbeRig'
    bpy.ops.object.mode_set(mode='EDIT')
    arm.data.edit_bones[0].name='mixamorig:Hips'
    arm.data.edit_bones[0].head=(0,0,0);arm.data.edit_bones[0].tail=(0,0,1)
    b=arm.data.edit_bones.new('mixamorig:Spine');b.head=(0,0,1);b.tail=(.4,0,1.6);b.parent=arm.data.edit_bones[0]
    bpy.ops.object.mode_set(mode='OBJECT')
    mat=bpy.data.materials.new('ProbeClay');mat.diffuse_color=(.65,.65,.65,1)
    mat.use_nodes=True;mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=mat.diffuse_color
    for i,(x,z,scale) in enumerate([(0,.55,(.24,.18,.55)),(.20,1.25,(.12,.12,.38)),(.46,1.5,(.08,.08,.18))]):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=12,location=(x,0,z))
        obj=bpy.context.object;obj.name=f'ProbeBody{i}';obj.scale=scale
        bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
        obj.data.materials.append(mat)
        for p in obj.data.polygons:p.use_smooth=True
        group=obj.vertex_groups.new(name='mixamorig:Hips' if i==0 else 'mixamorig:Spine')
        group.add(list(range(len(obj.data.vertices))),1.,'REPLACE')
        mod=obj.modifiers.new('Skin','ARMATURE');mod.object=arm;obj.parent=arm
    export_fbx(str(path),embed_textures=False,bake_anim=False)


def main():
    args=sys.argv[sys.argv.index('--')+1:]
    if args[0]=='material':
        material_control(Path(args[1]),Path(args[2]));return
    destination=Path(args[0]);destination.mkdir(parents=True,exist_ok=True)
    source=Path(args[1]) if len(args)>1 else destination/'control-zero.fbx'
    if len(args)==1:small_source(source)
    import shutil
    source_sha=sha(source);results=[]
    for culling in ('CullingOff','CullingOnCCW','CullingOnCW'):
        d=destination/culling;d.mkdir(exist_ok=False)
        shutil.copyfile(source,d/'source.fbx')
        job=dict(source_artifact_sha256=source_sha,appearance=dict(version=VERSION,width=2),
            stature_m=1.75,output_scope='full',character_id='probe',character_sha256=source_sha,
            binding_sha256='0'*64,culling=culling)
        (d/'job.json').write_text(json.dumps(job))
        start=time.monotonic();run('export',d)
        results.append(dict(path=str((d/'artifact.fbx').resolve()),sha256=sha(d/'artifact.fbx'),
            bytes=(d/'artifact.fbx').stat().st_size,culling=culling,seconds=time.monotonic()-start))
    (destination/'manifest.json').write_text(json.dumps(dict(source=str(source),source_sha256=source_sha,
        candidates=results,csp_verified=False),indent=2))

if __name__=='__main__':main()
