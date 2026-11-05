"""Local real-file QA; inputs/outputs remain outside Git. No inference or CSP UI.

Blender --python converter/outline_qa.py -- export|verify ROOT
ROOT/inputs.json contains {sources:[{name,path,character_id,character_sha256,
stature_m}], scopes:[...], widths:[...]}. Verify runs in a fresh process.
"""
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from converter.outline_worker import run,sha,import_source
from converter.outline_spec import VERSION


def export(root):
    from converter.framing import crop_meshes
    from converter.retarget import export_fbx
    inputs=json.loads((root/'inputs.json').read_bytes());cases=[];zeros=[]
    for item in inputs['sources']:
        for scope in inputs['scopes']:
            base_dir=root/'bases'/item['name']/scope;base_dir.mkdir(parents=True,exist_ok=True)
            source=Path(item['path'])
            if scope!='full':
                arm,meshes=import_source(source);crop_meshes(arm,meshes,scope)
                source=base_dir/'framed.fbx';export_fbx(str(source),embed_textures=False,bake_anim=False)
            # Width 0 is the exact existing source/path, no outline file.
            zeros.append(dict(source=str(source.resolve()),source_sha256=sha(source),
                name=item['name'],scope=scope,width=0,legacy_path_unchanged=True))
            for width in inputs['widths']:
                d=root/'cases'/item['name']/scope/str(width);d.mkdir(parents=True,exist_ok=False)
                shutil.copyfile(source,d/'source.fbx')
                job=dict(source_artifact_sha256=sha(source),appearance=dict(version=VERSION,width=width),
                    stature_m=item['stature_m'],output_scope=scope,character_id=item['character_id'],
                    character_sha256=item['character_sha256'],binding_sha256='0'*64,culling='CullingOnCCW')
                (d/'job.json').write_text(json.dumps(job))
                start=time.monotonic();run('export',d)
                cases.append(dict(directory=str(d.resolve()),name=item['name'],scope=scope,width=width,
                    source_sha256=sha(source),output_sha256=sha(d/'artifact.fbx'),
                    source_bytes=source.stat().st_size,output_bytes=(d/'artifact.fbx').stat().st_size,
                    export_seconds=time.monotonic()-start))
                (root/'export-results.json').write_text(json.dumps(dict(zeros=zeros,cases=cases),indent=2))


def deformation(directory):
    """Same indexed skin + bind must transport the expansion with the body.

    Pose several real elbow/finger/knee bones, then compare measured output
    delta to the weighted skin-matrix transport of the original expansion.
    This catches a hull parented to an armature without actual skin weights.
    """
    import bpy
    from mathutils import Vector
    from converter.outline_fbx import PREFIX
    arm,meshes=import_source(directory/'artifact.fbx')
    base=sorted([m for m in meshes if not m.name.encode().startswith(PREFIX)],key=lambda m:m.name)
    hull=sorted([m for m in meshes if m.name.encode().startswith(PREFIX)],key=lambda m:m.name)
    if len(base)!=len(hull):raise ValueError('deformation pair mismatch')
    # Match by source name suffix; arbitrary extras were rejected by run verify.
    pairs=[]
    for m in base:
        candidates=[h for h in hull if h.name.endswith('_'+m.name)]
        if len(candidates)!=1:raise ValueError('deformation mesh identity mismatch')
        pairs.append((m,candidates[0]))
    angles={'mixamorig:LeftForeArm':.5,'mixamorig:RightForeArm':-.4,
        'mixamorig:LeftHandIndex1':.65,'mixamorig:LeftHandIndex2':.6,
        'mixamorig:RightHandMiddle1':.55,'mixamorig:LeftLeg':.4,
        'mixamorig:Spine2':.15,'mixamorig:Neck':.2}
    moved=[]
    for name,angle in angles.items():
        pb=arm.pose.bones.get(name)
        if pb: pb.rotation_mode='XYZ';pb.rotation_euler.x=angle;moved.append(name)
    if len(moved)<3: raise ValueError('insufficient real deformation bones')
    bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get()
    maximum=0.;motion=0.
    for m,h in pairs:
        a=m.evaluated_get(deps).data;b=h.evaluated_get(deps).data
        for v,w,actual_a,actual_b in zip(m.data.vertices,h.data.vertices,a.vertices,b.vertices):
            rest_delta=m.matrix_world.to_3x3() @ (w.co-v.co)
            expected=Vector((0,0,0));weight_sum=0.
            for influence in v.groups:
                bone=arm.pose.bones.get(m.vertex_groups[influence.group].name)
                if bone is None or influence.weight<=0:continue
                skin=arm.matrix_world @ bone.matrix @ bone.bone.matrix_local.inverted() @ arm.matrix_world.inverted()
                expected+=(skin.to_3x3() @ rest_delta)*influence.weight;weight_sum+=influence.weight
            expected/=weight_sum
            actual=m.matrix_world @ actual_b.co - m.matrix_world @ actual_a.co
            maximum=max(maximum,(actual-expected).length)
            motion=max(motion,(m.matrix_world @ actual_a.co-m.matrix_world @ v.co).length)
    if maximum>1e-5 or motion<.001:
        raise ValueError(f'outline deformation: error={maximum}m motion={motion}m')
    return dict(bones=moved,max_transport_error_m=maximum,max_body_motion_m=motion)


def verify(root):
    data=json.loads((root/'export-results.json').read_bytes());results=[]
    for zero in data['zeros']:
        if sha(zero['source'])!=zero['source_sha256']:raise ValueError('zero source changed')
    for case in data['cases']:
        d=Path(case['directory']);start=time.monotonic()
        try:
            run('verify',d)
            receipt=json.loads((d/'verified.json').read_bytes())
            # Dynamic pose test at maximum width for every source/scope.
            follow=deformation(d) if case['width']==2 else None
            results.append(dict(**case,ok=True,base=receipt['base'],
                max_displacement_error_m=receipt['outline']['max_displacement_error_m'],
                imported_culling=receipt['outline']['blender_import_backface_culling'],
                deformation=follow,verify_seconds=time.monotonic()-start))
        except Exception as exc:
            results.append(dict(**case,ok=False,error=str(exc),verify_seconds=time.monotonic()-start))
        (root/'verification-results.json').write_text(json.dumps(dict(zeros=data['zeros'],cases=results,
            csp_certified=False),indent=2))


if __name__=='__main__':
    stage,path=sys.argv[sys.argv.index('--')+1:];root=Path(path)
    {'export':export,'verify':verify}[stage](root)
