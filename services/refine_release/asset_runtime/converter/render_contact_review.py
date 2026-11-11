"""Close-up render of the exact delivered meshes; colour marks the audited patch."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    import bpy
    from mathutils import Vector
    from converter.thumbnail_render import _stage_scene,_body_axes,_point_at,_configure_engine
    p=argparse.ArgumentParser()
    for k in ('before','after','contract','output-dir'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.output_dir.mkdir(parents=True,exist_ok=True)
    c=json.loads(a.contract.read_text())
    for label,path in [('before',a.before),('after',a.after)]:
        for view in ('front','three_quarter'):
            bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.fbx(filepath=str(path.resolve()))
            _stage_scene(view,720);scene=bpy.context.scene
            arm=next(o for o in scene.objects if o.type=='ARMATURE');mesh=next(o for o in scene.objects if o.type=='MESH')
            right,forward,up=_body_axes(arm)
            center=arm.matrix_world@arm.pose.bones['mixamorig:Hips'].head-up*.055+forward*.085
            axis=forward if view=='front' else (right+forward).normalized()
            scene.camera.location=center+axis*2;scene.camera.data.ortho_scale=.39;_point_at(scene.camera,center,up)
            mat=bpy.data.materials.new('Measured contact patch');mat.diffuse_color=(.95,.38,.02,1)
            mat.use_nodes=True;mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(.95,.38,.02,1)
            mesh.data.materials.append(mat);index=len(mesh.data.materials)-1
            # The delivered mesh is triangulated and the contact contract binds
            # its topology; only render material indices change, never vertices.
            from converter.surface_contact import topology_hash
            import numpy as np
            faces=np.array([list(p.vertices) for p in mesh.data.polygons])
            if topology_hash(faces)!=c['topology_sha256']:raise ValueError('render_contact_topology')
            for i in c['hand_faces']:mesh.data.polygons[i].material_index=index
            _configure_engine(scene,'CYCLES',16);scene.render.filepath=str(a.output_dir/(label+'-contact-'+view+'.png'));bpy.ops.render.render(write_still=True)
if __name__=='__main__':main()
