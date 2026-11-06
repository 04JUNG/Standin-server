"""Blender-only native source preparation and GT mesh extraction (split-aware)."""
import bpy, json, sys, traceback
from pathlib import Path

def main():
    cfg=json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text())
    sys.path.insert(0,cfg['repo'])
    from converter.convert import convert
    root=Path(cfg['out']);root.mkdir(parents=True,exist_ok=True)
    seeds=json.loads(Path(cfg['seeds']).read_text())
    split={r['case_id']:r['split'] for r in json.loads(Path(cfg['split']).read_text())['records']}
    records=[]
    for row in seeds:
        case=row['case_id']
        if row['status']!='ready' or split[case]=='test':continue
        try:
            dest=root/(case+'.base.fbx')
            if not dest.exists():
                result=convert(bvh_path=row['bvh'],character_fbx=cfg['body'],out_path=str(dest),output_mode='rigged_rest',embed_textures=False)
                if not result.ok:raise ValueError('native conversion rejected')
            bpy.ops.wm.read_factory_settings(use_empty=True)
            bpy.ops.import_scene.fbx(filepath=str(dest),ignore_leaf_bones=False)
            arm=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
            bones=[dict(name=b.name.split(':')[-1],parent=b.parent.name.split(':')[-1] if b.parent else None,matrix=[list(x) for x in arm.matrix_world@b.matrix],local_length=b.length) for b in arm.pose.bones]
            # Only development and validation meshes, never P6 test assets.
            bpy.ops.wm.read_factory_settings(use_empty=True)
            bpy.ops.import_scene.fbx(filepath=str(Path(cfg['data'])/'정답_3d인형'/(case+'.fbx')),ignore_leaf_bones=False)
            dg=bpy.context.evaluated_depsgraph_get();vertices=[];faces=[]
            for obj in [o for o in bpy.context.scene.objects if o.type=='MESH']:
                ev=obj.evaluated_get(dg);mesh=ev.to_mesh();mesh.calc_loop_triangles();offset=len(vertices)
                vertices.extend([list(obj.matrix_world@v.co) for v in mesh.vertices])
                faces.extend([[offset+i for i in t.vertices] for t in mesh.loop_triangles]);ev.to_mesh_clear()
            meshpath=root/(case+'.mesh.json');meshpath.write_text(json.dumps(dict(vertices=vertices,faces=faces)))
            records.append(dict(case_id=case,split=split[case],base_fbx=str(dest),bones=bones,gt_mesh=str(meshpath),status='ready'))
        except Exception as e:
            records.append(dict(case_id=case,split=split[case],status='error',error=str(e)));traceback.print_exc()
        (root/'records.json').write_text(json.dumps(records))
        print('V5_PREPARED',case,records[-1]['status'],flush=True)
if __name__=='__main__':main()
