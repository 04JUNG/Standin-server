"""Build new immutable shape sources from explicitly pinned approved FBXs.

Run in Blender. Original FBXs and the live authoring release are read-only.
Only BODY displacement fields are transferred; approved geometry, rig, weights
and PSDs are kept. The small preview mesh is derived from that same full mesh.
"""
import argparse, base64, hashlib, importlib.util, json, re, sys, zlib
from pathlib import Path
import bpy, bmesh, numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v): return json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False)
def digest(v): return hashlib.sha256(canonical(v).encode()).hexdigest()
def arr(data):
    v=np.empty(len(data)*3);data.foreach_get('co',v);return v.reshape(-1,3)

def projection(points, vertices, faces):
    tree=BVHTree.FromPolygons(vertices.tolist(),faces.tolist(),all_triangles=True)
    ids=np.empty((len(points),3),np.int32);weights=np.empty((len(points),3));dist=[]
    for i,p in enumerate(points):
        hit,_,fi,d=tree.find_nearest(Vector(p))
        tri=faces[fi];a,b,c=vertices[tri];v0=b-a;v1=c-a;v2=np.array(hit)-a
        d00=v0@v0;d01=v0@v1;d11=v1@v1;d20=v2@v0;d21=v2@v1
        denom=d00*d11-d01*d01
        if abs(denom)<1e-24: raise ValueError('degenerate projection triangle')
        v=(d11*d20-d01*d21)/denom;w=(d00*d21-d01*d20)/denom
        ids[i]=tri;weights[i]=(1-v-w,v,w);dist.append(d)
    return ids,weights,np.array(dist)

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main(args):
    root=Path(args.root);src=root/'active/sources';out=root/'authoring'/args.gender;out.mkdir(parents=True,exist_ok=True)
    male=args.gender=='male';old='standin-body-0-7-6' if male else 'standin-female-body-0-3-9'
    expected='f7c74512d8f1adac528a2fd8a93baacc794ddeb4d6d298c57cd5e6fa8df19f77' if male else 'ceee9f20f03e7cbd2df28f875ac130317fb84cecaa71f76fd012d4f3b0a65898'
    if sha(args.fbx)!=expected:raise ValueError('unapproved input FBX')
    release_files={x['path']:x for x in json.loads((root/'active/converter-manifest.json').read_text(encoding='utf-8'))['files']}
    old_rule='standin-body-0.7.6' if male else 'standin-female-body-0.3.9'
    for ext in ('.blend','.py'):
        if sha(src/(old+ext))!=release_files['sources/'+old_rule+ext]['sha256']:
            raise ValueError('active authoring source differs from its release manifest')
    bpy.ops.wm.open_mainfile(filepath=str(src/(old+'.blend')))
    original=module(str(src/(old+'.py')),'original_controller')
    m,a=original.objects();s=original.source();oldco=s['co'];oldfaces=s['faces'];deltas=s['deltas']
    oldhip=a.data.bones['mixamorig:Hips'].head_local.z
    original_text=(src/(old+'.py')).read_text(encoding='utf-8')
    restorer=module(args.restorer,'approved_restorer')
    restorer.restore(args.fbx,args.drivers,None)
    a=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
    m=max((o for o in bpy.context.scene.objects if o.type=='MESH'),key=lambda o:len(o.data.vertices))
    co=arr(m.data.shape_keys.key_blocks['Basis'].data)
    # Imported coordinates are already metric; preserve all geometry and rig frames.
    assert np.max(abs(np.array(m.matrix_world)-np.eye(4)))<1e-5
    scale=float(a.data.bones['mixamorig:Hips'].head_local.z/oldhip)
    ids,bary,distance=projection(co/scale,oldco,oldfaces)
    groups={g.index:g.name for g in m.vertex_groups}
    protected=np.zeros(len(co))
    for vertex in m.data.vertices:
        protected[vertex.index]=sum(g.weight for g in vertex.groups if any(s in groups[g.group] for s in ('Hand','Foot','Toe','Head')))
    mask=1-original.smooth(.05,.65,protected)
    for name,delta in deltas.items():
        k=m.shape_key_add(name=name)
        transfer=np.einsum('ni,nij->nj',bary,delta[ids])*scale*mask[:,None]
        k.data.foreach_set('co',(co+transfer).ravel());k.value=0
    for name in ('SYSTEM_Proportions','AUTO_Pose','AUTO_Contact','AUTO_Anatomy','AUTO_Surface','AUTO_JointSafety'):
        if name not in m.data.shape_keys.key_blocks:m.shape_key_add(name=name)
    # Triangulation changes only polygon indexing, not vertices or shape-key data.
    bm=bmesh.new();bm.from_mesh(m.data);bmesh.ops.triangulate(bm,faces=list(bm.faces));bm.to_mesh(m.data);bm.free()
    m['standin_studio_mesh']=True;a['standin_studio_arm']=True
    rest={b.name:{'head':list(b.head_local),'tail':list(b.tail_local),'roll_axis':list(b.matrix_local.to_3x3().col[2])} for b in a.data.bones}
    a['frozen_rest']=canonical(rest)
    chin=1.482507467 if male else 1.494518995
    candidates=np.flatnonzero((abs(co[:,0])<.018)&(co[:,1]<0))
    ci=int(candidates[np.argmin(abs(co[candidates,2]-chin))])
    crown=int(np.argmax(co[:,2]));floor=float(co[:,2].min());top=float(co[crown,2]);chin=float(co[ci,2])
    shoulder=float(a.data.bones['mixamorig:LeftArm'].head_local.z)
    lm={'floor':floor,'top':top,'chin':chin,'chin_vertex':ci,'crown_vertex':crown,'head_y':float(a.data.bones['mixamorig:Head'].head_local.y),
        'ankle':float(a.data.bones['mixamorig:LeftFoot'].head_local.z),'knee':float(a.data.bones['mixamorig:LeftLeg'].head_local.z),
        'hip':float(a.data.bones['mixamorig:LeftUpLeg'].head_local.z),'shoulder':shoulder}
    m['landmarks']=canonical(lm);m['base_heads']=(top-floor)/(top-chin)
    m['asset_identity']='standin-approved-'+args.gender+'-'+expected[:20]
    keys=m.data.shape_keys.key_blocks;co=arr(keys['Basis'].data)
    ds={k.name:arr(k.data)-co for k in keys if k.name.startswith('BODY_')}
    faces=np.array([p.vertices[:] for p in m.data.polygons],np.int32)
    lock={'basis_sha256':hashlib.sha256(co.astype('<f4').tobytes()).hexdigest(),
          'topology_sha256':digest(faces.tolist()),'rig_sha256':digest(rest),
          'body_deltas_sha256':hashlib.sha256(b''.join(n.encode()+ds[n].astype('<f4').tobytes() for n in sorted(ds))).hexdigest(),
          'weights_sha256':digest([[(groups[g.group],float(g.weight)) for g in v.groups] for v in m.data.vertices])}
    m['asset_lock']=canonical(lock)
    # Native source remains full resolution. Only the UI preview uses this LOD.
    lod=m.copy();lod.data=m.data.copy();lod.animation_data_clear();bpy.context.scene.collection.objects.link(lod)
    lod.name='Standin_Preview_LOD';lod.shape_key_clear();lod.modifiers.clear();lod.parent=None
    for k in list(lod.keys()):del lod[k]
    lod['standin_preview_lod']=True
    bpy.ops.object.select_all(action='DESELECT');lod.select_set(True);bpy.context.view_layer.objects.active=lod
    dec=lod.modifiers.new('Preview only','DECIMATE');dec.ratio=min(1.,56000/len(co));dec.use_collapse_triangulate=True
    bpy.ops.object.modifier_apply(modifier=dec.name)
    lp=arr(lod.data.vertices);li,lw,ld=projection(lp,co,faces)
    if len(lp)>65535 or np.percentile(ld,99.9)>.0005:raise ValueError('preview LOD quality gate failed')
    # Snap preview to actual source surface so all controls share the same projection.
    lod.data.vertices.foreach_set('co',np.einsum('ni,nij->nj',lw,co[li]).ravel())
    lod['projection']=base64.b64encode(zlib.compress(li.astype('<i4').tobytes()+lw.astype('<f4').tobytes())).decode()
    lod.hide_render=True;lod.hide_set(True)
    version='standin-body-0.8.0' if male else 'standin-female-body-0.4.0'
    text=re.sub(r"^VERSION\s*=.*$",'VERSION='+repr(version),original_text,flags=re.M)
    # The inherited controller's field class gets the chosen anatomy defaults.
    height=round((top-floor)*100,6)
    text=re.sub(r"('height_cm':\([^,]+,[^,]+,)[^,]+,",lambda x:x[1]+str(height)+',',text,count=1)
    text+='\n'+Path(__file__).with_name('controller_tail.py').read_text(encoding='utf-8')
    text+='\nPRESETS={"BASE":("기본형",{})}\n'
    controller=out/'standin_body.py';controller.write_text(text,encoding='utf-8')
    # No embedded scripts need to execute when the file is opened.
    for t in list(bpy.data.texts):bpy.data.texts.remove(t)
    bpy.context.preferences.filepaths.file_preview_type='NONE';bpy.context.preferences.filepaths.save_version=0
    path=out/'model.blend';bpy.ops.wm.save_as_mainfile(filepath=str(path))
    report={'gender':args.gender,'source_fbx_sha256':expected,'vertices':len(co),'bones':len(a.data.bones),'triangles':len(faces),
        'body_controls':len(original.FIELDS),'preview_vertices':len(lp),'preview_projection_p999_mm':float(np.percentile(ld,99.9)*1000),
        'transfer_distance_p95_mm':float(np.percentile(distance[mask>.5],95)*1000),'landmarks':lm,'base_heads':float(m['base_heads']),
        'blend_sha256':sha(path),'controller_sha256':sha(controller),'rules':version}
    (out/'build.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--gender',choices=['male','female'],required=True)
    ap.add_argument('--fbx',required=True);ap.add_argument('--drivers',required=True);ap.add_argument('--restorer',required=True)
    main(ap.parse_args(sys.argv[sys.argv.index('--')+1:]))
