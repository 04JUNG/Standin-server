"""Protected vertex-subset LOD. Run in Blender; never writes to an input.

Limited planar dissolution is constrained to low-deformation interiors. A frozen
integer correspondence transfers EVERY shape key and skin weight exactly. This
keeps protected face connectivity and source normals, unlike uniform collapse.
"""
import argparse,hashlib,importlib.util,json,math,re,sys
from pathlib import Path
import bpy,bmesh,numpy as np
from mathutils import Matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.character_lod.contract import APPROVED,require_hash,require_subset
from mathutils import Vector

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def arr(d):
 a=np.empty(len(d)*3);d.foreach_get('co',a);return a.reshape(-1,3)
def module(path):
 spec=importlib.util.spec_from_file_location('lod_controller',path);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);return c

def protection(m,a,co,hand_policy):
 names={g.index:g.name for g in m.vertex_groups};mask=np.zeros(len(co),bool)
 tokens=('Hand','Foot','Toe','Head','Neck') if hand_policy=='frozen' else ('Foot','Toe','Head','Neck')
 for v in m.data.vertices:
  mask[v.index]=sum(g.weight for g in v.groups if any(t in names[g.group] for t in tokens))>.2
 if hand_policy=='articulated':
  for v in m.data.vertices:
   if any('Hand' in names[g.group] and names[g.group].endswith('3') and g.weight>.05 for g in v.groups):mask[v.index]=True
  for bone in a.data.bones:
   if 'Hand' in bone.name and any(x in bone.name for x in ('Thumb','Index','Middle','Ring','Pinky')):
    mask|=np.linalg.norm(co-np.array(bone.head_local),axis=1)<.010
 # Protect full joint neighbourhoods; region is measured in native metric space.
 for side in ('Left','Right'):
  for name,radius in [('Arm',.070),('ForeArm',.055),('Hand',.045),('UpLeg',.085),('Leg',.065),('Foot',.050)]:
   center=np.array(a.data.bones['mixamorig:'+side+name].head_local);mask|=np.linalg.norm(co-center,axis=1)<radius
 # Every PSD is transferred by the same frozen correspondence; joint cores are protected.
 lm=json.loads(m['landmarks']);mask[[lm['chin_vertex'],lm['crown_vertex']]]=True
 # One-ring halo ensures protected triangles remain exactly connected.
 edges=np.array([e.vertices[:] for e in m.data.edges]);p=mask.copy()
 mask[edges[np.any(p[edges],axis=1)].ravel()]=True
 return mask

def local_topology(m,protected,rounds,source_uv):
 # Dissolve isolated vertex stars, never a growing coplanar region. A UV seam
 # and one-ring protected boundary are immutable. Every authored shape key is
 # tested for local plane departure before a point can be removed.
 bm=bmesh.new();bm.from_mesh(m.data);layer=bm.verts.layers.int.new('source_index');uv_layer=bm.loops.layers.uv.new('pinned_skin_uv')
 for v in bm.verts:v[layer]=v.index+1
 for f,values in zip(bm.faces,source_uv.reshape(-1,3,2)):
  for loop,uv in zip(f.loops,values):loop[uv_layer].uv=uv
 fields=[arr(k.data) for k in m.data.shape_keys.key_blocks if k.name=='Basis' or k.name.startswith(('BODY_','PSD_'))]
 for iteration in range(int(rounds)):
  bm.verts.ensure_lookup_table();bm.normal_update();chosen=[];blocked=set()
  for v in bm.verts:
   sid=v[layer]-1
   if protected[sid] or v in blocked or not v.is_manifold:continue
   ring=[e.other_vert(v) for e in v.link_edges]
   if len(ring)<4 or len(ring)>8 or any(protected[u[layer]-1] for u in ring):continue
   # Seam detection uses actual per-corner UVs on all incident faces.
   if any(len({tuple(round(x,6) for x in l[uv_layer].uv) for l in u.link_loops})>1 for u in [v,*ring]):continue
   ri=np.array([u[layer]-1 for u in ring]);normal=np.array(v.normal);safe=True
   for field in fields:
    points=field[ri];center=points.mean(0);_,_,vt=np.linalg.svd(points-center,full_matrices=False)
    if abs((field[sid]-center)@vt[-1])>.00008:safe=False;break
    # Avoid a single long replacement diagonal on narrow body features.
    if np.linalg.norm(points-field[sid],axis=1).max()>.018:safe=False;break
   if safe:
    chosen.append(v);blocked.update(ring);blocked.add(v)
    for u in ring:blocked.update(e.other_vert(u) for e in u.link_edges)
  bmesh.ops.dissolve_verts(bm,verts=chosen,use_face_split=False,use_boundary_tear=False)
  bmesh.ops.triangulate(bm,faces=[f for f in bm.faces if len(f.verts)>3],quad_method='BEAUTY',ngon_method='BEAUTY')
  print('LOCAL_ROUND',iteration+1,'removed',len(chosen),flush=True)
 bm.verts.index_update();ids=np.array([v[layer]-1 for v in bm.verts]);faces=np.array([[v.index for v in f.verts] for f in bm.faces],np.int32);uv=np.array([[l[uv_layer].uv[:] for l in f.loops] for f in bm.faces]);bm.free()
 original={tuple(sorted(p.vertices[:])) for p in m.data.polygons if np.any(protected[list(p.vertices)])};output={tuple(sorted(ids[f])) for f in faces}
 require_subset(ids,np.flatnonzero(protected),[p.vertices[:] for p in m.data.polygons],faces)
 return ids,faces,uv

def topology(m,protected,angle,source_uv):
 bm=bmesh.new();bm.from_mesh(m.data);layer=bm.verts.layers.int.new('source_index')
 for v in bm.verts:v[layer]=v.index+1
 uv_layer=bm.loops.layers.uv.new('pinned_skin_uv')
 for face,values in zip(bm.faces,source_uv.reshape(-1,3,2)):
  for loop,uv in zip(face.loops,values):loop[uv_layer].uv=uv
 verts=[v for v in bm.verts if all(not protected[u.index] for f in v.link_faces for u in f.verts)]
 edges=[e for e in bm.edges if all(not protected[v.index] for f in e.link_faces for v in f.verts)]
 bmesh.ops.dissolve_limit(bm,angle_limit=math.radians(angle),use_dissolve_boundaries=False,verts=verts,edges=edges,delimit={'NORMAL','MATERIAL','SEAM','SHARP','UV'})
 bmesh.ops.triangulate(bm,faces=list(bm.faces));bm.verts.index_update()
 ids=np.array([v[layer]-1 for v in bm.verts]);faces=np.array([[v.index for v in f.verts] for f in bm.faces],np.int32);uv=np.array([[loop[uv_layer].uv[:] for loop in f.loops] for f in bm.faces]);bm.free()
 assert len(set(ids))==len(ids) and set(np.flatnonzero(protected))<=set(ids)
 original={tuple(sorted(p.vertices[:])) for p in m.data.polygons if np.any(protected[list(p.vertices)])}
 output={tuple(sorted(ids[f])) for f in faces};require_subset(ids,np.flatnonzero(protected),[p.vertices[:] for p in m.data.polygons],faces)
 return ids,faces,uv

def copy_drivers(source,target):
 if not source.animation_data:return
 for f in source.animation_data.drivers:
  g=target.driver_add(f.data_path);g.driver.type=f.driver.type;g.driver.expression=f.driver.expression
  for v in f.driver.variables:
   w=g.driver.variables.new();w.name=v.name;w.type=v.type
   for src,dst in zip(v.targets,w.targets):

    if v.type in ('SINGLE_PROP','CONTEXT_PROP'):dst.id_type=src.id_type
    dst.id=src.id;dst.data_path=src.data_path
    if v.type=='TRANSFORMS':
     for k in ('bone_target','transform_type','transform_space','rotation_mode'):setattr(dst,k,getattr(src,k))

def main(args):
 require_hash(args.blend,APPROVED[args.gender]['blend']);require_hash(args.controller,APPROVED[args.gender]['controller'])
 skin_manifest=json.loads((Path(__file__).resolve().parents[2]/'converter/skin_assets.v1.json').read_text());cid='male-base-v0-8-0' if args.gender=='male' else 'female-base-v0-4-0'
 require_hash(args.skin_uv,skin_manifest['models'][cid]['uv']['sha256'])
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
 bpy.ops.wm.open_mainfile(filepath=args.blend);c=module(args.controller);m,a=c.objects();co=arr(m.data.shape_keys.key_blocks['Basis'].data)
 for bone in a.pose.bones:bone.matrix_basis=Matrix.Identity(4)
 a.update_tag();bpy.context.view_layer.update()
 reference_properties={k:m[k] for k in m.keys()}
 import gzip
 skin=json.loads(gzip.decompress(Path(args.skin_uv).read_bytes()));assert skin['triangles']==[list(p.vertices) for p in m.data.polygons]
 source_uv=np.array(skin['uv']);protected=protection(m,a,co,args.hand_policy)
 extra_fields=[];full_reference=None
 if args.method=='surface':
  from scripts.character_lod.surface_topology import mask,simplify
  from scripts.character_lod.reference_compute import payload
  protected=mask(m,a,co,source_uv,np.array(skin['triangles'],np.int32))
  if not args.registry:raise ValueError('surface method requires the pinned recipe registry')
  registry=json.loads(Path(args.registry).read_text());recipes=[v for k,v in registry['characters'].items() if k.startswith(args.gender+'-') and v['rules']==c.VERSION]
  for recipe in recipes:
   values=c.normalize(recipe['defaults']);extra_fields.append(c.compute(values)[0]);c.validate_geometry(values)
  base_values=recipes[0]['defaults'];custom={**base_values,'height_cm':190.,'heads':7.,'physique':1.2,'muscle':1.8,'shoulders':.6,'leg_length':.55,'arm_length':.3}
  extra_fields.append(c.compute(c.normalize(custom))[0]);c.validate_geometry(custom)
  full_reference=payload(c.source(),reference_properties)
 if args.extra_protected:protected[np.load(args.extra_protected)]=True
 trials=[];np.save(out/'source-co.npy',co)
 topo=local_topology if args.method=='local' else topology
 if args.method=='surface':
  topo=lambda mesh,p,level,uv: simplify(mesh,p,args.rounds,uv,level/1000,args.span_mm/1000,extra_fields)[:3]
 levels=(args.tolerance_mm if args.method=='surface' else ((1.,3.) if args.method=='local' else (.25,1.,3.,7.,15.)))
 cache={}
 for angle in levels:
  ids,faces,uv=topo(m,protected,angle,source_uv);cache[angle]=(ids,faces,uv);trials.append(dict(level=angle,level_unit='mm' if args.method=='surface' else ('rounds' if args.method=='local' else 'degrees'),vertices=len(ids),triangles=len(faces),retained=len(ids)/len(co)))
  print('TRIAL',trials[-1],flush=True)
 (out/'trials.json').write_text(json.dumps({'protected_vertices':int(protected.sum()),'trials':trials},indent=2))
 # Emit distinct quality levels, even when protected density prevents 75%/50%.
 for angle in (levels if args.method=='surface' else ((1.,3.) if args.method=='local' else (1.,7.))):
  ids,faces,uv=cache[angle];tag=args.gender+'-protected-'+str(int(angle))+('-articulated' if args.hand_policy=='articulated' else '')+('-local' if args.method=='local' else '')
  if args.method=='surface':tag=args.gender+'-surface-'+str(round(angle*1000))+'um-ref-v3'
  folder=out/tag;folder.mkdir()
  n=m.copy();n.data=bpy.data.meshes.new(tag);bpy.context.collection.objects.link(n);n.name=tag
  n.data.from_pydata(co[ids].tolist(),[],faces.tolist());n.data.update()
  for material in m.data.materials:n.data.materials.append(material)
  n.vertex_groups.clear()
  for g in m.vertex_groups:n.vertex_groups.new(name=g.name)
  for v,src in zip(n.data.vertices,ids):
   for g in m.data.vertices[int(src)].groups:n.vertex_groups[g.group].add([v.index],g.weight,'REPLACE')
  for key in m.data.shape_keys.key_blocks:
   k=n.shape_key_add(name=key.name);k.data.foreach_set('co',arr(key.data)[ids].ravel());k.value=key.value;k.slider_min=key.slider_min;k.slider_max=key.slider_max;k.mute=key.mute
  copy_drivers(m.data.shape_keys,n.data.shape_keys)
  for p in n.data.polygons:p.use_smooth=True
  normals=np.array([v.normal[:] for v in m.data.vertices]);n.data.normals_split_custom_set_from_vertices(normals[ids].tolist())
  # Authoring sources have no UV; FBX skin UVs are repackaged separately by ID.
  if m.data.uv_layers:raise ValueError('UV-bearing source needs explicit corner transfer')
  lm=json.loads(m['landmarks']);inverse={int(src):i for i,src in enumerate(ids)}
  for k in ('chin_vertex','crown_vertex'):lm[k]=inverse[lm[k]]
  n['landmarks']=canonical(lm);n['asset_identity']='standin-lod-'+tag+'-'+sha(args.blend)[:16]
  groups={g.index:g.name for g in n.vertex_groups};base=arr(n.data.shape_keys.key_blocks['Basis'].data);ds={k.name:arr(k.data)-base for k in n.data.shape_keys.key_blocks if k.name.startswith('BODY_')}
  lock={'basis_sha256':hashlib.sha256(base.astype('<f4').tobytes()).hexdigest(),'topology_sha256':digest(faces.tolist()),'rig_sha256':digest(json.loads(a['frozen_rest'])),'body_deltas_sha256':hashlib.sha256(b''.join(k.encode()+ds[k].astype('<f4').tobytes() for k in sorted(ds))).hexdigest(),'weights_sha256':digest([[(groups[g.group],float(g.weight)) for g in v.groups] for v in n.data.vertices])}
  n['asset_lock']=canonical(lock);n['standin_studio_mesh']=True;m['standin_studio_mesh']=False
  if full_reference:
   import base64
   n['lod_reference_graph']=full_reference;n['lod_reference_graph_sha256']=hashlib.sha256(base64.b64decode(full_reference)).hexdigest()
   n['lod_source_indices']=base64.b64encode(ids.astype('<i4').tobytes()).decode()
   n['neutral_hand_pose']='source rest; no finger override'
  # Fresh UI-only LOD and fresh barycentric mapping to THIS candidate. SSR2
  # uses uint16 faces, so a full candidate cannot be the preview mesh.
  preview=n.copy();preview.data=n.data.copy();bpy.context.collection.objects.link(preview);preview.shape_key_clear();preview.modifiers.clear();preview.parent=None
  for k in list(preview.keys()):del preview[k]
  bpy.ops.object.select_all(action='DESELECT');preview.select_set(True);bpy.context.view_layer.objects.active=preview
  dec=preview.modifiers.new('Candidate UI preview only','DECIMATE');dec.ratio=min(1.,50000/len(ids));dec.use_collapse_triangulate=True
  bpy.ops.object.modifier_apply(modifier=dec.name)
  sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
  from scripts.character_upgrade.build_source import projection
  pi,pw,projection_distance=projection(arr(preview.data.vertices),base,faces)
  preview.data.vertices.foreach_set('co',np.einsum('ni,nij->nj',pw,base[pi]).ravel())
  assert len(preview.data.vertices)<65536
  import base64,zlib
  preview['standin_preview_lod']=True;preview.hide_render=True;preview.hide_set(True)
  preview['projection']=base64.b64encode(zlib.compress(pi.astype('<i4').tobytes()+pw.astype('<f4').tobytes())).decode()
  hidden=[]
  for o in list(bpy.context.scene.objects):
   if o not in (n,a,preview):hidden.append(o);bpy.context.scene.collection.objects.unlink(o)
  version='standin-lod-'+tag
  text=re.sub(r'^VERSION\s*=.*$', 'VERSION='+repr(version),Path(args.controller).read_text(),flags=re.M)
  if full_reference:
   from scripts.character_lod.reference_compute import controller_tail
   text+=controller_tail(Path(args.controller).read_text())
  else:
   # Preserve the approved smoothing field at retained source points. Re-smoothing
   # a different graph changes large physique controls even with exact morph keys.
   c.CACHE.clear();m['standin_studio_mesh']=True;n['standin_studio_mesh']=False
   bpy.context.scene.collection.objects.link(m)
   original=c.source();smooth={k:v[ids] for k,v in original.get('fat_smooth',{}).items()}
   bpy.context.scene.collection.objects.unlink(m);m['standin_studio_mesh']=False;n['standin_studio_mesh']=True
   text+='\n# Offline source-topology smoothing fields, resampled by immutable subset.\n'
   for name,data in smooth.items():
    raw=base64.b64encode(zlib.compress(data.astype('<f8').tobytes())).decode()
    text+=f'_LOD_{name} = {raw!r}\n'
   text+='\n_LOD_PARENT_SOURCE = source\ndef source(scene=None):\n    s = _LOD_PARENT_SOURCE(scene)\n'
   for name in smooth:text+=f"    s['fat_smooth'][{name!r}] = np.frombuffer(zlib.decompress(base64.b64decode(_LOD_{name})),dtype='<f8').reshape(-1,3)\n"
   text+='    return s\n'
  (folder/'controller.py').write_text(text);np.save(folder/'skin-uv.npy',uv);np.save(folder/'source-indices.npy',ids);np.save(folder/'protected-source.npy',np.flatnonzero(protected))
  bpy.context.preferences.filepaths.save_version=0;bpy.data.libraries.write(str(folder/'model.blend'),{bpy.context.scene},fake_user=False,compress=True)
  manifest={'schema':'protected-body-lod.v1','id':tag,'rules':version,'gender':args.gender,'source_blend_sha256':sha(args.blend),'source_controller_sha256':sha(args.controller),'extra_protected_sha256':sha(args.extra_protected) if args.extra_protected else None,'source_vertices':len(co),'vertices':len(ids),'triangles':len(faces),'protected_vertices':int(protected.sum()),'protected_source_indices_and_connectivity_exact':True,'deformed_coordinates_exact':False,'correspondence':'source-vertex-subset','method':args.method,'rounds_or_angle':angle,'surface_rounds':args.rounds if args.method=='surface' else None,'exact_source_control_graph':bool(full_reference),'source_control_math_remains_full_resolution':bool(full_reference),'neutral_fingers':'identity matrix_basis','hand_policy':args.hand_policy,'preview_vertices':len(preview.data.vertices),'preview_projection_p999_mm':float(np.percentile(projection_distance,99.9)*1000),'bones':len(a.data.bones),'body_shape_keys':len(ds),'total_shape_keys':len(n.data.shape_keys.key_blocks),'controls':len(c.FIELDS),'psd_keys':len([k for k in n.data.shape_keys.key_blocks if k.name.startswith('PSD_')]),'uv_source':'approved FBX corner UVs; seam-delimited transfer; fresh candidate skin hashes','promotion':'candidate-only; human signoff and skin certification required','files':{p.name:sha(p) for p in folder.iterdir() if p.is_file()}}
  (folder/'manifest.json').write_text(json.dumps(manifest,indent=2));print('BUILT',tag,len(ids),flush=True)
  bpy.data.objects.remove(n,do_unlink=True);bpy.data.objects.remove(preview,do_unlink=True)
  for o in hidden:bpy.context.scene.collection.objects.link(o)
  m['standin_studio_mesh']=True

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['planar','local','surface'],default='local');ap.add_argument('--registry');ap.add_argument('--rounds',type=int,default=18);ap.add_argument('--tolerance-mm',type=float,nargs='+',default=[.15,.3]);ap.add_argument('--span-mm',type=float,default=30.);ap.add_argument('--extra-protected');ap.add_argument('--blend',required=True);ap.add_argument('--skin-uv',required=True);ap.add_argument('--controller',required=True);ap.add_argument('--gender',required=True);ap.add_argument('--hand-policy',choices=['frozen','articulated'],default='frozen');ap.add_argument('--out',required=True);main(ap.parse_args(sys.argv[sys.argv.index('--')+1:]))
