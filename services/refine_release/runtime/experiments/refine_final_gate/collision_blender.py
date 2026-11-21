"""Read-only collision measurements for submitted worlds, including hand/head."""
import sys,json,hashlib,bpy,bmesh
from pathlib import Path
import numpy as np
from mathutils.bvhtree import BVHTree
job=Path(sys.argv[sys.argv.index('--')+1]);request=json.loads(job.read_text());rows=[]
def groups(names):
 out={'head':[n for n in names if n.startswith('Head')],'neck':['Neck'],'chest':['Spine1','Spine2'],'pelvis':['Hips','Spine']}
 for s in ['Left','Right']:
  out.update({s+'_upperarm':[s+'Arm'],s+'_forearm':[s+'ForeArm'],s+'_hand':[n for n in names if n.startswith(s+'Hand')],s+'_thigh':[s+'UpLeg'],s+'_shin':[s+'Leg'],s+'_foot':[n for n in names if n.startswith(s+'Foot') or n.startswith(s+'Toe')]})
 return {k:[n for n in ns if n in names] for k,ns in out.items()}
def burial_measure(v,f,weights,names,patches,patch_faces):
 # A convex envelope alone overestimates concave interiors. Require an inward
 # nearest-surface normal as a second, independent condition on the actual mesh.
 result=[]
 targets={'torso':['Hips','Spine','Spine1','Spine2'],'head':[n for n in names if n.startswith('Head')]}
 for target,ns in targets.items():
  mass=weights[:,[names.index(n) for n in ns if n in names]].sum(1);faces=f[mass[f].min(1)>.55]
  if len(faces)<4:continue
  ids=np.unique(faces);bm=bmesh.new()
  for pos in v[ids]:bm.verts.new(pos)
  bmesh.ops.convex_hull(bm,input=list(bm.verts),use_existing_faces=False);bm.normal_update()
  center=v[ids].mean(0);planes=[]
  for face in bm.faces:
   normal=np.array(face.normal);point=np.array(face.verts[0].co)
   if np.dot(center-point,normal)>0:normal=-normal
   planes.append((normal,float(np.dot(point,normal))))
  bm.free()
  if len(planes)<4:continue
  normals=np.array([x[0] for x in planes]);offset=np.array([x[1] for x in planes]);tree=BVHTree.FromPolygons(v.tolist(),faces.tolist(),all_triangles=True)
  for moving in sorted(patches):
   if not moving.endswith(('_forearm','_hand')):continue
   mf=patch_faces[moving];mi=np.unique(mf);tri=v[mf];areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
   vertex_area=np.zeros(len(v));np.add.at(vertex_area,mf.ravel(),np.repeat(areas/3,3));inside=((v[mi]@normals.T-offset)<-1e-7).all(1);depth=np.zeros(len(v))
   for vi in mi[inside]:
    point,normal,_,distance=tree.find_nearest(v[vi].tolist())
    if point is not None and np.dot(v[vi]-np.array(point),np.array(normal))<0:depth[vi]=distance
   result.append(dict(moving_region=moving,target_region=target,vertex_depths=depth[mi].tolist(),vertex_area_fractions=(vertex_area[mi]/max(areas.sum(),1e-12)).tolist(),method='convex_envelope_and_actual_surface_normal'))
 return result

def measure(v,f,weights,names):
 patches={};patch_faces={};coverage=np.zeros(len(f),bool)
 for key,ns in groups(names).items():
  if not ns:continue
  mass=weights[:,[names.index(n) for n in ns]].sum(1);mask=mass[f].min(1)>.55;coverage|=mask
  if mask.any():
   patch_faces[key]=f[mask];patches[key]=BVHTree.FromPolygons(v.tolist(),f[mask].tolist(),all_triangles=True)
 adjacent={frozenset(x) for x in [('head','neck'),('neck','chest'),('chest','pelvis')]}
 for s in ['Left','Right']:
  for pair in [('chest',s+'_upperarm'),(s+'_upperarm',s+'_forearm'),(s+'_forearm',s+'_hand'),('pelvis',s+'_thigh'),(s+'_thigh',s+'_shin'),(s+'_shin',s+'_foot')]:adjacent.add(frozenset(pair))
 pairs={};witnesses=[];extent={};keys=sorted(patches)
 for i,a in enumerate(keys):
  for b in keys[i+1:]:
   if frozenset((a,b)) in adjacent:continue
   overlap=patches[a].overlap(patches[b]);pairs[a+'/'+b]=len(overlap)
   if not overlap:continue
   ext=dict(complete=True,inward_samples=0,max_depth=0.,max_area_fraction=0.)
   extent[a+'/'+b]=ext
   for moving,target,column in [(a,b,0),(b,a,1)]:
    ids=np.unique(patch_faces[moving][[t[column] for t in overlap]])
    tri=v[patch_faces[moving]];ar=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    ext['max_area_fraction']=max(ext['max_area_fraction'],float(ar[np.unique([t[column] for t in overlap])].sum()/max(ar.sum(),1e-12)))
    samples=[]
    for vi in ids:
     nearest=patches[target].find_nearest(v[int(vi)].tolist())
     if nearest[0] is None:ext['complete']=False;continue
     point,normal,fi,_=nearest;point=np.array(point);normal=np.array(normal);gap=float(np.dot(v[int(vi)]-point,normal))
     if gap>=-1e-7:continue
     ext['inward_samples']+=1;ext['max_depth']=max(ext['max_depth'],-gap)
     face=patch_faces[target][fi];tri=v[face];uv=np.linalg.lstsq(np.stack([tri[1]-tri[0],tri[2]-tri[0]],axis=1),point-tri[0],rcond=None)[0];bary=[float(1-uv.sum()),float(uv[0]),float(uv[1])]
     samples.append(dict(pair=a+'/'+b,moving_region=moving,target_region=target,vertex=int(vi),face=face.tolist(),bary=bary,initial_gap=gap))
    witnesses.extend(sorted(samples,key=lambda x:x['initial_gap'])[:16])
 return dict(burial=burial_measure(v,f,weights,names,patches,patch_faces),extent=extent,contact_witnesses=witnesses,pairs=pairs,intersections={k:n for k,n in pairs.items() if n},tested_patch_face_fraction=float(coverage.mean()),scope='nonadjacent weighted-interior region pairs; seams excluded',missing_required_groups=[k for k in ['head','Left_hand','Right_hand','chest','Left_thigh','Right_thigh'] if k not in patches])
for group in request['cases']:
 rig=json.loads(Path(group['rig']).read_text());names=[b['name'] for b in rig['bones']];base=np.array([b['matrix'] for b in rig['bones']]);mesh=np.load(group['mesh']);v,f,weights=mesh['vertices'],mesh['faces'],mesh['weights'];vh=np.c_[v,np.ones(len(v))];inv=np.linalg.inv(base)
 if group.get('calibration_output'):
  bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
  if hashlib.sha256(Path(group['source']).read_bytes()).hexdigest()!=group['source_sha256']:raise ValueError('Source FBX hash mismatch')
  bpy.ops.import_scene.fbx(filepath=group['source']);arm=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE');by={b.name.split(':')[-1]:b for b in arm.data.bones};pose={b.name.split(':')[-1]:b for b in arm.pose.bones}
  rest=np.array([np.asarray(arm.matrix_world@by[n].matrix_local) for n in names]);actual=np.array([np.asarray(arm.matrix_world@pose[n].matrix) for n in names]);error=float(abs(actual-base).max())
  if error>1e-5:raise ValueError('Source/rig calibration mismatch')
  Path(group['calibration_output']).write_text(json.dumps(dict(names=names,rest_world=rest.tolist(),source_sha256=group['source_sha256'],source_pose_error=error),indent=2))
  bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
 for candidate in group['candidates']:
  w=np.load(candidate['world'])['world'];vertices=np.einsum('vn,nvi->vi',weights,np.einsum('nij,vj->nvi',w@inv,vh)[:,:,:3]);row=dict(case=group['case'],candidate=candidate['id'],**measure(vertices,f,weights,names));rows.append(row)
 Path(request['output']).write_text(json.dumps(rows,indent=2));print('MEASURED',group['case'],len(group['candidates']),flush=True)
