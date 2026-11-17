"""Plan a frozen-hand male body topology in pinned Blender, before build.py.

Inputs are the approved original, v4 frozen graph seed and original quality
pose cache. Output is a source-triangle barycentric plan; never a final-FBX edit.
"""
import argparse,sys,json,gzip,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import bpy
from scripts.character_lod.progressive_body import hybrid_seed,collapse,SourceSurfaceProjector,read_seed
from scripts.character_lod.quality import dist,load
from scripts.character_lod.contract import require_hash
from scripts.character_lod.region_policy import hand_graph_mask
from scripts.character_lod.correspondence import Correspondence
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seed',required=True);p.add_argument('--original-quality',required=True);p.add_argument('--originals',required=True);p.add_argument('--skin-uv',required=True);p.add_argument('--converter-fields',required=True);p.add_argument('--target',type=int,default=93000);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);r=Path(a.out);r.mkdir(parents=True,exist_ok=False);seed=Path(a.seed);oldref=Path(a.original_quality);original=Path(a.originals)
if not 0<a.target<93038:raise ValueError('target must be below 93038')
from scripts.character_lod.contract import APPROVED
for key in ('blend','controller'):
 name='standin-body-0-8-0.'+('blend' if key=='blend' else 'py')
 require_hash(original/name,APPROVED['male'][key])
c=load(original/'standin-body-0-8-0.blend',original/'standin-body-0-8-0.py');m,arm=c.objects();co=c.arr(m.data.shape_keys.key_blocks['Basis'].data);oldfaces=np.array([p.vertices[:] for p in m.data.polygons],np.int32);skin=json.loads(gzip.decompress(Path(a.skin_uv).read_bytes()));require_hash(seed/'source-regions.npy',json.loads((seed/'manifest.json').read_text())['files']['source-regions.npy']);labels=np.load(seed/'source-regions.npy');seed_ids,seed_faces,seed_uv,seed_meta=read_seed(seed);ids,faces,uv=hybrid_seed(oldfaces,np.asarray(skin['uv']),seed_ids,seed_faces,seed_uv,labels);np.save(r/'original-basis.npy',co)
body=np.isin(labels[ids],[3,4,5]);original_hand=hand_graph_mask(c.source()['names'],c.source()['W']);graph=(~body)|original_hand[ids];pinned=graph.copy();pinned[np.unique(faces[graph[faces].any(1)])]=True
lo=np.full((len(ids),2),np.inf);hi=np.full((len(ids),2),-np.inf);np.minimum.at(lo,faces.ravel(),uv.reshape(-1,2));np.maximum.at(hi,faces.ravel(),uv.reshape(-1,2));seam=(hi-lo>1e-6).any(1);pinned|=seam
for side in ['Left','Right']:
 for name in ['Arm','ForeArm','Hand','UpLeg','Leg','Foot']:
  center=np.array(arm.data.bones['mixamorig:'+side+name].head_local);distance=np.linalg.norm(co[ids]-center,axis=1);pinned|=distance<.01;pinned[np.argsort(distance)[:8]]=True
cases=json.loads((oldref/'quality.json').read_text())['source_cases'];arrays=[];full_arrays=[];names=[];static_guards=[];geometry_pairs=[]
for case,record in cases.items():
 out,_,ref=c.compute(c.normalize(record['controls']));static_guards.append((out,ref))
 for pose in record['poses']:
  frame_co=np.load(oldref/(case+'-'+pose+'.npz'))['co'];full_arrays.append(frame_co);arrays.append(frame_co[ids]);names.append(case+'/'+pose)
 full_arrays.append(out);arrays.append(out[ids]);names.append(case+'/geometry-output');ia=len(arrays)-1
 full_arrays.append(ref);arrays.append(ref[ids]);names.append(case+'/geometry-reference');geometry_pairs.append((ia,len(arrays)-1))
actual_manifest=json.loads((Path(a.converter_fields)/'manifest.json').read_text())
if sorted(row['case'] for row in actual_manifest)!=['base','custom','preset']:raise ValueError('three original Converter fields required')
for actual_case in ('base','preset','custom'):
 require_hash(Path(a.converter_fields)/(actual_case+'.npy'),next(row['field_sha256'] for row in actual_manifest if row['case']==actual_case))
 actual_co=np.load(Path(a.converter_fields)/(actual_case+'.npy'),allow_pickle=False)
 if actual_co.shape!=co.shape or not np.isfinite(actual_co).all():raise ValueError('invalid actual Converter source field')
 full_arrays.append(actual_co);arrays.append(actual_co[ids]);names.append('actual-converter/'+actual_case)
arrays.append(co[ids]);full_arrays.append(co);names.append('basis-guard');fields=np.stack(arrays);projector=SourceSurfaceProjector(co,oldfaces,np.asarray(skin['uv']),labels,original_hand,np.array([v.normal[:] for v in m.data.vertices]),np.stack(full_arrays));(r/'progressive-config.json').write_text(json.dumps({'seed':str(seed),'initial_vertices':len(ids),'initial_body':int(body.sum()),'pinned_body':int((pinned&body).sum()),'full_hand_graph_vertices':int(original_hand.sum()),'body_seam_points':int((seam&body).sum()),'fields':names,'target':a.target,'placement':'surface','normalize_cost':.5,'max_edge_m':.08,'actual_converter_frame_weight':1.15,'surface_cost_ratio':1.02,'bulk_fat_frame_weight':1.05},indent=2)+'\n');print('PROGRESSIVE_START',len(ids),'body',body.sum(),'body pins',(pinned&body).sum(),'fields',len(fields),flush=True)
report=[]
def checkpoint(count,data):
 mapping=Correspondence(data['indices'],data['weights']);selected=mapping.indices[:,0];newfaces=data['faces'];newuv=data['uv'];pins=data['protected'];frozen=data['graph'];folder=r/('endpoint-'+str(count));folder.mkdir(exist_ok=False);np.savez_compressed(folder/'topology.npz',**data);stats=[];inverse=np.isin(labels,[3,4,5]);orientations=[]
 for i,(out,ref) in enumerate(static_guards):
  a=mapping.apply(out)[newfaces];b=mapping.apply(ref)[newfaces];na=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);nb=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0]);orientations.append(int(((na*nb).sum(1)<=0).sum()))
 for case in ['male-base-v0-8-0','custom-large']:
  for pose in ['neutral','raised','elbow','deep','sit','cross','hands']:
   original_co=np.load(oldref/(case+'-'+pose+'.npz'))['co'];metric=dist(original_co[inverse],mapping.apply(original_co),newfaces);stats.append({'case':case,'pose':pose,**metric})
 row={'vertices':count,'body_vertices':int(np.isin(labels[selected],[3,4,5]).sum()),'static_reversed_faces_by_case':orientations,'body_surface_worst_p99_mm':max(x['p99_mm'] for x in stats),'poses':stats};report.append(row);(r/'progressive-probes.json').write_text(json.dumps(report,indent=2)+'\n');print('PROGRESSIVE_PROBE',count,'worst body p99',row['body_surface_worst_p99_mm'],'flips',orientations,flush=True)
outputs,stats=collapse(ids,faces,uv,fields,pinned,graph,a.target,checkpoints=[n for n in (100000,95000) if n>a.target],max_edge=.08,on_checkpoint=checkpoint,geometry_pairs=geometry_pairs,placement="surface",normalize_cost=.5,cost_frames=[i for i,name in enumerate(names) if "/geometry-" not in name and name!="basis-guard"],projector=projector,frame_weights=[1.15 if name.startswith("actual-converter/") else 1.05 if (name.startswith("male-bulk-") or name.startswith("male-fat-max-")) else 1. for name in names]);(r/'progressive-final.json').write_text(json.dumps(stats,indent=2)+'\n')

if stats['vertices']!=a.target:raise RuntimeError('topology guards prevented the requested target')
