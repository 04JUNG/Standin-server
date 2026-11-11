"""Export approved-source presets, round-trip FBXs, and render hash-bound thumbnails."""
import argparse,hashlib,importlib.util,json,shutil,sys
from pathlib import Path
import bpy,numpy as np
from mathutils import Matrix,Vector
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.build_shape_release import BASIC,grid_control,label_for
from converter.preview_style import configure_workbench

def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--approved',required=True);ap.add_argument('--out',required=True);ap.add_argument('--gender',required=True)
args=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);root=Path(args.out);sex=args.gender;folder=root/sex;source=Path(args.approved)/'authoring'/sex
probe=json.loads((folder/'probe.json').read_text(encoding='utf-8'));release=folder/'release';release.mkdir(exist_ok=True)
for d in ('fbx','png','qa'):(release/d).mkdir(exist_ok=True)
bodies=[];recipes={};characters={};verification={}
for index,(preset,entry) in enumerate(probe['presets'].items()):
    bpy.ops.wm.open_mainfile(filepath=str(source/'model.blend'))
    spec=importlib.util.spec_from_file_location('build_presets_'+sex,source/'standin_body.py');sb=importlib.util.module_from_spec(spec);spec.loader.exec_module(sb);sb.register()
    m,rig=sb.objects();values=entry['controls'];sb.set_controls(values,correct=False)
    for b in rig.pose.bones:b.matrix_basis=Matrix.Identity(4)
    exported=sb.export_fixed(str(folder/'exports'));meta=json.loads((exported/'manifest.json').read_text(encoding='utf-8'))
    ver=sb.VERSION.rsplit('-',1)[1];cid=sex+'-'+preset.lower().replace('_','-')+'-v'+ver.replace('.','-')
    fbx=release/'fbx'/(cid+'.fbx')
    if preset=='BASE':
        # Keep the already-published immutable asset identity byte-for-byte.
        approved=Path(args.approved)/'release-verified/fbx'/(cid+'.fbx');shutil.copyfile(approved,fbx)
    else:shutil.copyfile(exported/'model.fbx',fbx)
    digest=sha(fbx);order=[k for k in BASIC[sex] if k in sb.FIELDS]+[k for k in sb.FIELDS if k not in BASIC[sex]]
    controls=[grid_control(k,*sb.FIELDS[k][:2],values[k],label_for(k,sb.FIELDS[k][3])) for k in order]
    recipes[cid]={'rules':sb.VERSION,'preset':preset,'asset':meta['shape']['asset'],'asset_sha256':digest,'defaults':values,'controls':[{k:v for k,v in c.items() if k!='label'} for c in controls]}
    body={'bodyId':cid,'bodyVersion':ver,'characterId':cid,'assetSha256':digest,'rigVersion':'approved-anatomy-54-v1','measurementVersion':'shape-authoring-v1','displayName':('남성' if sex=='male' else '여성')+' · '+entry['label'],'gender':sex,'role':'default' if preset=='BASE' else 'preset','order':(10 if sex=='male' else 30)+index,'supportedPoseIds':[], 'shape':{'schemaVersion':'body-shape.v1','controls':controls,'basic':len(BASIC[sex])},'shapeSource':{'rules':sb.VERSION,'blendSha256':sha(source/'model.blend'),'controllerSha256':sha(source/'standin_body.py')}}
    if preset=='BASE':body['displayName']='기본 남성' if sex=='male' else '기본 여성'
    characters[cid]={'display_name':body['displayName'],'artifact_uri_env':'BODY_'+cid.upper().replace('-','_')+'_URI','sha256':digest,'rig_profile':'mixamo','revision':ver}
    verification[preset]={'neutral':{k:meta[k] for k in ('vertices','bones','roundtrip_max_error_mm','roundtrip_joint_error_mm','roundtrip_weight_error')},'poses':{}}
    # Both joints are exercised for each body, including the strongest volume presets.
    for pose,arm,thigh,knee in [('raise',-68,0,0),('sit',60,90,90)]:
        for b in rig.pose.bones:b.matrix_basis=Matrix.Identity(4);b.rotation_mode='QUATERNION'
        for side,sign in [('Left',1),('Right',-1)]:
            sb.rotate(rig,side+'Arm',(0,1,0),sign*arm)
            if thigh:sb.rotate(rig,side+'UpLeg',(-1,0,0),thigh)
            if knee:sb.rotate(rig,side+'Leg',(1,0,0),knee)
        sb.correct_pose();co=sb.evaluated(m)
        if not np.isfinite(co).all():raise ValueError('nonfinite posed surface')
        lift={s:float(rig.pose.bones['Standin_'+s+'ScapularLift'].location.length) for s in ('Left','Right')}
        if pose=='raise' and min(lift.values())<=.001:raise ValueError('scapula helper did not move')
        verification[preset]['poses'][pose]={'finite':True,'scapularLift':lift,'activeCorrectives':sum(k.value>1e-5 for k in m.data.shape_keys.key_blocks)}
        # Save lightweight projected surface geometry for consistent visual QA.
        np.save(release/'qa'/(cid+'-'+pose+'.npy'),co.astype(np.float32))
    co=sb.compute(values)[0];faces=[list(p.vertices) for p in m.data.polygons]
    np.save(release/'qa'/(cid+'-neutral.npy'),co.astype(np.float32));np.save(release/'qa'/(cid+'-faces.npy'),np.asarray(faces,dtype=np.int32))
    for name,_ in sb.HANDLERS:
        for h in list(getattr(bpy.app.handlers,name)):
            if getattr(h,'_standin',False):getattr(bpy.app.handlers,name).remove(h)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh=bpy.data.meshes.new('preset');mesh.from_pydata(co.tolist(),[],faces);mesh.update();obj=bpy.data.objects.new('body',mesh);bpy.context.collection.objects.link(obj)
    for p in mesh.polygons:p.use_smooth=True
    scene=bpy.context.scene;configure_workbench(scene);camera=bpy.data.objects.new('camera',bpy.data.cameras.new('camera'));scene.collection.objects.link(camera)
    camera.location=(0,-5,1.);camera.rotation_euler=(Vector((0,0,1.))-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.type='ORTHO';camera.data.ortho_scale=2.25;scene.camera=camera
    scene.render.resolution_x=scene.render.resolution_y=512;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';png=release/'png'/(cid+'.png');scene.render.filepath=str(png);bpy.ops.render.render(write_still=True)
    body['neutralPreview']={'path':str(png),'sha256':sha(png),'characterSha256':digest,'pose':'t_pose','framing':'body-comparison.v1'};bodies.append(body)
    write(release/'body-models.local.json',{'assets':bodies});write(release/'recipes.json',recipes);write(release/'characters.json',characters);write(release/'verification.json',verification)
    print('PRESET_READY',cid,meta['roundtrip_max_error_mm'],flush=True)
print('PRESETS_READY',sex,len(bodies),flush=True)
