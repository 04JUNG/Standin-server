"""Package the neutral FBXs that already passed independent round-trip checks."""
import argparse,hashlib,importlib.util,json,shutil,sys
from pathlib import Path
import bpy
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.build_shape_release import BASIC,grid_control,label_for
from converter.shape_spec import load_registry
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
args=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);root=Path(args.root);out=root/'release-verified';out.mkdir(exist_ok=False)
sources={};recipes={};characters={};bodies=[]
for sex in ('male','female'):
    folder=root/'authoring'/sex;validation=json.loads((folder/'validation.json').read_text(encoding='utf-8'));meta=validation['neutral']
    if validation['default_error_mm']>.01 or any(x['degenerate_faces'] or x['reversed_faces'] or not x['finite'] for x in validation['cases'].values()):raise ValueError('source geometry QA failed')
    bpy.ops.wm.open_mainfile(filepath=str(folder/'model.blend'))
    spec=importlib.util.spec_from_file_location('package_'+sex,folder/'standin_body.py');sb=importlib.util.module_from_spec(spec);spec.loader.exec_module(sb)
    rules=sb.VERSION;ver=rules.rsplit('-',1)[1];cid=sex+'-base-v'+ver.replace('.','-');src={}
    for key,name,ext in [('blend','model.blend','.blend'),('controller','standin_body.py','.py')]:
        dest=out/'sources'/(rules+ext);dest.parent.mkdir(exist_ok=True);shutil.copyfile(folder/name,dest);src[key]=str(dest);src[key+'_sha256']=sha(dest)
    sources[rules]=src
    neutral=folder/'qa-exports'/meta['identity']/'model.fbx'
    assert sha(neutral)==meta['fbx_sha256']
    dest=out/'fbx'/(cid+'.fbx');dest.parent.mkdir(exist_ok=True);shutil.copyfile(neutral,dest)
    values=validation['controls'];order=[k for k in BASIC[sex] if k in sb.FIELDS]+[k for k in sb.FIELDS if k not in BASIC[sex]]
    controls=[grid_control(k,*sb.FIELDS[k][:2],values[k],label_for(k,sb.FIELDS[k][3])) for k in order]
    recipes[cid]={'rules':rules,'preset':'BASE','asset':meta['shape']['asset'],'asset_sha256':meta['fbx_sha256'],'defaults':values,
                  'controls':[{k:v for k,v in c.items() if k!='label'} for c in controls]}
    bodies.append({'bodyId':cid,'bodyVersion':ver,'characterId':cid,'assetSha256':meta['fbx_sha256'],
        'rigVersion':'approved-anatomy-54-v1','measurementVersion':'shape-authoring-v1','displayName':'기본 남성' if sex=='male' else '기본 여성',
        'gender':sex,'role':'default','order':10 if sex=='male' else 20,'supportedPoseIds':[],
        'shape':{'schemaVersion':'body-shape.v1','controls':controls,'basic':len([k for k in BASIC[sex] if k in sb.FIELDS])},
        'shapeSource':{'rules':rules,'blendSha256':src['blend_sha256'],'controllerSha256':src['controller_sha256']}})
    characters[cid]={'display_name':bodies[-1]['displayName'],'artifact_uri_env':'BODY_'+cid.upper().replace('-','_')+'_URI',
                     'sha256':meta['fbx_sha256'],'rig_profile':'mixamo','revision':ver}
write(out/'shape-bakes.local.json',{'schema':'shape-bake.v2','sources':sources,'characters':recipes})
write(out/'characters.local.json',{'schema_version':1,'characters':characters})
write(out/'body-models.local.json',{'version':'anatomy-20261111-v5','defaultCharacterId':bodies[0]['characterId'],'assets':bodies})
load_registry(out/'shape-bakes.local.json');print('VERIFIED_RELEASE_READY',out,flush=True)
