"""Real HTTP contract -> isolated Blender -> final FBX -> PNG/GLB integration QA."""
import sys,os,json,base64,hashlib,time
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));R=ROOT/'artifacts/shape-staging/release';OUT=R.parent/'api-qa';OUT.mkdir(exist_ok=True)
recipes=json.loads((R/'shape-bakes.local.json').read_text())['characters'];reg=json.loads((R/'characters.local.json').read_text())
os.environ['CONVERTER_SHAPE_REGISTRY']=str(R/'shape-bakes.local.json')
from converter_api.registry import CharacterRegistry
from converter_api.runner import BlenderRunner,RunnerSettings
from converter_api.app import create_app
from converter.shape_spec import digest_shape
from fastapi.testclient import TestClient
registry=CharacterRegistry(R/'characters.local.json',default_character_id=next(iter(recipes)),environ={v['artifact_uri_env']:str(R/'fbx'/(k+'.fbx')) for k,v in reg['characters'].items()})
runner=BlenderRunner(replace(RunnerSettings.from_env(ROOT),timeout_seconds=600,camera_timeout_seconds=600));client=TestClient(create_app(registry=registry,runner=runner))
assert client.get('/healthz').status_code==200
contract=client.get('/preview-contract').json();cases=[]
poses={
 'walk':'rokoko_Shamwow_Guy_mixamo_01353_mirror',
 'sit':'rokoko_SitDown_UseLaptop_mixamo_00918',
 'bend':'Dig And Plant Seeds_00165',
 'sit90':'rokoko_Lawyer_02_mixamo_00008',
 'deep':'Finding_01',
}
for cid in recipes:cases.append((cid,'walk','front',None))
for cid in ['male-fat-max-v0-6-7','male-muscle-max-v0-6-7','female-new-fat-v0-2-2','female-new-muscular-v0-2-2']:
 for pose in ['sit','bend']:cases.append((cid,pose,'side',None))
for cid in ['male-base-v0-6-7','female-base-v0-2-2']:
 for frac in [.5,1.]:
  change={'physique':(1.7 if cid.startswith('male') else 1.4)*frac,'muscle':(2. if cid.startswith('male') else 4.)*frac,'height':180.,'shoulders':.3}
  if cid.startswith('female'):change.update(fat_gain=2.*frac,breast_size=.5*frac)
  cases.append((cid,'sit','front',change))
 for fmt in ['model','back']:cases.append((cid,'bend',fmt,None))
for cid in recipes:
 if cid not in ['male-fat-max-v0-6-7','male-muscle-max-v0-6-7','female-new-fat-v0-2-2','female-new-muscular-v0-2-2']:
  for pose in ['sit','bend']:cases.append((cid,pose,'side',None))
stress=os.environ.get('QA_STRESS')=='1'
if stress:cases=[(cid,pose,'side',None) for cid in recipes for pose in ['sit90','deep']]
shard=int(os.environ.get("QA_SHARD","0"));shards=int(os.environ.get("QA_SHARDS","1"))
rows=[]
for i,(cid,pose,view,change) in enumerate(cases,start=65 if stress else 0):
 if i % shards != shard:continue
 ident=f'{i:02d}-{cid}-{pose}-{view}';cache=OUT/(ident+'.json')
 if cache.exists():
  old=json.loads(cache.read_text())
  if old.get('ok') and old.get('preview_revision')==contract['preview_revision'] and (not change or all(abs(old['shape_bake']['controls'].get('height_cm' if k=='height' else k, float('inf'))-v)<1e-8 for k,v in change.items())):rows.append(old);continue
 bvhpath=Path('/Users/dowon/dev/Standin-server/data/bvh')/(poses[pose]+'.bvh');bvh=bvhpath.read_bytes();r=recipes[cid]
 fields={'character_id':cid,'output_scope':'full','preview_view':('front' if view=='model' else view),'preview_format':('model' if view=='model' else 'png'),'expected_bvh_sha256':hashlib.sha256(bvh).hexdigest(),'expected_character_sha256':r['asset_sha256'],'expected_preview_revision':contract['preview_revision']}
 shape=None
 if change:
  values={c['key']:c['default'] for c in r['controls']};values.update(change)
  shape={'schemaVersion':'body-shape.v1','controls':values,'shapeSha256':digest_shape(cid,r['asset_sha256'],values)};fields['shape']=json.dumps(shape)
 start=time.monotonic();response=client.post('/convert-framed',files={'bvh':('pose.bvh',bvh,'application/octet-stream')},data=fields)
 if response.status_code!=200:
  row={'id':ident,'ok':False,'status':response.status_code,'error':response.text};cache.write_text(json.dumps(row,indent=2));print(row,flush=True);raise SystemExit(1)
 d=response.json();fbx=base64.b64decode(d.pop('fbx_base64'));preview=base64.b64decode(d.pop('preview_base64'))
 assert hashlib.sha256(fbx).hexdigest()==d['fbx_sha256'];assert hashlib.sha256(preview).hexdigest()==d['preview_sha256']
 assert d['shape_sha256']==(shape['shapeSha256'] if shape else None)
 assert d['shape_bake']['pose_transfer_max_joint_error_mm']<=.01
 (OUT/(ident+'.fbx')).write_bytes(fbx);(OUT/(ident+('.glb' if view=='model' else '.png'))).write_bytes(preview)
 row={'id':ident,'ok':True,'character_id':cid,'pose_id':poses[pose],'source_bvh_sha256':fields['expected_bvh_sha256'],'view':view,'seconds':time.monotonic()-start,**d}
 cache.write_text(json.dumps(row,ensure_ascii=False,indent=2));rows.append(row);print(ident,'PASS',round(row['seconds'],2),flush=True)
(OUT/('summary.json' if shards==1 else f'summary-{shard}-{int(stress)}.json')).write_text(json.dumps({'cases':len(rows),'passed':sum(r['ok'] for r in rows),'rows':rows},ensure_ascii=False,indent=2))
