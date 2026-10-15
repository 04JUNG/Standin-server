"""Package verified sources for a new, immutable S3 staging release.

Upload is opt-in. This command never updates ECS, production, or existing keys.
"""
import argparse,hashlib,json,shutil,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/shape-staging';R=BASE/'release';P=BASE/'package'
BUCKET='standinstagingapp-assetsbucket5cb76180-fiwlikrhl1sk'
def sha(data):return hashlib.sha256(data).hexdigest()
def write(name,value):
 p=P/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def copy(src,name):
 p=P/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p)
def publish(files,digest,prefix):
 for f in files:
  subprocess.run(['aws','s3api','put-object','--bucket',BUCKET,'--key',prefix+'/'+f['path'],'--body',str(P/f['path']),'--if-none-match','*'],check=True,stdout=subprocess.DEVNULL)
 subprocess.run(['aws','s3api','put-object','--bucket',BUCKET,'--key',prefix+'/manifest.json','--body',str(P/'manifest.json'),'--if-none-match','*','--content-type','application/json'],check=True,stdout=subprocess.DEVNULL)
 with tempfile.TemporaryDirectory(prefix='shape-s3-verify-') as tmp:
  for f in files+[{'path':'manifest.json','size':(P/'manifest.json').stat().st_size,'sha256':digest}]:
   p=Path(tmp)/'object';subprocess.run(['aws','s3api','get-object','--bucket',BUCKET,'--key',prefix+'/'+f['path'],str(p)],check=True,stdout=subprocess.DEVNULL)
   assert p.stat().st_size==f['size'] and sha(p.read_bytes())==f['sha256'],f['path']
 (BASE/'upload-verification.json').write_text(json.dumps({'bucket':BUCKET,'prefix':prefix,'objects_verified':len(files)+1,'all_sizes_and_sha256_match':True,'manifest_sha256':digest},indent=2))

def main(upload=False):
 rows=[json.loads(p.read_text()) for p in (BASE/'api-qa').glob('*.json') if not p.name.startswith('summary')]
 assert len(rows)==103 and all(r.get('ok') for r in rows),'all 103 real API checks must pass before packaging'
 runtime_qa=json.loads((BASE/'runtime-qa/summary.json').read_text());assert runtime_qa['passed'] and runtime_qa['models']==19
 bff_qa=json.loads((BASE/'bff-real-pair.json').read_text());assert bff_qa['passed']
 assert len({r['preview_revision'] for r in rows})==1,'QA must use one final runtime revision'
 assert not P.exists(),'use a fresh package directory; never overwrite a published release'
 P.mkdir();recipes=json.loads((R/'shape-bakes.local.json').read_text());bodies=json.loads((R/'body-models.local.json').read_text());characters=json.loads((R/'characters.local.json').read_text());catalog=[];env={}
 for body in bodies['assets']:
  cid=body['characterId'];r=recipes['characters'][cid]
  checks=[q for q in rows if q['character_id']==cid];supported=sorted({q['pose_id'] for q in checks});assert len(supported)==5
  body['supportedPoseIds']=supported;body['neutralPreview']['path']='/app/body-staging/png/'+cid+'.png'
  for key in ['blend','controller']:
   src=Path(r[key]);rel='sources/'+src.name;copy(src,rel);assert sha(src.read_bytes())==r[key+'_sha256'];r[key]='/app/body-staging/'+rel
  copy(R/'fbx'/(cid+'.fbx'),'fbx/'+cid+'.fbx');copy(R/'png'/(cid+'.png'),'png/'+cid+'.png')
  assert sha((P/'fbx'/(cid+'.fbx')).read_bytes())==body['assetSha256']
  sex='male' if cid.startswith('male-') else 'female'
  runtime=Path('/Users/dowon/dev/Standin-client-shapekey-live/public/mock')/('shape-runtime-0.6.7.ssr' if sex=='male' else 'shape-runtime-female-0.2.2.ssr')
  rel='runtime/'+('male-0-6-7.ssr' if sex=='male' else 'female-0-2-2.ssr');copy(runtime,rel)
  expected=('a7ebf01c92a1d2eff63ef714b1ca1c51c6dfd83cef3c4d8e87c0ea816af20cb2' if sex=='male' else 'b8c6a2dec1ebd1f2c73bfcb98b8e5e45a0950593fea7e28af9d0bc6b0d101bc4')
  assert sha(runtime.read_bytes())==expected
  body['shapeRuntime']={'path':'/app/body-staging/'+rel,'sha256':expected,'bytes':runtime.stat().st_size,'format':'standin-shape-runtime.v1','rules':r['rules'],'preset':r['preset']}
  body['shape']['basic']=7 if sex=='female' else 5
  labels={'height':'키','heads':'등신','shoulders':'어깨너비','muscle':'근육량','physique':'지방','fat_gain':'지방 볼륨','breast_size':'가슴 크기','breast_projection':'가슴 돌출도','breast_width':'가슴 너비','breast_height':'가슴 높이','breast_spacing':'가슴 간격','fat_distribution':'지방 분포','chest_width':'흉곽 너비','chest_depth':'흉곽 두께'}
  for control in body['shape']['controls']:
   if control['key'] in labels:control['label']=labels[control['key']]
  report={'status':'integration_verified','scope':'five real BVHs; not full-library or zero-intersection certification','body_id':cid,'body_version':body['bodyVersion'],'asset_sha256':body['assetSha256'],'supported_pose_ids':supported,'cases':checks}
  write('qa/'+cid+'.json',report)
  catalog.append({'body_id':cid,'body_version':body['bodyVersion'],'availability':'draft','fbx_path':'fbx/'+cid+'.fbx','asset_sha256':body['assetSha256'],'rig_version':body['rigVersion'],'measurement_version':body['measurementVersion'],'qa_report':'qa/'+cid+'.json','qa_sha256':sha((P/'qa'/(cid+'.json')).read_bytes()),'quality_priority':0,'supported_pose_ids':supported,'metadata':{'presentation_style':'stylized'},'attributes':{},'projections':[]})
  env[characters['characters'][cid]['artifact_uri_env']]='/app/body-staging/fbx/'+cid+'.fbx'
 write('qa/live-runtime.json',runtime_qa);write('qa/bff-pair.json',bff_qa)
 write('shape-bakes.json',recipes);write('body-models.json',bodies);write('characters.json',characters)
 write('body-catalog.json',{'schema_version':'body-catalog.v1','catalog_version':'shape-staging-20261015','default_body_id':None,'assets':catalog})
 write('qa/release-verification.json',{'status':'integration_verified','api_cases':103,'models':19,'live_runtime_max_error_mm':runtime_qa['max_error_mm'],'bff_pairs_verified':True,'source_models_unchanged':True,'preview_revision':rows[0]['preview_revision'],'notes':['Male and female live runtimes are from Claude; label/basic schema merged without editing Claude originals.','Authoring contact correction is now baked in the converter before preview/export.','Catalog matching stays draft; no full-library or no-intersection claim.','No ECS service or production pointer changed.']})
 files=[]
 for p in sorted(P.rglob('*')):
  if not p.is_file():continue
  n=p.relative_to(P).as_posix()
  roles=(['converter'] if n.startswith('sources/') or n in ('characters.json','shape-bakes.json') else ['bff'] if n.startswith(('png/','runtime/')) or n=='body-models.json' else ['inference','converter'] if n.startswith('fbx/') else ['inference'])
  files.append({'path':n,'roles':roles,'size':p.stat().st_size,'sha256':sha(p.read_bytes())})
 write('manifest.json',{'schema':'body-assets.v1','version':'shape-20261015-v2','files':files})
 digest=sha((P/'manifest.json').read_bytes());prefix='body-models/releases/shape-20261015-'+digest[:12]
 release={'version':'shape-20261015-v2','bucket':BUCKET,'manifestKey':prefix+'/manifest.json','manifestSha256':digest,'bodyEnvironment':env,'defaultCharacterId':bodies['defaultCharacterId'],'fileCount':len(files),'bytes':sum(f['size'] for f in files),'activation':'not activated; deploy converter+BFF code and merge Claude UI separately','flagsAfterDeployment':{'BODY_SHAPE_EDIT_ENABLED':'true','BODY_SHAPE_BAKE_ENABLED':'true'}}
 (BASE/'release.json').write_text(json.dumps(release,indent=2)+'\n')
 if upload:publish(files,digest,prefix)
 print(json.dumps(release,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--upload',action='store_true');args=ap.parse_args()
 if args.upload and P.exists():
  release=json.loads((BASE/'release.json').read_text());manifest=json.loads((P/'manifest.json').read_text());digest=sha((P/'manifest.json').read_bytes())
  assert digest==release['manifestSha256']
  for f in manifest['files']:assert (P/f['path']).stat().st_size==f['size'] and sha((P/f['path']).read_bytes())==f['sha256']
  publish(manifest['files'],digest,release['manifestKey'].rsplit('/',1)[0]);print('staging upload and full re-download verified')
 else:main(args.upload)
