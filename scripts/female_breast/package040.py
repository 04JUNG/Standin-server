"""Assemble the complete immutable review bundle; never publishes or activates.

The merged review catalog retains every existing card unchanged and keeps its
default. Runtime deployment/aggregate budgets and CSP skin acceptance remain
separate approval gates; this bundle does not rewrite a production pin.
"""
import argparse,copy,hashlib,json,shutil,struct,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.prepare_preset_bodies import stage_prepared

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')

def main(a):
 c=a.candidate.resolve();meta=read(c/'candidate.json');qa=read(c/'qa.json');stress=read(c/'pose-stress.json');runtime_qa=read(c/'runtime-independent.json')
 if not qa['success'] or not stress['no_added_global_pairs'] or not stress['no_added_breast_pairs'] or not runtime_qa['passed']:
  raise ValueError('Review bundle requires all geometry/runtime gates; thresholds are not adjustable here')
 path_qa=read(c/'posed-contact-path.json')
 if not path_qa['passed'] or not path_qa['complete'] or len(path_qa['rows'])!=301:
  raise ValueError('Complete independent posed contact path is required')
 raw=(c/'preview-full.ssr').read_bytes();header=json.loads(raw[8:8+struct.unpack('<I',raw[4:8])[0]])
 source={k:meta['source'][v] for k,v in [('blendSha256','blend_sha256'),('controllerSha256','controller_sha256')]}
 if header['source']!=source or runtime_qa['source']!=source or header['rules']!=meta['rules'] or len(raw)>32*1024*1024:
  raise ValueError('Runtime source/rules/size gate')
 registry=read(c/'release/shape-bakes.local.json');new=read(c/'release/body-models.local.json');chars=read(c/'release/characters.local.json')
 old=read(a.baseline/'production-bff-body-models.json');old_registry=read(a.baseline/'production-converter-shape-bakes.json')
 skin=read(c/'skin/skin-assets.json');proof=read(c/'skin/correspondence.json');prepared=read(c/'prepared-linux/results.json')
 if len(new['assets'])!=11 or len(skin['models'])!=11 or len(proof)!=11 or len(prepared)!=11 or not all(x['ok'] for x in prepared):
  raise ValueError('Incomplete preset/runtime/skin/prepared coverage')
 for entry in read(c/'prepared-linux/shape-bakes.local.json')['prepared_neutrals']['entries'].values():
  receipt=read(c/'prepared-linux/qa'/Path(entry['path']).name)
  if receipt['identity']['runtime'][2:4]!=['linux','x86_64']:raise ValueError('Linux/amd64 prepared runtime required')
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
 for folder in ['fbx','png']:
  (out/folder).mkdir()
  for path in (c/'release'/folder).glob('*'):
   if path.is_file() and path.suffix in ['.fbx','.png']:shutil.copy2(path,out/folder/path.name)
 (out/'sources').mkdir();new_source=registry['sources'][meta['rules']]
 for key,ext in [('blend','.blend'),('controller','.py')]:
  path=out/'sources'/(meta['rules'].replace('.','-')+ext);shutil.copy2(new_source[key],path)
  if sha(path)!=meta['source'][key+'_sha256']:raise ValueError('Source copy mismatch')
  new_source[key]='/app/body-staging/sources/'+path.name
 (out/'runtime').mkdir();runtime_path=out/'runtime'/(meta['rules'].replace('.','-')+'.ssr');runtime_path.write_bytes(raw)
 for position,asset in enumerate(new['assets']):
  cid=asset['characterId'];prior_cid=cid.rsplit('-breast7-',1)[0]
  prior=next(x for x in old['assets'] if x['characterId']==prior_cid)
  recipe=registry['characters'][cid]
  original_source=old_registry['characters'][prior_cid]
  if recipe['defaults']!=original_source['defaults']:raise ValueError('Current preset defaults changed')
  controls=copy.deepcopy(prior['shape']['controls'])
  next(x for x in controls if x['key']=='breast_size')['max']=2.
  asset['shape']={**copy.deepcopy(prior['shape']),'controls':controls}
  asset['rigVersion']=prior['rigVersion']
  asset.update(displayName=prior['displayName']+' · 가슴 개선 후보',gender='female',role='preset',order=100+position)
  asset['supportedPoseIds']=prior['supportedPoseIds']
  asset['shapeRuntime']={'path':'/app/body-staging/runtime/'+runtime_path.name,'sha256':sha(runtime_path),'bytes':len(raw),
    'format':header['format'],'rules':meta['rules'],'preset':recipe['preset']}
  png=out/'png'/(cid+'.png');asset['neutralPreview']={'path':'/app/body-staging/png/'+png.name,'sha256':sha(png),'characterSha256':asset['assetSha256'],'pose':'t_pose','framing':'body-comparison.v1'}
  recipe['controls']=[{k:v for k,v in x.items() if k!='label'} for x in controls]
  if sha(out/'fbx'/(cid+'.fbx'))!=asset['assetSha256'] or skin['models'][cid]['character_sha256']!=asset['assetSha256']:
   raise ValueError('FBX/skin/catalog pin mismatch')
 # Validate prepared receipts against local exact source bytes, then write the
 # deploy-shaped index. Neither old prepared entries nor approval data are copied.
 local_registry=copy.deepcopy(registry)
 local_registry['sources']=read(c/'release/shape-bakes.local.json')['sources']
 stage_prepared(c/'prepared-linux',local_registry,out,'/app/body-staging')
 registry['prepared_neutrals']=local_registry['prepared_neutrals']
 write(out/'shape-bakes.json',registry);write(out/'characters.json',chars)
 write(out/'body-models.candidate.json',new)
 merged=copy.deepcopy(old);merged['version']='female040-breast7-review-20261123';merged['assets']+=new['assets']
 if merged['assets'][:len(old['assets'])]!=old['assets'] or merged['defaultCharacterId']!=old['defaultCharacterId']:
  raise ValueError('Historical catalog/default changed')
 write(out/'body-models.review.json',merged)
 merged_registry=copy.deepcopy(old_registry)
 for section in ['sources','characters']:
  merged_registry[section].update(copy.deepcopy(registry[section]))
 previous_prepared=merged_registry.get('prepared_neutrals',{'schema':'prepared-neutral-index.v1','entries':{}})
 merged_registry['prepared_neutrals']={'schema':previous_prepared['schema'],
   'entries':{**previous_prepared['entries'],**registry['prepared_neutrals']['entries']}}
 for section in ['sources','characters']:
  if any(merged_registry[section][k]!=v for k,v in old_registry[section].items()):
   raise ValueError('Historical registry source/recipe changed')
 write(out/'shape-bakes.review.json',merged_registry)
 old_chars=read(a.baseline.parent/'inputs/skin/characters.json')
 merged_chars=copy.deepcopy(old_chars);merged_chars['characters'].update(chars['characters'])
 write(out/'characters.review.json',merged_chars)
 shutil.copytree(c/'skin',out/'skin')
 for name in ['candidate.json','qa.json','pose-stress.json','posed-contact-path.json','runtime-independent.json']:
  shutil.copy2(c/name,out/'qa'/name)
 files=[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()]
 write(out/'review-bundle.json',{'schema':'female-breast-review-bundle.v1','rules':meta['rules'],'source':source,
  'files':files,'coverage':{'presets':11,'linux_prepared':11,'uv_skin':11,'runtime_cases':runtime_qa['cases']},
  'historic_catalog_unchanged':True,'default_unchanged':True,'activation_performed':False,
  'skin_csp_certification_copied':False,'skin_csp_review_status':skin['csp_review_status'],
  'deployment_status':'review_only; aggregate body-assets release, budgets and skin acceptance require a separate reviewed activation'})
 print('COMPLETE_REVIEW_BUNDLE',out,len(files),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
 main(ap.parse_args())
