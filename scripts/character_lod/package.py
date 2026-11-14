"""Create an isolated local registry and immutable compatibility data; no uploads."""
import argparse,copy,gzip,hashlib,json,math,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from converter.shape_spec import controller_key

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main(a):
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False);original=json.loads(Path(a.registry).read_text());r={'schema':'shape-bake.v2','sources':{},'characters':{}};jobs=[]
 root=Path(a.assets);sourceassets=Path(a.skin_assets);skins=json.loads((ROOT/'converter/skin_assets.v1.json').read_text());skins['models']={};skins['revision']='protected-'+a.identity_suffix;(out/'fbx').mkdir();(out/'skins').mkdir()
 for sex,ver,rules,stem in [('male','0-8-0','standin-body-0.8.0','standin-body-0-8-0'),('female','0-4-0','standin-female-body-0.4.0','standin-female-body-0-4-0')]:
  base=sex+'-base-v'+ver;quality=json.loads((root/('quality-'+sex)/'quality.json').read_text());selected=Path(getattr(a,sex)) if getattr(a,sex) else None
  for variant in ('original','candidate') if selected else ('original',):
   if variant=='original':folder=root/'originals';blend=folder/(stem+'.blend');controller=folder/(stem+'.py');version=rules
   else:folder=selected;blend=folder/'model.blend';controller=folder/'controller.py';version=json.loads((folder/'manifest.json').read_text())['rules']
   r['sources'][version]={'blend':str(blend),'controller':str(controller),'blend_sha256':sha(blend),'controller_sha256':sha(controller)}
   for case in ('base','preset','custom'):
    origcid=base if case!='preset' else sex+'-ideal83-v'+ver;raw=copy.deepcopy(original['characters'][origcid]);values=copy.deepcopy(raw['defaults'])
    if case=='custom':values=quality['source_cases']['custom-large']['controls'];raw['preset']='CUSTOM_REVIEW'
    cid=sex+'-'+variant+'-'+case+'-'+a.identity_suffix;raw['rules']=version;raw['defaults']=values
    for control in raw['controls']:
     key=controller_key(control['key']);value=values[key]
     if case=='custom':value=round(control['min']+math.floor((value-control['min'])/control['step']+1e-7)*control['step'],6);values[key]=value
     control['default']=value
    if variant=='original':fbx=root/'originals'/(origcid+'.fbx') if case!='preset' else Path(a.original_fbx)/(origcid+'.fbx')
    else:
     q=json.loads(Path(getattr(a,sex+'_quality')).read_text())['candidates'][folder.name];record=q['custom-large' if case=='custom' else origcid]
     if 'neutral_export' not in record:raise ValueError('candidate does not pass neutral geometry/export gates: '+cid)
     fbx=Path(record['neutral_export'])
    target=out/'fbx'/(cid+'.fbx');shutil.copyfile(fbx,target);raw['asset_sha256']=sha(target)
    if variant=='candidate':raw['asset']='local-review-'+folder.name
    r['characters'][cid]=raw;jobs.append({'gender':sex,'variant':variant,'case':case,'character':cid})
    skinrow=copy.deepcopy(json.loads((ROOT/'converter/skin_assets.v1.json').read_text())['models'][origcid]);skinrow['character_sha256']=raw['asset_sha256']
    # Original UVs are exact; candidate corner UVs/weights are pinned by the
    # Blender finalize step, after its actual FBX export has been reloaded.
    for entry in [skinrow['uv'],*skinrow['textures'].values(),*skinrow['preview_textures'].values()]:
     src=sourceassets/entry['path'];assert sha(src)==entry['sha256'];shutil.copyfile(src,out/'skins'/src.name)
    skins['models'][cid]=skinrow
 write(out/'shape-bakes.json',r);write(out/'jobs.json',jobs);write(out/'skin-assets.json',skins)
 print('LOCAL_PACKAGE',out,len(jobs))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--identity-suffix',default='lod-review-20261114');p.add_argument('--registry',required=True);p.add_argument('--assets',required=True);p.add_argument('--skin-assets',required=True);p.add_argument('--original-fbx',required=True);p.add_argument('--male');p.add_argument('--female');p.add_argument('--male-quality');p.add_argument('--female-quality');p.add_argument('--out',required=True);main(p.parse_args())
