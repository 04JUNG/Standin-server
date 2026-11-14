"""Actual neutral FBX full-body review for EVERY published preset and large custom."""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
import bpy
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.character_lod.quality import load
from scripts.character_lod.render_review import render
from converter.outline_worker import tree
from converter.outline_fbx import write as write_fbx
from converter.skin_fbx import augment
from converter.skin_worker import verify_import
p=argparse.ArgumentParser();p.add_argument('--assets',required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);root=Path(a.assets);out=root/'review';package=root/'delivery-package';registry=json.loads((package/'shape-bakes.compact.host.json').read_text());manifest=json.loads((package/'skin-assets.json').read_text());rows=[];jobs=json.loads((package/'jobs.json').read_text())
for sex in ('male','female'):
 cases=json.loads((root/('quality-'+sex)/'quality.json').read_text())['source_cases']
 for cid,record in cases.items():
  for variant in ('original','candidate'):
   base=next(j['character'] for j in jobs if j['gender']==sex and j['variant']==variant and j['case']=='base');recipe=registry['characters'][base];source=registry['sources'][recipe['rules']];c=load(source['blend'],source['controller']);c.set_controls(record['controls'],correct=False);bundle=c.export_fixed(str(out/'shape-exports'));fbx=bundle/'model.fbx';asset=manifest['models'][base];uv=json.loads(gzip.decompress((package/'skins'/asset['uv']['path']).read_bytes()))
   for skin in ('default','flesh-guide'):
    actual=fbx;png=None;receipt=None
    if skin!='default':
     png=package/'skins'/asset['textures'][skin]['path'];data,version=tree(fbx);augment(data,asset,uv,png.read_bytes(),skin);actual=out/'fbx'/('shape-'+sex+'-'+variant+'-'+cid+'-'+skin+'.fbx');write_fbx(actual,data,version);errors,final=verify_import(fbx,actual,asset,uv,png.read_bytes());receipt={'errors':errors,'final':final,'csp_certified':False}
    dest=out/'images'/('shape-'+sex+'-'+variant+'-'+cid+'-'+skin+'.png');render(actual,uv if png else None,png,sex,'full',dest);rows.append(dict(gender=sex,variant=variant,shape=cid,pose='neutral',skin=skin,view='full',image='images/'+dest.name,file=str(actual),skin_verification=receipt,controls=record['controls']))
   (out/'shapes.json').write_text(json.dumps(rows,indent=2)+'\n');print('SHAPE_RENDER',sex,variant,cid,flush=True)
