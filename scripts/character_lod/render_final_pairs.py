"""Expose real Converter hand smoothing differences, separate from authoring QA."""
import argparse,json,gzip,sys
from pathlib import Path
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.character_lod.render_review import render
from converter.shape_bake import import_full_character
from converter import retarget as rt
p=argparse.ArgumentParser();p.add_argument('--assets',required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);root=Path(a.assets);review=root/'review';out=review/'converter-images';out.mkdir(exist_ok=False);rows=json.loads((root/'benchmark/results.json').read_text());skins=json.loads((root/'delivery-package/skin-assets.json').read_text());cameras={};renders=[]
for sex in ('male','female'):
 original=next(r for r in rows if r['gender']==sex and r['case']=='base' and r['skin']=='default' and r['variant']=='original' and r['trial']==0);rt.reset_scene();arm,meshes=import_full_character(original['file'].replace('/work',str(root),1))
 def head(name):return list(arm.matrix_world@arm.data.bones['mixamorig:'+name].head_local)
 cameras[sex]={'full':{'target':[0,0,.92],'offset':[0,-5,.15],'scale':2.15},'left_hand':{'target':head('LeftHandMiddle1'),'offset':[0,-5,1.4],'scale':.22},'right_hand':{'target':head('RightHandMiddle1'),'offset':[0,-5,1.4],'scale':.22}}
 (out/'cameras.json').write_text(json.dumps(cameras))
 for skin in ('default','flesh-guide'):
  for variant in ('original','candidate'):
   row=next(r for r in rows if r['gender']==sex and r['case']=='base' and r['skin']==skin and r['variant']==variant and r['trial']==0);path=Path(row['file'].replace('/work',str(root),1));asset=skins['models'][row['character']];uv=json.loads(gzip.decompress((root/'delivery-package/skins'/asset['uv']['path']).read_bytes()));png=root/'delivery-package/skins'/asset['textures'][skin]['path'] if skin!='default' else None
   for view in ('full','left_hand','right_hand'):
    dest=out/f'{sex}-{variant}-{skin}-{view}.png';render(path,uv if png else None,png,sex,view,dest);renders.append({'gender':sex,'variant':variant,'shape':'converter-ob015','pose':'hands','skin':skin,'view':view,'image':'converter-images/'+dest.name,'file':str(path),'note':'actual Converter hand-library and topology-dependent fist smoothing'})
   print('FINAL_RENDER',sex,skin,variant,flush=True)
(review/'converter-renders.json').write_text(json.dumps(renders,indent=2)+'\n')
