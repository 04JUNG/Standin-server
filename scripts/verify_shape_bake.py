import sys,os,json,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from converter.shape_spec import resolve_shape,digest_shape
from converter_api.runner import BlenderRunner,RunnerSettings
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'artifacts/shape-staging/release'
os.environ['CONVERTER_SHAPE_REGISTRY']=str(R/'shape-bakes.local.json')
recipes=json.loads((R/'shape-bakes.local.json').read_text())['characters']
# Use the same 6-decimal controls stored by the authoring controllers.
for r in recipes.values():
 r['defaults']={k:round(v,6) for k,v in r['defaults'].items()}
 for c in r['controls']:
  c['default']=round(c['default'],6);c['min']=round(c['min'],6)
(R/'shape-bakes.local.json').write_text(json.dumps({'schema':'shape-bake.v1','characters':recipes},ensure_ascii=False,indent=2))
runner=BlenderRunner(RunnerSettings.from_env(ROOT));runner.settings=__import__('dataclasses').replace(runner.settings,timeout_seconds=600,camera_timeout_seconds=600)
OUT=ROOT/'artifacts/shape-staging/verification';OUT.mkdir(exist_ok=True)
rows=[]
ids=sys.argv[1:] or ['male-base-v0-6-7','female-new-fat-v0-2-2']
bvh=Path('/Users/dowon/dev/Standin-server/data/bvh/rokoko_Shamwow_Guy_mixamo_01353_mirror.bvh').read_bytes()
for cid in ids:
 r=recipes[cid];recipe=resolve_shape(cid,r['asset_sha256']);t=time.monotonic()
 try:
  result=runner.convert(bvh_bytes=bvh,character_path=R/'fbx'/(cid+'.fbx'),character_id=cid,character_sha256=r['asset_sha256'],conversion_id=cid,preview_view='front',shape_recipe=recipe)
  (OUT/(cid+'.fbx')).write_bytes(result.artifact);(OUT/(cid+'.png')).write_bytes(result.preview)
  row={'id':cid,'ok':True,'seconds':time.monotonic()-t,'report':result.report}
 except Exception as e:
  row={'id':cid,'ok':False,'error':str(e),'report':getattr(e,'report',None)}
 rows.append(row);(OUT/(cid+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2));print(json.dumps(row,ensure_ascii=False),flush=True)
 if not row['ok']:raise SystemExit(1)
