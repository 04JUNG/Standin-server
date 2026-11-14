"""Alternating paired, fully uncached Converter + skin + file-ready benchmark.

Run in the pinned Linux amd64 container with cpus=2, memory=4g. Each conversion
uses a fresh worker; every model uses a verified prepared neutral. Custom is an
explicit OFFLINE local recipe for this experiment, not a runtime cache change.
"""
import argparse,hashlib,json,os,statistics,sys,time
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(data):return hashlib.sha256(data).hexdigest()
def main(a):
 out=Path(a.out);out.mkdir(parents=True,exist_ok=a.resume);package=Path(a.package)
 os.environ.update(CONVERTER_SHAPE_REGISTRY=a.registry,CONVERTER_FINAL_CACHE_ENABLED='0',CONVERTER_PREPARED_PRESETS_ENABLED='1',CONVERTER_SKIN_ASSET_ROOT=str(package/'skins'))
 from converter.shape_spec import load_registry,resolve_shape
 from converter_api.runner import BlenderRunner,RunnerSettings
 from converter_api.skin import _prepare_uncached
 from converter import skin_spec
 registry=load_registry();jobs=json.loads((package/'jobs.json').read_text());bvh=Path(a.bvh).read_bytes();rows=json.loads((out/'results.json').read_text()) if a.resume and (out/'results.json').exists() else []
 settings=replace(RunnerSettings.from_env(ROOT),blender_binary=a.blender,worker_path=Path(__file__).with_name('bench_worker.py'),retarget_cache_bytes=0,fast_scene_reset=False,shape_timeout_seconds=600,process_env={'CONVERTER_FINAL_CACHE_ENABLED':'0','CONVERTER_PREPARED_PRESETS_ENABLED':'1','STANDIN_LAB_PREVIEW_TRACE':'1','STANDIN_LAB_FIRST_ENTRY_PROFILE':'1'})
 for sex in (('male',) if a.smoke else ('male','female')):
  for case in (('base',) if a.smoke else ('base','preset','custom')):
   for skin in (('default',) if a.smoke else ('default','flesh-guide')):
    for trial in range(1 if a.smoke else a.repeats):
     for variant in (('original','candidate') if trial%2==0 else ('candidate','original')):
      if any(r['gender']==sex and r['case']==case and r['skin']==skin and r['trial']==trial and r['variant']==variant for r in rows):continue
      job=next(j for j in jobs if j['gender']==sex and j['case']==case and j['variant']==variant);cid=job['character'];raw=registry['characters'][cid];recipe=resolve_shape(cid,raw['asset_sha256']);row={**job,'skin':skin,'trial':trial,'bvh_sha256':sha(bvh),'controls':recipe['values'],'cache_policy':'completed FBX/retarget/appearance disabled; prepared neutral required'};start=time.monotonic()
      try:
       runner=BlenderRunner(settings);r=runner.convert(bvh_bytes=bvh,character_path=package/'fbx'/(cid+'.fbx'),character_id=cid,character_sha256=raw['asset_sha256'],conversion_id=cid+'-'+skin+'-'+str(trial),shape_recipe=recipe,hands={'left':'fist','right':'open'})
       row['converter_seconds']=time.monotonic()-start;data=r.artifact;row['report']=r.report
       if r.report['shape_bake']['neutral_preparation']['status']!='prepared':raise ValueError('both sides require prepared neutrals')
       post=time.monotonic()
       if skin!='default':
        data,receipt=_prepare_uncached(runner,data,dict(source_artifact_sha256=sha(data),skin={'version':skin_spec.VERSION,'id':skin},binding_sha256=sha((cid+str(trial)).encode()),character_id=cid,character_sha256=raw['asset_sha256'],output_scope='full'),checkpoint=lambda:None);row['skin_receipt']=receipt
       row['skin_seconds']=time.monotonic()-post;dest=out/(cid+'-'+skin+'-'+str(trial)+'.fbx');save=time.monotonic()
       with dest.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
       row.update(ok=True,bytes=len(data),sha256=sha(data),file=str(dest),file_save_seconds=time.monotonic()-save,file_ready_seconds=time.monotonic()-start)
      except Exception as e:row.update(ok=False,error=str(e),seconds_until_failure=time.monotonic()-start)
      rows.append(row);write(out/'results.json',rows);print('LOD_BENCH',sex,case,skin,trial,variant,row.get('ok'),round(row.get('file_ready_seconds',0),3),row.get('error'),flush=True)
 from scripts.character_lod.contract import require_paired_runs
 require_paired_runs(rows,1 if a.smoke else a.repeats)
 summary=[]
 for sex in ('male','female'):
  for case in ('base','preset','custom'):
   for skin in ('default','flesh-guide'):
    item={'gender':sex,'case':case,'skin':skin,'runs':{}}
    for variant in ('original','candidate'):
     samples=[r for r in rows if all(r[k]==v for k,v in [('gender',sex),('case',case),('skin',skin),('variant',variant)])];times=[r['file_ready_seconds'] for r in samples if r['ok']];item['runs'][variant]={'successes':len(times),'failures':len(samples)-len(times),'mean':statistics.mean(times) if times else None,'median':statistics.median(times) if times else None,'min':min(times) if times else None,'max':max(times) if times else None}
    if all(v['successes']==a.repeats for v in item['runs'].values()):item['median_reduction_percent']=100*(1-item['runs']['candidate']['median']/item['runs']['original']['median'])
    summary.append(item)
 write(out/'summary.json',{'scope':'Local Converter, final reload, optional verified skin and local fsync file-ready. No API/network/UI/CSP timing. Nested worker spans must not be added together.','repeats':a.repeats,'rows':summary})
 if any(not r['ok'] for r in rows):raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--package',required=True);p.add_argument('--registry',required=True);p.add_argument('--bvh',required=True);p.add_argument('--blender',default='/opt/blender/blender');p.add_argument('--repeats',type=int,default=3);p.add_argument('--out',required=True);main(p.parse_args())
