"""Check an immutable SSR2 reference asset against fresh source controller output.

Uses the hash-pinned public catalog and the same independent random controls and
unrounded limits as check_sampled_runtime. Geometry is compared with the source;
positions are evaluated by the compiled real client parser/evaluator, not a Python
copy of that evaluator. No output runtime is rewritten or implicitly approved.
"""
import argparse,base64,hashlib,importlib.util,json,os,struct,subprocess,sys,time,zlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from check_sampled_runtime import BASIC_GUESS,catalog_controls,published_value

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser()
 for key in ('runtime','blend','controller','catalog','catalog-sha256','client-evaluator','report'):ap.add_argument('--'+key,required=True)
 ap.add_argument('--seed',type=int,required=True);ap.add_argument('--extra-cases');args=ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
 raw=Path(args.runtime).read_bytes();assert raw[:4]==b'SSR2' and 8<=len(raw)<=32*1024*1024
 size=struct.unpack('<I',raw[4:8])[0];assert size%4==0 and size<=256*1024 and 8+size<len(raw)
 h=json.loads(raw[8:8+size]);assert h['encoding']=='female-reference-v4-zlib' and h['dataVersion']==1
 assert all(sha(path)==h['source'][key] for key,path in [('blendSha256',args.blend),('controllerSha256',args.controller)])
 bounds=catalog_controls(args.catalog,args.catalog_sha256,h,dict(sha256=sha(args.runtime),bytes=len(raw)))
 assert 0<h['rawBytes']<=192*1024*1024
 decoder=zlib.decompressobj();payload=decoder.decompress(raw[8+size:],h['rawBytes']+1);assert decoder.eof and not decoder.unused_data and len(payload)==h['rawBytes'] and hashlib.sha256(payload).hexdigest()==h['rawSha256']
 arrays={}
 for name,spec in h['arrays'].items():
  dtype=np.dtype(spec['dtype']);count=int(np.prod(spec['shape']));assert spec['offset']%dtype.itemsize==0 and count*dtype.itemsize==spec['bytes'] and 0<=spec['offset']<=len(payload)-spec['bytes']
  a=np.frombuffer(payload,dtype=dtype,count=count,offset=spec['offset']).reshape(spec['shape']);assert np.isfinite(a).all();arrays[name]=a
 import bpy
 bpy.ops.wm.open_mainfile(filepath=args.blend);spec=importlib.util.spec_from_file_location('reference_check_source',args.controller);sb=importlib.util.module_from_spec(spec);spec.loader.exec_module(sb);sb.register();assert sb.VERSION==h['rules']
 preview=sb.preview_mesh();faces=np.asarray([p.vertices[:] for p in preview.data.polygons],dtype='<u2');assert len(preview.data.vertices)==h['vertexCount'] and np.array_equal(faces,arrays['faces'])
 ref,ids,bary,offset=sb._lod_reference();assert ids.shape==(94901,1) and not offset.any() and np.array_equal(bary,np.ones_like(bary));source=ref.source();assert np.array_equal(source['co'],arrays['base']) and np.array_equal(source['faces'],arrays['referenceFaces'])
 projection=zlib.decompress(base64.b64decode(preview['projection']));nv=h['vertexCount'];pi=np.frombuffer(projection[:nv*12],dtype='<i4').reshape(nv,3);pw=np.frombuffer(projection[nv*12:],dtype='<f4').reshape(nv,3)
 assert np.array_equal(ids[pi,0],arrays['previewIndices']) and np.array_equal(pw,arrays['previewWeights'])
 for name in h['deltaNames']:assert np.array_equal(source['deltas'][name],arrays['raw:'+name])
 for name in h['offsetNames']:assert np.array_equal(ref.anatomy_delta(source,name)-source['deltas'][name],arrays['offset:'+name])
 geometry=dict(vertexCount=nv,faceCount=len(faces),indicesMatchFinalSource=True,facesSha256=hashlib.sha256(faces.tobytes()).hexdigest(),referenceProjectionMatchesFinalSource=True,allAuthoredDeltasMatchFinalSource=True,staticAnatomicalOffsetsMatchFinalSource=True)
 rng=np.random.default_rng(args.seed);keys=list(sb.FIELDS);basic=[k for k in BASIC_GUESS if k in sb.FIELDS];cases=[]
 folder=Path(args.report).with_suffix('');folder.mkdir(exist_ok=False)
 def case(preset,mode,c):
  c={k:float(v) for k,v in c.items()};name=f'{len(cases):04d}-{preset}-{mode}';start=time.perf_counter();truth=sb.preview_compute(sb.normalize(c));path=folder/(name+'.f64');path.write_bytes(truth.astype('<f8').tobytes());cases.append(dict(id=name,preset=preset,mode=mode,controls=c,truthFile=str(path),truthSha256=sha(path),sourceComputeMs=(time.perf_counter()-start)*1000))
 for preset,values in h['presets'].items():
  case(preset,'preset',values)
  for mode,n in [('few',12),('basic',8),('all',4)]:
   for _ in range(n):
    c=dict(values);moved=keys if mode=='all' else basic if mode=='basic' else list(rng.choice(keys,int(rng.integers(2,7)),replace=False))
    for k in moved:
     lo,hi=bounds[preset][k]['min'],bounds[preset][k]['max'];x=rng.uniform(lo,hi) if mode!='few' else np.clip(values[k]+rng.normal(0,.3)*(hi-lo),lo,hi);c[k]=published_value(bounds[preset][k],x)
    case(preset,mode,c)
  print('REFERENCE_SOURCE_PRESET_CASES_READY',preset,len(cases),flush=True)
 if args.extra_cases:
  for row in json.loads(Path(args.extra_cases).read_bytes()):case(row['preset'],row['mode'],row['controls'])
 case_path=folder/'cases.json';case_path.write_text(json.dumps(dict(runtime=str(Path(args.runtime).resolve()),runtimeSha256=sha(args.runtime),catalogSha256=args.catalog_sha256,cases=cases),indent=2)+'\n');client_report=folder/'actual-client.json'
 subprocess.run(['node',args.client_evaluator,str(case_path),str(client_report)],check=True)
 client=json.loads(client_report.read_bytes());assert client['runtimeSha256']==sha(args.runtime) and client['catalogSha256']==args.catalog_sha256 and client['inputCasesSha256']==sha(case_path) and client['clientEvaluatorSha256']==sha(args.client_evaluator) and client['actualClientParserAndEvaluator'] and len(client['rows'])==len(cases)
 limits={'preset_mm':.5,'few':5.,'basic':10.,'all':20.};quality={};extras=[]
 for preset in h['presets']:
  rows=[r for r in client['rows'] if r['preset']==preset];per={}
  for mode in ('preset','few','basic','all'):
   values=[r['maxErrorMm'] for r in rows if r['mode']==mode];assert len(values)>0 and all(np.isfinite(values))
   per['preset_mm' if mode=='preset' else mode]=max(values) if mode=='preset' else dict(max_mm=max(values),median_mm=float(np.median(values)),p90_mm=float(np.percentile(values,90)),n=len(values))
  quality[preset]=per;print('REFERENCE_INDEPENDENT_CHECK',preset,json.dumps(per),flush=True)
 passed=all(q['preset_mm']<=limits['preset_mm'] and all(q[m]['max_mm']<=limits[m] for m in ('few','basic','all')) for q in quality.values())
 report=dict(runtime=args.runtime,runtimeSha256=sha(args.runtime),catalogSha256=args.catalog_sha256,source=h['source'],rules=h['rules'],encoding=h['encoding'],seed=args.seed,quality=quality,publishedControls=bounds,geometry=geometry,limits_mm=limits,size=dict(bytes=len(raw),limitBytes=32*1024*1024),passed=bool(passed),actualClientParserAndEvaluator=True,clientEvaluatorSha256=sha(args.client_evaluator),clientReport=dict(path=str(client_report),sha256=sha(client_report)),algorithmSha256=sha(__file__))
 Path(args.report).write_text(json.dumps(report,indent=2)+'\n');print('REFERENCE_CHECK_COMPLETE',passed,flush=True)
if __name__=='__main__':main()
