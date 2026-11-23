"""Isolated local review origin; no production credentials or user documents."""
import json,os,sys,subprocess,uuid,time,hashlib
from pathlib import Path
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import argparse
p=argparse.ArgumentParser();p.add_argument("--candidate",type=Path,required=True);p.add_argument("--web-root",type=Path,required=True);p.add_argument("--port",type=int,default=8927);p.add_argument("--blender",default="/Applications/Blender.app/Contents/MacOS/Blender");args=p.parse_args()
REPO=Path(__file__).resolve().parents[2]
CANDIDATE=args.candidate.resolve()
WEB=args.web_root.resolve()
BASE=CANDIDATE.parent
JOBS=CANDIDATE/'review-jobs';JOBS.mkdir(exist_ok=True)
sys.path.insert(0,str(REPO))
os.environ['CONVERTER_SHAPE_REGISTRY']=str(CANDIDATE/'release/shape-bakes.local.json')
from converter.shape_spec import resolve_shape,digest_shape,load_registry
app=FastAPI()
catalog=json.load(open(CANDIDATE/'release/body-models.local.json'))
registry=load_registry()

@app.get('/api/catalog')
def get_catalog():return catalog

@app.get('/api/runtime')
def runtime():return FileResponse(CANDIDATE/'preview-full.ssr',media_type='application/octet-stream')

@app.post('/api/export')
def export(data:dict):
 cid=data.get('characterId');raw=registry['characters'].get(cid)
 if raw is None:raise HTTPException(400,'unknown review character')
 controls={c['key']:c['default'] for c in raw['controls']};controls.update(data.get('controls',{}))
 envelope={'schemaVersion':'body-shape.v1','controls':controls,'shapeSha256':digest_shape(cid,raw['asset_sha256'],controls)}
 body=next(a for a in catalog['assets'] if a['characterId']==cid)
 recipe=resolve_shape(cid,raw['asset_sha256'],envelope,body['shapeSource'])
 dest=JOBS/uuid.uuid4().hex;dest.mkdir();job=dest/'job.json';job.write_text(json.dumps({'recipe':recipe}))
 started=time.monotonic()
 with (dest/'worker.log').open('w') as log:
  run=subprocess.run([args.blender,'-b','--disable-autoexec','--python-exit-code','1','--python',str(REPO/'scripts/female_breast/qa_export.py'),'--','--job',str(job)],stdout=log,stderr=subprocess.STDOUT)
 if run.returncode:raise HTTPException(500,'review export failed; inspect worker.log')
 result=json.load(open(dest/'result.json'));result['seconds']=time.monotonic()-started;result['downloadUrl']='/api/download/'+dest.name;result['jobId']=dest.name
 (dest/'http-result.json').write_text(json.dumps(result,indent=2))
 return result

@app.get('/api/download/{job_id}')
def download(job_id:str):
 if len(job_id)!=32 or any(c not in '0123456789abcdef' for c in job_id):raise HTTPException(404)
 path=JOBS/job_id/'download.fbx'
 if not path.is_file():raise HTTPException(404)
 receipt=json.load(open(path.parent/'result.json'))
 (path.parent/'download-served.json').write_text(json.dumps({'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'timestamp':time.time()}))
 return FileResponse(path,media_type='application/octet-stream',filename='female040-breast7-4-size-'+str(receipt['controls']['breast_size'])+'.fbx',headers={'X-Artifact-SHA256':receipt['fbx_sha256']})

@app.post('/api/observe')
def observe(data:dict):
 with (CANDIDATE/'review-ui-events.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),**data})+'\n')
 return {'ok':True}

app.mount('/',StaticFiles(directory=WEB,html=True))

if __name__=="__main__":
 import uvicorn
 uvicorn.run(app,host="127.0.0.1",port=args.port)
