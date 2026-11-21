"""Private RC4 worker. Main API binds requests to its current library BVH hash."""
import json, subprocess, os
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from . import service
app = FastAPI()

@app.get("/healthz")
def health(): return {"alive": True}

@app.get("/refine/release/ready")
def ready():
    try: service.settings()
    except Exception: raise HTTPException(503, "worker not ready")
    return {"ready": True, "version": service.VERSION}

@app.post("/refine/release")
def execute(image: UploadFile = File(...), pose_id: str = Form(...), source_sha256: str = Form(...), original_bvh: UploadFile = File(...)):
    try:
        cfg = service.settings()
        from .assets import MAX_BVH_BYTES, digest
        original = original_bvh.file.read(MAX_BVH_BYTES + 1)
        if len(original) > MAX_BVH_BYTES or digest(original) != source_sha256:
            raise ValueError('original BVH hash mismatch')
        result=service.execute(pose_id, image.file.read(12*1024*1024+1), bvh_bytes=original)
        if result['status']=='accepted':
            job=Path(cfg['REFINE_RELEASE_STATE'])/result['handle']
            with (job/'preview.log').open('w') as log:
                subprocess.run([cfg['blender'],'--background','--factory-startup','--python',str(Path(__file__).with_name('preview_blender.py')),'--',str(job/'result.fbx'),str(job/'preview.glb')],check=True,timeout=45,stdout=log,stderr=subprocess.STDOUT)
            result['preview_sha256']=service.sha(job/'preview.glb')
            (job/'result.json').write_text(json.dumps(result))
        return result
    except service.Busy: raise HTTPException(409, "worker busy")
    except (KeyError, ValueError, StopIteration): raise HTTPException(422, "invalid or mismatched original")
    except Exception: raise HTTPException(503, "worker failed; no artifact released")

@app.get("/refine/release/{handle}/fbx")
def artifact(handle: str):
    try: path = service.artifact(handle)
    except Exception: raise HTTPException(404, "accepted artifact not found")
    return FileResponse(path, media_type="application/octet-stream", filename="refined.fbx")

@app.get("/refine/release/{handle}/preview")
def preview(handle: str):
    try:
        fbx=service.artifact(handle);path=fbx.with_name('preview.glb')
        record=json.loads(fbx.with_name('result.json').read_text())
        if service.sha(path)!=record['preview_sha256']:raise ValueError()
    except Exception:raise HTTPException(404,"verified preview not found")
    return FileResponse(path,media_type="model/gltf-binary")
