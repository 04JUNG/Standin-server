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
def execute(image: UploadFile = File(...), pose_id: str = Form(...), source_sha256: str = Form(...)):
    try:
        cfg = service.settings()
        entry = json.loads(Path(cfg["REFINE_RELEASE_REGISTRY"]).read_text())[pose_id]
        root = Path(entry["assets"])
        manifest = root / "input/manifest.json"
        if service.sha(manifest) != entry["manifest_sha256"]: raise ValueError()
        row = next(r for r in json.loads(manifest.read_text())["rows"] if r["case_id"] == entry["case"])
        original = service.inside(root, "input/"+row["selection"]["verified_selected_bvh"])
        if service.sha(original) != source_sha256: raise ValueError()
        result=service.execute(pose_id, image.file.read(12*1024*1024+1), expected_bvh=original)
        if result['status']=='accepted':
            job=Path(cfg['REFINE_RELEASE_STATE'])/result['handle']
            with (job/'preview.log').open('w') as log:
                subprocess.run([cfg['blender'],'--background','--factory-startup','--python',str(Path(__file__).with_name('preview_blender.py')),'--',str(job/'result.fbx'),str(job/'preview.glb')],check=True,timeout=45,stdout=log,stderr=subprocess.STDOUT)
            result['preview_sha256']=service.sha(job/'preview.glb')
            (job/'result.json').write_text(json.dumps(result))
        return result
    except service.Busy: raise HTTPException(409, "worker busy")
    except (KeyError, ValueError, StopIteration): raise HTTPException(422, "unregistered or mismatched original")
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
