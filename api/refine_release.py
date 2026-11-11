"""Opt-in FBX endpoint; legacy /refine remains unchanged."""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
import os
import httpx

def worker_url():
    return os.getenv("REFINE_RELEASE_WORKER_URL", "").rstrip("/")

def remote(method, path, **kwargs):
    if os.getenv("REFINE_RELEASE_ENABLED") != "1":
        raise HTTPException(503, "release refine is disabled")
    try:
        response = httpx.request(method, worker_url()+path, timeout=310, **kwargs)
        response.raise_for_status()
        return response
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, "release worker rejected request")
    except httpx.HTTPError:
        raise HTTPException(503, "release worker unavailable")
from services.refine_release import service

router = APIRouter(prefix="/refine/release", tags=["release-refine"])

@router.get("/ready")
def ready():
    if worker_url(): return remote("GET", "/refine/release/ready").json()
    try: service.settings()
    except Exception: raise HTTPException(503, "release refine is not configured or is disabled")
    return {"ready": True, "version": service.VERSION}

@router.post("")
def refine_release(image: UploadFile = File(...), pose_id: str = Form(...)):
    from api.app import STATE
    from src.repo import get_bvh_path
    if not worker_url():
        try: service.settings()
        except Exception: raise HTTPException(503, "release refine is unavailable")
    bvh = get_bvh_path(STATE["db_path"], pose_id)
    if not bvh: raise HTTPException(404, "unknown pose_id")
    data = image.file.read(12*1024*1024+1)
    if len(data) > 12*1024*1024: raise HTTPException(413, "image exceeds limit")
    if worker_url():
        return remote("POST", "/refine/release", data={"pose_id":pose_id, "source_sha256":service.sha(bvh)}, files={"image":("rough.png",data,"image/png")}).json()
    try: return service.execute(pose_id, data, expected_bvh=bvh)
    except service.Busy: raise HTTPException(409, "release worker is busy; retry later")
    except (ValueError, KeyError): raise HTTPException(422, "invalid image or unregistered original asset")
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Release refine failed")
        raise HTTPException(503, "release processing failed; no artifact released")

@router.get("/{handle}/fbx")
def download(handle: str):
    if len(handle)!=32 or any(c not in "0123456789abcdef" for c in handle): raise HTTPException(404, "unknown handle")
    if worker_url():
        from fastapi import Response
        return Response(remote("GET", f"/refine/release/{handle}/fbx").content, media_type="application/octet-stream", headers={"Content-Disposition":"attachment; filename=refined.fbx"})
    try: path = service.artifact(handle)
    except service.Unavailable: raise HTTPException(503, "release refine is unavailable")
    except Exception: raise HTTPException(404, "accepted artifact not found")
    return FileResponse(path, media_type="application/octet-stream", filename="refined.fbx")

@router.get("/{handle}/preview")
def preview(handle: str):
    if len(handle)!=32 or any(c not in "0123456789abcdef" for c in handle) or not worker_url(): raise HTTPException(404,"unknown preview")
    from fastapi import Response
    return Response(remote("GET", f"/refine/release/{handle}/preview").content,media_type="model/gltf-binary")
