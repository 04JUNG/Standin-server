"""
FastAPI 레이어 — 도원의 Python 추론 서버.

경계: [앱 서버 팀] --HTTP--> [이 서비스]  (문서화된 OpenAPI 계약 = /docs)
  POST /analyze         멀티파트 PNG 러프 컷 → CutResult(JSON)
  GET  /pose/{id}/bvh   후보 pose_id → 라이브러리 BVH 파일(동원 내보내기 팀이 소비)
  GET  /healthz         기동 확인

성능 원칙:
  · 인덱스/모델은 lifespan에서 1회 로드(요청마다 X).
  · /analyze는 `def`(동기)로 두어 FastAPI가 threadpool에서 실행 → 블로킹 포즈추론이
    이벤트 루프를 막지 않게 한다(VLM은 I/O라 async 이득이 있으나, MVP는 이 방식이 가장 단순·안전).
"""
from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, Response

from src.config import CFG
from src.pipeline import Pipeline
from src.library import build_synthetic_index
from src.repo import build_db, load_entries, get_bvh_path, get_pose_meta
from api.models import (CutResultOut, PersonOut, CandidateOut,
                        ExportOrderRequest, ExportOrder, ExportItem)

DB_PATH = os.getenv("DB_PATH", "data/poses.db")

STATE: dict = {}


def _ensure_db():
    if not os.path.exists(DB_PATH):
        build_db(build_synthetic_index(), DB_PATH)
    return load_entries(DB_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    entries = _ensure_db()                      # 1회 로드
    STATE["pipeline"] = Pipeline(entries)       # VLM/검출/포즈 팩토리도 1회 초기화
    STATE["db_path"] = DB_PATH
    yield
    STATE.clear()


app = FastAPI(title="Standin Pose Pipeline", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    ok = "pipeline" in STATE
    return {"ok": ok, "provider": CFG.vlm_provider, "pose_backend": CFG.pose_backend}


def _load_image(data: bytes):
    """PNG bytes → (image, w, h). PIL 없으면 원본 bytes와 기본 크기로 폴백."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img, img.width, img.height
    except Exception:
        return data, 512, 768


class _HintImg(str):
    """mock provider용: dev에서 hint 문자열로 분석을 유도(실모델이면 무시됨)."""
    @property
    def hint(self): return str(self)


@app.post("/analyze", response_model=CutResultOut)
def analyze(file: UploadFile = File(...), hint: str = Form(default="")):
    data = file.file.read()
    image, w, h = _load_image(data)
    # dev 편의: mock provider일 때만 hint를 이미지 대용으로 사용
    if CFG.vlm_provider == "mock" and hint:
        image = _HintImg(hint)

    pipe: Pipeline = STATE["pipeline"]
    res = pipe.process_cut(image, w, h)

    people = []
    per_person = getattr(res, "person_candidates", [])
    for i, desc in enumerate(res.descriptors):
        cands = per_person[i] if i < len(per_person) else []
        people.append(PersonOut(
            index=i,
            box=desc.box.as_list() if desc.box else None,
            tags=desc.tag_dict(),
            candidates=[CandidateOut(
                pose_id=c.pose_id, view=c.view.value, distance=c.distance,
                tags=c.tags, rerank_score=c.rerank_score,
                bvh_url=f"/pose/{c.pose_id}/bvh",
            ) for c in cands],
        ))
    return CutResultOut(
        route=res.route, count_confidence=res.count_confidence,
        detector_count=res.detector_count, vlm_count=res.vlm_count,
        people=people, notes=res.notes,
    )


@app.get("/pose/{pose_id}/bvh")
def get_pose_bvh(pose_id: str):
    """동원 핸드오프: 선택된 후보의 라이브러리 BVH 원본을 그대로 반환.
    CSP용 미러링/축 보정은 이 서비스가 아니라 동원의 내보내기 단계가 담당(결정 문서 참조)."""
    path = get_bvh_path(STATE["db_path"], pose_id)
    if not path:
        raise HTTPException(404, f"unknown pose_id: {pose_id}")
    if not os.path.exists(path):
        # 합성 단계: 실제 BVH 파일이 아직 없음 → 계약 확인용 플레이스홀더
        raise HTTPException(
            409, f"pose '{pose_id}' 등록됨(경로={path})이나 BVH 파일 미존재. "
                 f"실 라이브러리 빌드 전 단계.")
    return FileResponse(path, media_type="application/octet-stream",
                        filename=f"{pose_id}.bvh")


@app.post("/export-order", response_model=ExportOrder)
def export_order(req: ExportOrderRequest):
    """작가 선택 → 동원 Export 주문서. DB에서 bvh_url·set_id·tags로 강화한다.
    /analyze와 별개 계약: 여기 오는 건 'Top5 중 고른 하나'들이다."""
    db = STATE["db_path"]
    items, notes = [], []
    for sel in req.selections:
        meta = get_pose_meta(db, sel.pose_id)
        if meta is None:
            raise HTTPException(404, f"unknown pose_id: {sel.pose_id}")
        # BVH는 1인 → 선택 1개 = item 1개(1인 BVH 1개). 얽힘은 set_id로 묶어만 둔다.
        items.append(ExportItem(
            person_index=sel.person_index, pose_id=sel.pose_id,
            bvh_url=f"/pose/{sel.pose_id}/bvh", view=sel.view,
            set_id=meta["set_id"], set_role=meta["set_role"],
            tags={"shot": meta["shot"], "action": meta["action"],
                  "relationship": meta["relationship"], "view": sel.view},
        ))
    set_ids = {it.set_id for it in items if it.set_id}
    for sid in set_ids:
        notes.append(f"set_id='{sid}': 한 상호작용의 1인 BVH들 → 상대 위치는 작가가 CSP에서 조정.")
    return ExportOrder(
        cut_id=req.cut_id, source_image=req.source_image,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        items=items, notes=notes,
    )
