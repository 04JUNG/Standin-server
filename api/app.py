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
import json
import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, Response

from src.config import CFG
from src.pipeline import Pipeline
from src.library import build_synthetic_index
from src.library_source import ensure_library
from src.repo import build_db, load_entries, get_bvh_path, get_pose_meta
from api.models import (CutResultOut, PersonOut, CandidateOut,
                        ExportOrderRequest, ExportOrder, ExportItem)

DB_PATH = os.getenv("DB_PATH", "data/poses.db")

STATE: dict = {}


class StartupError(RuntimeError):
    """설정이 잘못돼 안전하게 서비스할 수 없다. 기동을 중단한다."""


def _ensure_db():
    """포즈 라이브러리를 준비한다.

    개발과 프로덕션의 정책이 다르다.
      · 개발  — 라이브러리가 없으면 합성으로 만들어 바로 띄운다(오프라인 편의).
      · 프로덕션 — 합성으로 대체하지 않는다. 라이브러리가 없으면 기동을 실패시킨다.
        가짜 후보를 정상처럼 서빙하면 작가가 잘못된 포즈를 받고도 알 수 없다
        (CLAUDE.md §10: 구현하지 않은 것을 작동하는 것처럼 보이게 하지 않는다).
    """
    fetched = ensure_library(CFG.data_dir, DB_PATH, CFG.pose_library_uri or None)
    if fetched:
        print(f"[startup] 포즈 라이브러리를 받았습니다: {CFG.pose_library_uri}")

    if not os.path.exists(DB_PATH):
        if CFG.is_production:
            raise StartupError(
                f"포즈 라이브러리가 없습니다(DB_PATH={DB_PATH}). "
                "POSE_LIBRARY_URI로 번들 위치를 지정하거나 볼륨으로 마운트하세요. "
                "프로덕션에서는 합성 라이브러리로 대체하지 않습니다."
            )
        print(f"[startup] {DB_PATH} 없음 → 합성 라이브러리 생성(개발 모드)")
        build_db(build_synthetic_index(), DB_PATH)

    return load_entries(DB_PATH)


def _check_backends() -> None:
    """프로덕션에서 mock 백엔드로 뜨는 것을 막는다."""
    if not CFG.is_production:
        return
    mocked = [
        name
        for name, value in (("VLM_PROVIDER", CFG.vlm_provider), ("POSE_BACKEND", CFG.pose_backend))
        if value == "mock"
    ]
    if mocked:
        raise StartupError(
            f"프로덕션에서 mock 백엔드를 쓸 수 없습니다: {', '.join(mocked)}. "
            "실제 provider/backend를 지정하세요."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _check_backends()
    entries = _ensure_db()                      # 1회 로드
    STATE["pipeline"] = Pipeline(entries)       # VLM/검출/포즈 팩토리도 1회 초기화
    STATE["db_path"] = DB_PATH
    STATE["pose_count"] = len(entries)
    print(f"[startup] 준비 완료 — 포즈 {len(entries)}개, env={CFG.app_env}, "
          f"vlm={CFG.vlm_provider}, pose={CFG.pose_backend}")
    yield
    STATE.clear()


app = FastAPI(title="Standin Pose Pipeline", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    # 라이브러리가 비면 후보를 하나도 못 내므로 healthy로 보고하지 않는다.
    # ECS/ALB가 이 응답으로 태스크 교체를 판단한다.
    pose_count = STATE.get("pose_count", 0)
    ok = "pipeline" in STATE and pose_count > 0
    body = {
        "ok": ok,
        "env": CFG.app_env,
        "provider": CFG.vlm_provider,
        "pose_backend": CFG.pose_backend,
        "pose_count": pose_count,
    }
    return body if ok else Response(
        content=json.dumps(body), status_code=503, media_type="application/json"
    )


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
