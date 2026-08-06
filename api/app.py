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

import hashlib
import io
import json
import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, Response

from src.config import CFG
from src.pipeline import Pipeline
from src.library import build_synthetic_index
from src.library_source import ensure_library
from src.refine import refine_bvh
from src.repo import (FEATURE_VERSION, build_db, load_entries,
                      get_bvh_path, get_pose_meta)
from src.thumbnails import THUMBNAIL_VIEWS, find_thumbnail, thumbnail_url
from src.runtime_guard import (
    MockBackendError,
    actual_backend_names,
    ensure_production_backends,
)
from api.models import (CutResultOut, PersonOut, CandidateOut, SkeletonOut,
                        ImageInfoOut, InferenceMetadataOut,
                        RefineRequest, RefineResponse,
                        ExportOrderRequest, ExportOrder, ExportItem)

DB_PATH = os.getenv("DB_PATH", "data/poses.db")
REFINE_DIR = os.path.join(CFG.data_dir, CFG.refine_dir)

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


def _check_backends(pipeline: Pipeline) -> None:
    """프로덕션에서 실제로 생성된 mock 백엔드로 뜨는 것을 막는다."""
    try:
        ensure_production_backends(
            pipeline,
            is_production=CFG.is_production,
            requested_vlm=CFG.vlm_provider,
            requested_pose=CFG.pose_backend,
        )
    except MockBackendError as exc:
        raise StartupError(str(exc)) from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    entries = _ensure_db()                      # 1회 로드
    pipeline = Pipeline(entries)                # VLM/검출/포즈 팩토리도 1회 초기화
    _check_backends(pipeline)                   # 팩토리 폴백 후 실제 인스턴스 검사
    actual_vlm, actual_pose = actual_backend_names(
        pipeline, CFG.vlm_provider, CFG.pose_backend
    )
    STATE["pipeline"] = pipeline
    STATE["db_path"] = DB_PATH
    STATE["pose_count"] = len(entries)
    STATE["provider"] = actual_vlm
    STATE["pose_backend"] = actual_pose
    print(f"[startup] 준비 완료 — 포즈 {len(entries)}개, env={CFG.app_env}, "
          f"vlm={actual_vlm}, pose={actual_pose}")
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
        "provider": STATE.get("provider", CFG.vlm_provider),
        "pose_backend": STATE.get("pose_backend", CFG.pose_backend),
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
        skel = desc.skeleton
        raw_scores = None
        if desc.raw_scores is not None:
            raw = np.asarray(desc.raw_scores, dtype=float).reshape(-1)
            if raw.shape == (17,) and np.isfinite(raw).all():
                raw_scores = raw.tolist()
        people.append(PersonOut(
            index=i,
            box=desc.box.as_list() if desc.box else None,
            tags=desc.tag_dict(),
            skeleton=(SkeletonOut(
                keypoints=desc.skeleton.keypoints.tolist(),
                scores=desc.skeleton.scores.tolist(),
            ) if desc.skeleton is not None else None),
            candidates=[CandidateOut(
                pose_id=c.pose_id, view=c.view.value, distance=c.distance,
                tags=c.tags, rerank_score=c.rerank_score,
                bvh_url=f"/pose/{c.pose_id}/bvh",
                thumbnail_url=thumbnail_url(CFG.data_dir, c.pose_id, c.view.value),
            ) for c in cands],
            # 이미 추출한 스켈레톤을 실어 보낸다(연산 추가 0) → /refine이 순수 함수가 된다.
            keypoints=np.asarray(skel.keypoints, dtype=float).reshape(-1, 2).tolist()
                      if skel is not None else None,
            scores=np.asarray(skel.scores, dtype=float).reshape(-1).tolist()
                   if skel is not None else None,
            raw_scores=raw_scores,
            confidence=(res.person_confidence[i]
                        if i < len(res.person_confidence) else "low"),
            skeleton_state=desc.skeleton_state,
            skeleton_source=desc.skeleton_source,
            coverage_class=desc.coverage_class,
            slot_origin=desc.slot_origin,
            search_stability=desc.search_stability,
            distance_metric=desc.distance_metric,
            rank_distance=desc.rank_distance,
            confidence_threshold=desc.confidence_threshold,
            valid_limbs=list(desc.valid_limbs),
            refinable_limbs=list(desc.refinable_limbs),
            refine_allowed=desc.refine_allowed,
            quality_trace=desc.quality_trace,
            quality_reasons=desc.quality_reasons,
        ))
    vlm_model = (CFG.gemini_model if STATE.get("provider") == "gemini"
                 else CFG.openai_model if STATE.get("provider") == "openai"
                 else "mock")
    return CutResultOut(
        route=res.route, count_confidence=res.count_confidence,
        detector_count=res.detector_count, vlm_count=res.vlm_count,
        people=people, notes=res.notes,
        image=ImageInfoOut(width=w, height=h),
        inference_metadata=InferenceMetadataOut(
            deployment_version=CFG.deployment_version,
            vlm_provider=STATE.get("provider", CFG.vlm_provider),
            vlm_model=vlm_model,
            pose_backend=STATE.get("pose_backend", CFG.pose_backend),
            pose_model_version=os.getenv("POSE_MODEL_VERSION", "runtime-default"),
            pose_library_version=CFG.pose_library_version,
            feature_version=FEATURE_VERSION,
        ),
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


@app.get("/pose/{pose_id}/thumbnail")
def get_pose_thumbnail(pose_id: str, view: str):
    """번들에 포함된 후보 시점 PNG를 반환한다."""
    if view not in THUMBNAIL_VIEWS:
        raise HTTPException(400, f"unsupported thumbnail view: {view}")
    if get_pose_meta(STATE["db_path"], pose_id) is None:
        raise HTTPException(404, f"unknown pose_id: {pose_id}")

    path = find_thumbnail(CFG.data_dir, pose_id, view)
    if path is None:
        raise HTTPException(404, f"thumbnail not found: pose_id={pose_id}, view={view}")
    return FileResponse(
        path,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=86400"},
    )


# ---- 포즈 미세조정 (docs/REFINE_DESIGN.md) --------------------------------

def _refine_handle(req: RefineRequest) -> str:
    """같은 입력 → 같은 파일. 작가가 같은 후보를 다시 눌러도 재계산하지 않는다."""
    kp = np.asarray(req.keypoints, dtype=float).round(2).tobytes()
    sc = np.asarray(req.scores or [], dtype=float).round(3).tobytes()
    policy = (
        f"|allowed={req.refine_allowed}|limbs="
        f"{','.join(sorted(req.refinable_limbs or []))}|"
    ).encode()
    h = hashlib.sha1(f"{req.pose_id}|{req.view}|".encode() + policy + kp + sc)
    return h.hexdigest()[:16]


@app.post("/refine", response_model=RefineResponse)
def refine(req: RefineRequest):
    """
    작가가 고른 후보 1개를 러프에 맞춰 미세조정한다.

    입력은 /analyze 응답을 그대로 되돌려주면 된다(러프 이미지 재전송 불필요).
    **refined=False는 오류가 아니다** — 안전 게이트가 조정을 버리고 베이스를 준 것이며,
    이 경우 bvh_url은 원래의 /pose/{id}/bvh와 동등하다(§4-3 "좋아지거나, 그대로").
    """
    meta = get_pose_meta(STATE["db_path"], req.pose_id)
    if meta is None:
        raise HTTPException(404, f"unknown pose_id: {req.pose_id}")
    base = meta["bvh_path"]
    if not base or not os.path.exists(base):
        raise HTTPException(409, f"pose '{req.pose_id}' 등록됨(경로={base})이나 BVH 파일 미존재.")
    if len(req.keypoints) != 17:
        raise HTTPException(422, f"keypoints는 17개여야 합니다(받은 값: {len(req.keypoints)})")
    if req.scores is not None and len(req.scores) != 17:
        raise HTTPException(422, f"scores는 17개여야 합니다(받은 값: {len(req.scores)})")

    if req.refine_allowed is False:
        return RefineResponse(
            pose_id=req.pose_id, view=req.view, refined=False,
            reason="skeleton_policy", bvh_url=f"/pose/{req.pose_id}/bvh",
            loss_base=None, loss_final=None, gain=None, backend="none",
        )

    # 얽힘 세트는 조정하지 않는다. hug_01_A/B를 각자 돌리면 두 사람이 맞물리던
    # 정합(손이 어깨에 닿는 등)이 깨지는데, BVH는 상대 위치를 안 실으므로
    # 그 깨짐을 되돌릴 방법이 없다. 세트 refine은 세트 전체를 함께 푸는 별도 과제.
    if meta.get("set_id"):
        return RefineResponse(
            pose_id=req.pose_id, view=req.view, refined=False,
            reason="entangled_set", bvh_url=f"/pose/{req.pose_id}/bvh",
            loss_base=None, loss_final=None, gain=None, backend="none")

    handle = _refine_handle(req)
    out_path = os.path.join(REFINE_DIR, f"{handle}.bvh")
    try:
        res = refine_bvh(base, req.keypoints, req.scores, req.view,
                         out_path=out_path, search_distance=req.search_distance,
                         allowed_limbs=req.refinable_limbs)
    except ValueError as exc:                      # 알 수 없는 view 등
        raise HTTPException(422, str(exc)) from exc

    if res.refined:
        # 사이드카: handle만으로는 어떤 포즈인지 알 수 없다. 동원의 '파일명·소재 폴더
        # 규칙'(EXPORT_CONTRACT §4-3)과 export 주문서 연결에 pose_id가 필요하다.
        with open(os.path.join(REFINE_DIR, f"{handle}.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"handle": handle, "pose_id": req.pose_id, "view": req.view,
                       "base_bvh_url": f"/pose/{req.pose_id}/bvh",
                       "limbs": list(res.limbs),
                       "limb_decisions": res.limb_decisions,
                       "created_at": datetime.now(timezone.utc).isoformat(
                           timespec="seconds")}, f, ensure_ascii=False)

    return RefineResponse(
        pose_id=req.pose_id, view=req.view,
        refined=res.refined, reason=res.reason,
        bvh_url=(f"/refined/{handle}/bvh" if res.refined
                 else f"/pose/{req.pose_id}/bvh"),
        loss_base=None if np.isnan(res.loss_base) else res.loss_base,
        loss_final=None if np.isnan(res.loss_final) else res.loss_final,
        gain=None if np.isnan(res.loss_base) else res.gain,
        backend=res.backend,
        limbs=list(res.limbs),
        limb_decisions=res.limb_decisions,
    )


@app.get("/refined/{handle}/bvh")
def get_refined_bvh(handle: str):
    """
    조정본 BVH 다운로드. 동원 내보내기는 /pose/{id}/bvh와 동일하게 소비하면 된다
    (HIERARCHY가 원본 그대로라 CSP 미러링·축 보정 로직은 손댈 필요 없음).

    ⚠ 배포 주의: 이 파일은 **refine을 처리한 인스턴스의 로컬 디스크**에 있다.
      추론 서버를 태스크 2개 이상으로 띄우면 POST /refine과 이 GET이 다른 태스크에
      떨어져 404가 난다. 적용 전까지 추론 서버는 단일 태스크로 운영할 것.

      해소 방법은 정해졌다 — `POST /refine` 응답에 BVH 본문을 실어 이 GET 자체를
      없앤다(docs/REFINE_HANDOFF.md §3). 그 뒤 이 엔드포인트는 제거 대상이다.
    """
    if not handle.isalnum():                       # 경로 조작 차단
        raise HTTPException(400, "invalid handle")
    path = os.path.join(REFINE_DIR, f"{handle}.bvh")
    if not os.path.exists(path):
        raise HTTPException(404, f"unknown refine handle: {handle} (만료됐거나 미생성)")
    # 사이드카가 있으면 pose_id로 내려준다 — 동원의 파일명 규칙이 handle을 모른다.
    name = f"{handle}.refined.bvh"
    side = os.path.join(REFINE_DIR, f"{handle}.json")
    if os.path.exists(side):
        try:
            with open(side, encoding="utf-8") as f:
                name = f"{json.load(f)['pose_id']}.refined.bvh"
        except Exception:
            pass
    return FileResponse(path, media_type="application/octet-stream", filename=name)


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
