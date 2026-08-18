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
import re
import time
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response

from src.config import CFG
from src.logging_setup import (configure_logging, log_error, log_info,
                               log_warn, request_id_var)
from src import notify as alerts
from src.ops_metrics import COLLECTOR, TASK_ID
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

# uvicorn이 자기 로깅을 세운 뒤 이 모듈을 import한다. 여기서 덮어써야 로그가
# JSON 한 종류로 남는다(그대로 두면 uvicorn 형식과 우리 형식이 섞인다).
configure_logging()

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
        log_info("pose_library", "포즈 라이브러리를 받았습니다",
                 source=CFG.pose_library_uri, libraryVersion=CFG.pose_library_version)

    if not os.path.exists(DB_PATH):
        if CFG.is_production:
            raise StartupError(
                f"포즈 라이브러리가 없습니다(DB_PATH={DB_PATH}). "
                "POSE_LIBRARY_URI로 번들 위치를 지정하거나 볼륨으로 마운트하세요. "
                "프로덕션에서는 합성 라이브러리로 대체하지 않습니다."
            )
        log_warn("pose_library", "라이브러리 없음 → 합성 라이브러리 생성(개발 모드)",
                 errorCode="LIBRARY_MISSING", dbPath=DB_PATH)
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
    try:
        entries = _ensure_db()                      # 1회 로드
        pipeline = Pipeline(entries)                # VLM/검출/포즈 팩토리도 1회 초기화
        _check_backends(pipeline)                   # 팩토리 폴백 후 실제 인스턴스 검사
    except Exception as exc:
        # 기동 실패는 컨테이너가 뜨지 않는다는 뜻이고, 그 상태에서는 앱이 더 이상
        # 아무것도 보고할 수 없다. 죽기 전에 동기로 한 번 보낸다.
        log_error("startup", "기동 실패", exc_info=True, errorCode="STARTUP_FAILED")
        alerts.notify_now(
            "P1", "STARTUP_FAILED",
            "추론 서버가 기동하지 못했습니다. 태스크가 반복 재시작합니다.",
            context={"env": CFG.app_env, "원인": type(exc).__name__, "상세": str(exc)[:300]},
        )
        raise

    actual_vlm, actual_pose = actual_backend_names(
        pipeline, CFG.vlm_provider, CFG.pose_backend
    )
    # 설정한 백엔드와 실제로 만들어진 백엔드가 다르면 조용한 폴백이다. production은
    # 위 _check_backends가 이미 막았으므로 여기 도달하면 개발 환경이다.
    if (actual_vlm, actual_pose) != (CFG.vlm_provider, CFG.pose_backend):
        log_warn("backend_fallback", "요청한 백엔드 대신 폴백으로 기동",
                 errorCode="BACKEND_FALLBACK",
                 requestedVlm=CFG.vlm_provider, actualVlm=actual_vlm,
                 requestedPose=CFG.pose_backend, actualPose=actual_pose)

    STATE["pipeline"] = pipeline
    STATE["db_path"] = DB_PATH
    STATE["pose_count"] = len(entries)
    STATE["provider"] = actual_vlm
    STATE["pose_backend"] = actual_pose
    log_info("startup", "준비 완료", poseCount=len(entries), env=CFG.app_env,
             vlm=actual_vlm, pose=actual_pose, libraryVersion=CFG.pose_library_version)
    alerts.notify(
        "P3", "STARTUP",
        f"추론 서버 기동 — 포즈 {len(entries)}개, vlm={actual_vlm}, pose={actual_pose}",
        context={"env": CFG.app_env, "version": CFG.deployment_version},
    )
    yield
    log_info("shutdown", "종료")
    STATE.clear()
    # 종료 신호 뒤에는 배치 창을 기다릴 수 없다. 버퍼에 남은 알림을 밀어낸다.
    alerts.flush()


app = FastAPI(title="Standin Pose Pipeline", version="0.1.0", lifespan=lifespan)


# BFF가 붙여 보내는 요청 ID. 로그 주입을 막기 위해 형식을 검사하고, 어긋나면 새로 만든다.
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def _route_pattern(request: Request) -> str:
    """라우트 **패턴**(`/pose/{pose_id}/bvh`)을 돌려준다.

    실제 경로를 쓰면 pose_id마다 다른 값이 되어 집계 카디널리티가 터진다.
    라우팅이 안 된 요청(404)은 경로 자체가 임의 값이므로 남기지 않는다.
    """
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


@app.middleware("http")
async def request_context(request: Request, call_next):
    """요청마다 requestId를 잡고, 끝날 때 http_request 한 줄을 남긴다.

    BFF가 `X-Request-Id`를 넘겨주므로 두 서비스의 로그가 같은 값으로 이어진다
    (마스터독스 「관측성」 §4). 응답 헤더로 되돌려주어 호출측도 확인할 수 있게 한다.
    """
    incoming = request.headers.get("X-Request-Id", "")
    request_id = incoming if _REQUEST_ID_RE.match(incoming) else f"req_{uuid.uuid4()}"
    token = request_id_var.set(request_id)
    started = time.monotonic()
    try:
        try:
            response = await call_next(request)
        except Exception as exc:
            route = _route_pattern(request)
            log_error("unhandled_error", "처리되지 않은 예외", exc_info=True,
                      route=route, method=request.method, errorCode="INTERNAL_ERROR")
            alerts.notify(
                "P2", "UNHANDLED_ERROR",
                f"{request.method} {route} 처리 중 예외가 발생했습니다.",
                key=f"P2:unhandled:{route}",
                context={"예외": type(exc).__name__},
            )
            # HTTPException이 아닌 예외는 여기까지 온다. 계약대로 JSON으로 답한다.
            response = JSONResponse(status_code=500, content={"detail": "internal error"})

        duration_ms = round((time.monotonic() - started) * 1000)
        response.headers["X-Request-Id"] = request_id
        route = _route_pattern(request)
        # 헬스체크는 30초마다 온다. 정상 응답까지 남기면 로그의 대부분이 healthz가 된다.
        error_code = (f"HTTP_{response.status_code}"
                      if response.status_code >= 400 else None)
        if route != "/healthz" or response.status_code != 200:
            emit = log_warn if response.status_code >= 500 else log_info
            emit("http_request", "",
                 route=route, method=request.method,
                 status=response.status_code, durationMs=duration_ms,
                 errorCode=error_code)
        # 로그와 같은 자리에서 지표도 센다. 두 곳에서 세면 반드시 어긋난다(계획 3단계).
        # 지표에는 healthz도 넣는다 — 로그는 시끄러워서 뺐지만 가용성 계산에는 필요하다.
        COLLECTOR.record(time.time(), response.status_code, duration_ms,
                         route=route, error_code=error_code)
        return response
    finally:
        request_id_var.reset(token)


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


@app.get("/ops/metrics")
def ops_metrics():
    """분 단위 롤업을 그대로 내보낸다(계획 3단계). BFF가 1분마다 긁어 RDS에 넣는다.

    ⚠ 내부 전용이다. 이 서비스는 무인증이라 ALB에 붙지 않으며 Cloud Map 내부 DNS로만
      닿는다(README의 공개 경계 원칙). 여기에 개인정보는 담지 않는다 — 카운트와
      라우트 패턴뿐이다.

    버킷을 내보낸 뒤 지우지 않는다. BFF가 쓰다 실패하면 그 분이 통째로 사라지기 때문이다.
    BFF 저장이 upsert라 같은 값을 다시 읽어도 결과가 같다.
    """
    return {"service": "inference", "taskId": TASK_ID, "buckets": COLLECTOR.snapshot()}


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
#
# handle 기반 로컬 캐시는 제거됐다(REFINE_HANDOFF §3 4단계). 같은 선택을 다시
# 눌러도 추론을 재호출하지 않는 멱등성은 BFF의 refined_artifacts PK
# (job_id, person_index, candidate_id)가 담당한다.


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

    try:
        # out_path=None → 로컬 파일을 쓰지 않는다. 조정본은 응답 본문으로만 나간다.
        res = refine_bvh(base, req.keypoints, req.scores, req.view,
                         out_path=None, search_distance=req.search_distance,
                         allowed_limbs=req.refinable_limbs)
    except ValueError as exc:                      # 알 수 없는 view 등
        raise HTTPException(422, str(exc)) from exc

    # 조정본은 응답 본문(bvh)으로만 나간다. 소비자가 두 번째 요청을 하지 않으므로
    # 롤링 배포 중 다른 태스크가 응답해 404가 나는 경로가 없고, 로컬 디스크에
    # 조정본이 무한정 쌓이지도 않는다(REFINE_HANDOFF §3).
    #
    # bvh_url은 refined 여부와 무관하게 항상 베이스를 가리킨다. 조정본에는 더 이상
    # URL이 없다 — 소비자는 bvh를 받아 자기 저장소에 보관한다.
    return RefineResponse(
        pose_id=req.pose_id, view=req.view,
        refined=res.refined, reason=res.reason,
        bvh_url=f"/pose/{req.pose_id}/bvh",
        bvh=res.bvh_text,
        loss_base=None if np.isnan(res.loss_base) else res.loss_base,
        loss_final=None if np.isnan(res.loss_final) else res.loss_final,
        gain=None if np.isnan(res.loss_base) else res.gain,
        backend=res.backend,
        limbs=list(res.limbs),
        limb_decisions=res.limb_decisions,
    )


# GET /refined/{handle}/bvh는 제거됐다(REFINE_HANDOFF §3 4단계).
# 조정본은 POST /refine 응답의 bvh 필드로만 나가고, 보관은 소비자(BFF)가 한다.
# 추론 컨테이너의 로컬 디스크에는 아무것도 남지 않는다.


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
