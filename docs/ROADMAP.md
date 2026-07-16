# 로드맵 — 우선순위와 다음 액션

> 클라이언트 `10_THIS_WEEK_PLAN.md`의 서버판. 스캐폴드는 서 있고(mock end-to-end 동작),
> 다음은 **실데이터·실모델 승격**이다. 근거는 `docs/SEARCH_EVAL_2026-07-14.md`(실데이터 14컷 평가).

---

## 1. 현재 상태

- ✅ VLM→검색 파이프라인 **mock으로 end-to-end 동작**(`run_demo.py` 6케이스, `test_smoke.py`).
- ✅ API 계약 확정: `/analyze`·`/pose/{id}/bvh`·`/export-order`·`/healthz`(`docs/API_CONTRACT.md`).
- ✅ 실 라이브러리 빌드 경로 구현(`BVH_DIR` → SQLite 77 poses / 308 projections로 1차 검증).
- ✅ 검색 자체는 동작 확인 — 깨끗한 쿼리에서 카테고리 변별됨(`SEARCH_EVAL` §3).
- ⚠ 실사용 치명 버그 1건 미수정(cv2 한글 경로), 라이브러리 편중, 통합 경계 미확정.

---

## 2. 우선순위

### P0 — 실사용을 막는 것

- ✅ **[버그] cv2 한글 경로** — `fd67c31`에서 `src/pose.py::_load_bgr`가 `cv2.imdecode(np.fromfile(...))`로 해소됨(`SEARCH_EVAL` P1은 머지 이전 스냅샷). 잔여 죽은 코드도 정리 완료.
- ✅ **[통합] 앱 서버 경계** — 얇은 앱 서버(BFF) 분리로 결정(`DECISIONS.md` 결정 4). 인증·Job·오류봉투·`/v1`은 BFF, 추론 서버는 순수 추론 유지. 남은 확인: BFF를 누가/어느 레포에 만들지 + 도입 시점(실인증 필요 시).

### P1 — 품질/신뢰도

- **[데이터] 라이브러리 다양화** — 77개가 서기·앉기 편중. 격투·눕기·크게 벌린 동작 등 실루엣이 뚜렷이 다른 포즈 보강 후 DB 재빌드·14컷 재평가(`SEARCH_EVAL` P2).
- **[검수] 컨택트 시트로 편중 확인** — `scripts/bvh_contact_sheet.py`로 실제 포즈 훑어 apex 편중/중복 확인(P4).
- **[임계값] 실데이터 보정** — `FALLBACK_DISTANCE`(현 0.45)·`MIN_SKELETON_SCORE`를 관측값으로 튜닝.

### P2 — 실모델 승격

- **[VLM] Gemini 연결** — `GEMINI_API_KEY`만 있으면 동작(`src/vlm/client.py::GeminiVLMClient`).
- **[포즈] rtmlib Body 연결** — `POSE_BACKEND=rtmlib`(`src/pose.py::RTMPoseModel`, 이미 구현).
- **[검출] 별도 검출기** — YOLO/RTMDet 연결(`src/detect.py::RTMLibDetector`, TODO). 추출 실패 컷(#7) 대응.
- **[BVH] 실 파서** — `src/library.py::load_bvh_pose`에 BVH 파서(bvhio) + 관절명→COCO17 매핑.

---

## 3. 권장 순서

```text
1. [P0] cv2 한글 경로 수정        → 한글 컷 즉시 실행 가능
2. [P0] 앱 서버 경계 확정(팀)      → 계약 형상 고정
3. [P1] 컨택트 시트로 편중 확인    → 데이터 문제 범위 확정
4. [P1] 라이브러리 다양화·재평가   → 검색 품질 상향
5. [P2] 실모델 어댑터 승격         → mock 탈출
```

`SEARCH_EVAL` §5 "다음 액션"과 정렬. 재현 커맨드는 그 문서 §6 참조.

---

## 4. 계약 완료 조건 (앱 서버 팀 관점)

이 서버가 "연동 준비됨"이 되려면:

- `/analyze`가 실 이미지에서 인물별 Top-K를 반환(mock 아님).
- `route`·`count_confidence`·폴백 신호가 실데이터에서 의미 있게 동작.
- `/pose/{id}/bvh`가 실 BVH 파일 반환(409 합성 단계 탈출).
- `matchLevel` 라벨 산출 책임 확정(서버 신호 전달 vs 앱 어댑터 — `API_CONTRACT.md` §8-5).
- 오류 형식·인증 경계 확정(§8-1).

---

## 5. 보류 (이번 범위 밖)

- 얽힘 세트(hug/fight) 검색·제작 — 비얽힘 2인이 MVP 우선(`EXPORT_CONTRACT.md` §3-3 참고).
- 벡터 DB 전환(pgvector/LanceDB) — SQLite로 충분, 스케일 시 `repo.py`만 교체.
- Job 비동기·진행 단계 스트리밍 — 추론이 짧으면 앱 서버가 감싼다. 길어지면 재검토(`API_CONTRACT.md` §8-2).
- CSP 미러링·축 보정 — 동원(내보내기) 책임, 이 서버 범위 밖(`DECISIONS.md` 결정 3).
