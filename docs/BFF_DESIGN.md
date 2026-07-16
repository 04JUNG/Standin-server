# BFF 설계 — 앱 서버 (Python / FastAPI)

> `DECISIONS.md` 결정 4에서 **앱 서버(BFF) 분리**로 확정. 이 문서는 그 BFF를 **Python/FastAPI**로
> 어떻게 만들지 구체화한다. 언어는 Python으로 결정(팀 Python 툴체인·배포 인프라 재사용).
>
> **경계 원칙: BFF는 얇게.** 하는 일은 셋뿐이다 — ① 인증 ② 동기 추론을 비동기 Job으로 감싸기
> ③ 계약 번역(오류봉투·matchLevel). 추론 로직(VLM·검출·포즈·검색)은 **이 저장소(도원)**가 소유하고
> BFF는 HTTP로만 호출한다. 추론 서버는 손대지 않는다.

```
[Tauri] ──/v1──> [BFF: FastAPI] ──HTTP──> [도원 추론 서버]
                 인증·Job·기록·오류봉투       POST /analyze (동기·무인증)
                 matchLevel 매핑             GET  /pose/{id}/bvh
```

⚠ 이 BFF는 **별도 레포**(예: `Standin-bff`)에 만든다. 이 문서는 그 레포의 시드(설계 초안)이며, 결정의 단일 소스는 `DECISIONS.md` 결정 4다.

---

## 1. 스택

| 관심사 | 선택 | 비고 |
|---|---|---|
| 프레임워크 | **FastAPI** | 추론 서버와 동일 → CI·배포 패턴 재사용 |
| 추론 서버 호출 | **httpx**(async) | `POST /analyze` 멀티파트·`GET /pose/{id}/bvh` 프록시 |
| Job 큐 | **`BackgroundTasks` / `asyncio.create_task`** | MVP는 in-process. Redis/Celery 불필요(추론 수 초, 동시 사용자 적음) |
| Job·유저 저장 | **SQLite**(BFF 전용 파일) | ⚠ 추론 라이브러리 `poses.db`와 **다른 DB**. PII·쓰기 많음 |
| 인증 | **JWT access + refresh(회전)** | `fastapi-users` 또는 `python-jose`+`passlib` 수동 |
| 비밀번호 해시 | **argon2 / bcrypt** | 평문 저장·로그 금지 |
| 설정 | env 주입(`.env`) | 추론 서버 URL·JWT 시크릿·서비스 토큰 |

**승격 경로**: 동시성↑ 또는 추론이 길어지면 큐만 `arq`/RQ/Celery + Redis로 교체(인터페이스 유지). 유저 규모↑면 SQLite→Postgres.

---

## 2. 엔드포인트 매핑 (클라 `/v1` → BFF → 추론 서버)

| 클라 요청 | BFF 처리 | 추론 서버 호출 |
|---|---|---|
| `POST /v1/auth/login·refresh·logout` · `GET /v1/users/me` | 토큰 발급·검증·회전 | — (BFF 단독) |
| `POST /v1/analysis/jobs` (멀티파트) | Job 생성 → `jobId` 즉시 반환, 백그라운드로 추론 | (백그라운드) `POST /analyze` |
| `GET /v1/analysis/jobs/{id}` | 저장된 status 폴링 반환 | — |
| `GET /v1/analysis/jobs/{id}/result` | `CutResult` → 클라 `PoseCandidate`로 매핑(+matchLevel) | — |
| `POST /v1/analysis/jobs/{id}/rerun` | 새 Job, `excludeCandidateIds` 전달 | `POST /analyze` |
| `GET /v1/pose-candidates/{id}/export` | 인증 확인 후 프록시 | `GET /pose/{pose_id}/bvh` |

클라 계약 원본: `Standin-client/docs/08_API_CONTRACT.md`. 추론 계약 원본: 이 저장소 `docs/API_CONTRACT.md`.

---

## 3. 핵심 패턴 — 동기 추론을 Job으로 감싸기

추론은 동기(`/analyze` 즉시 반환), 클라는 폴링을 원함 → BFF가 흡수한다.

```
POST /v1/analysis/jobs
  → jobs 테이블 row 생성(status=queued), jobId 즉시 반환
  → 백그라운드:
       httpx로 추론 POST /analyze 호출(수 초)
       완료 → 결과 저장 + status=completed
       실패 → status=failed + error(code)
GET /v1/analysis/jobs/{id}          → status 폴링(queued→running→completed/failed)
GET /v1/analysis/jobs/{id}/result   → 저장된 CutResult를 PoseCandidate로 매핑해 반환
```

**jobs 테이블(최소)**

```
jobs(job_id PK, user_id, status, source, created_at, updated_at,
     result_json NULL, error_code NULL, rerun_of NULL)
```

> ⚠ 클라의 세분 status(`detecting`/`skeleton`/`pose_search`/`rendering`)는 동기 추론이라 **중간 단계를 못 준다.**
> MVP는 `queued→running→completed/failed`만 노출한다. 클라 원칙("서버가 안 주는 진행률을 임의 생성 안 함")과 일치.
> 세분 단계가 필요하면 이후 추론 서버에 SSE/단계 콜백을 추가(`API_CONTRACT.md` §8-3).

---

## 4. 인증

- **JWT access(짧게) + refresh(회전)**. 클라 `ADR-002`의 refresh single-flight와 맞물린다(회전 시 이전 refresh 무효화).
- 유저 저장은 **BFF 전용 DB**(SQLite 파일 분리 또는 Postgres) — 추론 라이브러리와 절대 섞지 않는다.
- 비밀번호는 argon2/bcrypt 해시. **평문·토큰을 로그에 남기지 않는다**(이 저장소 `QA_SECURITY_RELEASE.md` §로깅과 동일 원칙).
- 라이브러리: `fastapi-users`(빠름) 또는 `OAuth2PasswordBearer` + `python-jose` + `passlib` 수동. 과설계 금지.

---

## 5. 계약 번역 — BFF가 소유 (추론 미결 §8-5·§7 해소 지점)

- **matchLevel 매핑**: 추론 서버는 원시 `distance`/`rerank_score`/`count_confidence`만 준다.
  BFF가 이를 클라 `matchLevel`(high/medium/low)로 매핑한다 — 클라 `08_API_CONTRACT.md` §6 표 기준.
  임계값 시드는 `SEARCH_EVAL_2026-07-14.md` 관측값(좋은 매칭 ~0.15, 앉기-서기 ~0.36, 추출 실패 ~0.6+).
- **오류봉투**: 추론의 `{"detail": ...}` → 클라의 `{"error":{code,message,requestId}}`로 변환하고 `requestId` 부여·로깅.
- **경로/버전**: `/v1` 프리픽스는 BFF가 소유(추론 서버는 프리픽스 없음).

---

## 6. 제안 모듈 구조 (BFF 레포)

```
standin-bff/
├─ app/
│  ├─ main.py            FastAPI 앱·lifespan(httpx client·DB 초기화)
│  ├─ config.py          env(추론 URL·JWT 시크릿·서비스 토큰)
│  ├─ deps.py            인증 의존성(current_user)
│  ├─ auth/              /v1/auth·/v1/users/me·토큰·유저 저장소
│  ├─ jobs/              /v1/analysis/jobs·백그라운드 러너·jobs 저장소
│  ├─ inference.py       httpx 어댑터: analyze()·get_bvh() (추론 서버 호출 한곳)
│  ├─ mapping.py         CutResult→PoseCandidate·matchLevel·오류봉투
│  └─ models.py          클라 계약 Pydantic(= /v1 응답)
├─ tests/
├─ .env.example          INFERENCE_BASE_URL·JWT_SECRET·SERVICE_TOKEN·DB_PATH
└─ docker-compose.yml    bff + 추론 서버(로컬)
```

**추론 호출을 `inference.py` 한곳에 격리** → 추론 서버 계약이 바뀌어도 여기만 고친다(도원 서버 `repo.py`가 DB를 격리하는 것과 같은 원칙).

---

## 7. 배포·보안

- **별도 레포·별도 배포.** 로컬은 `docker-compose`로 BFF+추론 함께.
- ⚠ **추론 서버는 공개 노출 금지**(무인증). BFF만 공개 엣지. BFF↔추론은 서비스 토큰 또는 네트워크 격리(내부망).
- BFF가 레이트리밋·인증·요청 로깅(`requestId`)을 소유. 민감 body·토큰·원본 이미지는 로그 금지.

---

## 8. 단계별 구축 (이번 주 이후)

```
Phase 0  실연결 최초 — 인증 없이 /v1/analysis/jobs가 /analyze 래핑만.
                       클라 VITE_USE_MOCK_API=false로 후보 뷰어 검증.
Phase 1  인증 —        JWT+refresh·/v1/auth/*·유저 DB. 클라 auth.http.ts 실연결.
Phase 2  견고화 —      rerun·export 프록시·matchLevel 튜닝·오류봉투·레이트리밋.
Phase 3  스케일 —      큐를 Redis 기반으로, 필요 시 추론 단계 스트리밍.
```

> 이번 주(이 저장소 `ROADMAP.md`)는 BFF 없이 Mock 또는 추론 `/analyze` 직접 호출로 진행. BFF는 실인증이 필요해지는 시점에 Phase 0부터 착수한다.
