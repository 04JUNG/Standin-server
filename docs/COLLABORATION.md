# 협업 규칙 — 브랜치 · 커밋 · PR · 소유 경계

> 클라이언트 `05_REPOSITORY_AND_COLLABORATION.md`의 서버판. 이미 있는 `.github/` 템플릿(PR·이슈)과
> `ci.yml`을 전제로, 이 저장소에서 실제로 지킬 규칙만 정리한다.

---

## 1. 브랜치 전략

`ci.yml`이 `main`·`develop`을 보호 대상으로 본다 → 2단 브랜치 + 기능 브랜치.

```text
main            항상 실행 가능·시연 가능(태그·릴리스 기준)
└─ develop      통합 브랜치(기능이 모이는 곳, CI push 트리거)
   ├─ feat/vlm-pose-search
   ├─ feat/rtmpose-real
   ├─ feat/bvh-loader
   └─ fix/...
```

규칙:

- **`main` 직접 push 금지.** 기능은 `feat/*` → `develop` → `main` 순으로 올린다.
- 기능별 짧은 브랜치. 큰 기능(예: 실 BVH 파이프라인)은 여러 PR로 분리.
- PR merge 전 최소 1명 확인. rebase 또는 squash 중 하나로 합의.
- `main`은 언제든 `run_demo.py`·`test_smoke.py`가 통과해야 한다.

---

## 2. 커밋 메시지

Conventional Commits 간단 적용(현 히스토리 `[Feat]:`/`feat(server):` 혼용 → 아래로 통일 권장):

```text
feat(search): view 우선순위 rerank 추가
feat(vlm): Gemini 어댑터 연결
fix(pose): cv2 한글 경로 imread 버그 수정
docs(api): /analyze 계약 문서화
refactor(repo): SQLite 스키마 분리
test(smoke): 폴백 신뢰도 케이스 추가
chore: requirements 정리
```

한 커밋에 파이프라인 로직·API·스크립트 대규모 변경을 섞지 않는다.

---

## 3. PR 규칙

`.github/PULL_REQUEST_TEMPLATE.md`가 자동 적용된다. 핵심 체크리스트:

- [ ] 로컬에서 `python tests/test_smoke.py` 통과
- [ ] `python -m compileall src api scripts` 통과(문법)
- [ ] **`.env` / `data/` / 모델 가중치(`*.onnx`) 안 들어감**
- [ ] 관련 이슈 링크

권장 크기:

- 리뷰 가능한 변경: 300~500줄 안팎(생성물·lockfile 제외).
- 한 PR은 하나의 파이프라인 단계 또는 하나의 계약 변경.
- **계약(`/analyze`·`/export-order`) 변경 시 반드시 `docs/API_CONTRACT.md`·`EXPORT_CONTRACT.md` 동시 수정**(§6).
- 검색 품질에 영향 주는 변경은 `scripts/eval_search.py` 결과(전/후)를 첨부.

---

## 4. 코드 소유 경계

### 충돌·파급이 큰 파일 (변경 전 공유)

- `src/schema.py` — Controlled Vocabulary. 값 변경 시 **라이브러리 전량 재태깅** 필요.
- `src/features.py` — 정규화 규격. 바꾸면 `feature_version` +1 하고 DB 재빌드(쿼리↔라이브러리 대칭 불변식).
- `src/config.py` — 검색 파라미터·임계값.
- `api/models.py` — 앱 서버와의 계약.
- `src/repo.py` — DB 스키마.

### 단계별 소유 (병렬 개발 지점)

각 모듈이 파이프라인 한 단계를 소유한다(`CLAUDE.md` 모듈 트리). 인터페이스만 지키면 내부는 자유:

```text
src/vlm/       [2][5] VLM 어댑터(Mock/Gemini/OpenAI)
src/detect.py  [3]    검출 + 개수 보정
src/pose.py    [4]    RTMPose 래퍼
src/library.py [7]    3D→2D 투영 색인
src/search.py  [7][8][9] kNN + rerank
```

불변식(수정 시 반드시 지킬 것)은 `CLAUDE.md` "핵심 설계 불변식" 6개 참조.

---

## 5. 환경변수

`.env.example`을 복사해 `.env`로 쓴다. **`.env`는 커밋 금지**(`.gitignore` 등록됨, `!.env.example`만 예외).

```text
VLM_PROVIDER=mock          # mock | gemini | openai
GEMINI_REQUEST_TIMEOUT_MS=45000  # Gemini HTTP 1회 상한
GEMINI_MAX_ATTEMPTS=3      # 최초 호출 포함, 429/503만 재시도(모델 1개당)
GEMINI_RETRY_BASE_SECONDS=0.5
GEMINI_RETRY_MAX_SECONDS=2.0
GEMINI_TOTAL_BUDGET_SECONDS=75   # 폴백 모델까지 합친 VLM 단계 전체 예산
GEMINI_FALLBACK_MODELS=gemini-2.5-flash-lite  # 1차가 503으로 소진되면 태울 다른 용량 풀
GEMINI_THINKING_BUDGET=0   # 0=끔 / -1=동적 / none=필드 미전송
POSE_BACKEND=mock          # mock | rtmlib
DB_PATH=data/poses.db      # ⚠ 동기화 폴더(드롭박스/OneDrive) 금지 — SQLite 락
BVH_DIR=                   # 실 BVH 폴더(build_db 시)
FALLBACK_DISTANCE=0.45     # 실데이터로 보정
```

- **API 키(`GEMINI_API_KEY` 등)는 절대 코드·로그·PR에 넣지 않는다.**
- 팀원별 로컬 설정(파이썬 런처 `py -3.12` 등 OS 편차)은 `docs/SEARCH_EVAL_*.md` §환경 주의처럼 문서화.

---

## 6. 문서 갱신 규칙

아래 변경은 관련 문서를 **같은 PR에서** 수정한다.

| 변경 | 함께 수정할 문서 |
|---|---|
| `/analyze`·`/pose`·`/healthz` request/response | `docs/API_CONTRACT.md` |
| `/export-order` 형식 | `docs/EXPORT_CONTRACT.md` |
| DB 스키마·저장 방식 | `docs/DECISIONS.md` |
| Controlled Vocabulary(태그 어휘) | `docs/API_CONTRACT.md` §4 + `src/vlm/prompts.py` |
| 검색 파라미터·임계값 기본값 | `docs/API_CONTRACT.md` + `CLAUDE.md` |

구현과 문서가 충돌하면 **코드가 정본**(`/docs` OpenAPI). 이 경우 문서를 코드에 맞춘다.

---

## 7. 이슈 템플릿

`.github/ISSUE_TEMPLATE/`에 두 종류가 있다:

- **버그 리포트** — 발생 상황·기대·실제·환경(OS/Python/provider)·시도.
- **기능/작업** — 목표·범위/하지 않을 것·완료 기준(DoD).

환경 칸에 **provider(mock/gemini)와 `py -3.12` 여부**를 반드시 적는다(재현성).

---

## 8. Definition of Done

기능 완료는 코드가 작성된 상태가 아니다.

- 요구 동작이 mock으로 재현됨(`run_demo.py` 또는 스모크 케이스 추가)
- `test_smoke.py`·`compileall` 통과
- mock↔실모델 인터페이스 대칭 유지(실패 시 mock 폴백)
- 계약·설계 변경이면 문서 반영(§6)
- 시크릿·데이터(`data/`, `.env`, `*.onnx`) 미포함
