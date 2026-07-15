# QA · 보안 · 릴리스 기준

> 클라이언트 `11_QA_SECURITY_RELEASE.md`의 서버판. 추론 서버 특유의 리스크(**API 키·BVH 라이선스·무인증
> 경계**)에 집중한다. 이 서버는 동기·무인증·무상태다(`docs/API_CONTRACT.md` §0).

---

## 1. 실행 환경

| 항목 | 값 |
|---|---|
| Python | **3.12** (CI 기준). Windows는 `py -3.12` — `python`이 pip 없는 msys2를 가리키는 경우 주의 |
| 코어 의존성 | `numpy`만(스캐폴드 실행) |
| API 의존성 | `fastapi`·`uvicorn`·`python-multipart`·`Pillow` |
| 실모델(선택) | `rtmlib`·`onnxruntime`·`opencv-python`·`google-genai` 등(주석 처리, 각자 설치) |

**OS 종속 이슈(문서화된 것):**

- ✅ **cv2 한글 경로** — `cv2.imread(유니코드 경로)`가 Windows에서 `None` 반환(OpenCV 알려진 한계).
  `src/pose.py::_load_bgr`가 `cv2.imdecode(np.fromfile(...))`로 우회해 해소됨(`SEARCH_EVAL_2026-07-14.md` P1은 머지 이전 기록).
- Blender 배치 스크립트는 패키지 앱 가상화 경로에서 외부 프로세스가 접근 불가 → `C:\tmp`에서 실행.

---

## 2. 기능 테스트

### 스모크 (필수, mock)

```bash
python tests/test_smoke.py          # 핵심 계약 검증(자체 러너, pytest 불필요)
python -m compileall src api scripts # 문법/빌드
```

CI(`.github/workflows/ci.yml`)가 PR마다 위 두 개를 mock으로 실행한다(모델·키 불필요).

### 검색 품질 (정성, 실데이터)

- `scripts/eval_search.py --image <cut.png>` — 러프→RTMPose→Top-K를 사람 눈으로 판정.
- **자기 쿼리 sanity check**: 라이브러리 BVH를 쿼리로 넣으면 `dist=0.000`·#1 자기 자신(피처 공간 대칭 정상).
- 검색에 영향 주는 변경은 **전/후 eval 결과를 PR에 첨부**(`COLLABORATION.md` §3).

### 확인할 케이스

- `route` 3갈래: `core`(검색) / `bust`(스킵) / `skip`(빈 후보).
- `count_confidence`: 검출기≠VLM 개수 → `low` 폴백.
- 추출 실패 컷: `distance` 큼(~1.2) → "매칭 없음"을 정직하게 냄(폴백 신호).
- `/pose/{id}/bvh`: 미등록(404) / 등록됐으나 파일 없음(409, 합성 단계 정상).

---

## 3. 보안

### API 키 (최우선)

- **키는 `.env`에만.** `.gitignore`가 `.env`·`.env.*`·`*.key`·`*.pem`을 막고 `!.env.example`만 예외.
- 코드·로그·PR·이슈에 키를 절대 노출하지 않는다.
- `build_vlm_client`는 키가 없으면 조용히 mock 폴백 → 키 누락이 크래시가 아니라 mock 동작으로 나타남(정상).

### 라이브러리 데이터 라이선스 ⚠

- **Mixamo/CMU BVH는 재배포 금지 조항**이 있어 **저장소에 커밋 금지**(`.gitignore`: `data/bvh/`·`data/*.db`).
- DB에는 **BVH 경로만** 저장하고 바이트는 파일시스템에 둔다(`DECISIONS.md` 결정 2). 파일이 곧 동원 산출물.
- `poses.license` 필드로 출처·조항을 추적한다("우리는 추천 서비스라 사용은 OK지만 필드로 관리").

### 무인증 경계 ⚠

- 이 서버는 **인증이 없다.** 공개 인터넷에 그대로 노출하지 않는다 — 내부망 또는 앱 서버 뒤(로컬 전제).
- 앱 서버가 이 층을 감쌀지, 서비스 토큰/네트워크 격리를 둘지는 팀 확인 항목(`API_CONTRACT.md` §8-1, §8-7).

### 입력·파일

- 업로드 PNG는 PIL로 로드(실패 시 더미 폴백). **크기·MIME 강검증은 앞단**이지만 서버도 방어적으로 재검증 권장(`API_CONTRACT.md` §8-4).
- `/pose/{pose_id}/bvh`는 DB에 등록된 경로만 반환(임의 경로 traversal 불가) — `pose_id`로 조회 후 파일 존재 확인.
- `DB_PATH`는 로컬 디스크. ⚠ 동기화 폴더(드롭박스/OneDrive)는 SQLite 락 오류.

### CORS

- 현재 CORS 미들웨어 미설정. 브라우저 직접 호출이 필요하면 **허용 오리진을 명시적으로 좁혀** 추가(와일드카드 금지). Tauri·앱 서버 경유면 불필요.

---

## 4. 로깅

포함 가능: provider/backend, `route`·`count_confidence`, `pose_id`, 처리 단계, 오류 stack.

포함 금지:

- **API 키·토큰**
- **원본 입력 이미지**(작가 미공개 창작물일 수 있음)
- BVH 내용
- VLM 원문 프롬프트에 실린 이미지

⚠ 입력 콘티는 미공개 저작물 가정. "자동으로 학습에 사용"한다고 가정하지 않는다 — 데이터 보관·학습 정책이 확정되기 전엔 그렇게 동작하지 않게 둔다.

---

## 5. 성능

- **모델·인덱스는 lifespan에서 1회 로드**(요청마다 X) — `api/app.py` `STATE`.
- `/analyze`는 동기(`def`)라 FastAPI가 threadpool에서 실행 → 블로킹 포즈 추론이 이벤트 루프를 막지 않는다.
- kNN은 MVP 규모(포즈 200~500 × 뷰 4)에서 **메모리 브루트포스로 즉시**. 벡터 DB는 스케일 시 `repo.py`만 교체(`DECISIONS.md` 결정 1).
- rtmpose-x 가중치 176MB → CPU onnxruntime. 첫 로드 시간 감안(healthz `ok`로 준비 확인).

---

## 6. 릴리스 / 실모델 승격 체크

`main` 태깅 또는 실모델 전환 전:

- [ ] `test_smoke.py`·`compileall` 통과
- [ ] `api/app.py` `version` 갱신(현 `0.1.0`)
- [ ] `requirements.txt`의 실모델 의존성 주석 해제·버전 고정
- [ ] `.env`로 `VLM_PROVIDER`/`POSE_BACKEND` 승격, 키 주입 확인(`GET /healthz`로 provider 확인)
- [ ] `BVH_DIR`로 실 라이브러리 빌드 후 자기 쿼리 sanity check
- [ ] `data/`·`.env`·`*.onnx` 커밋되지 않았는지 확인
- [ ] `FALLBACK_DISTANCE`·`MIN_SKELETON_SCORE`를 실데이터로 보정(`SEARCH_EVAL` §4)
- [ ] 무인증 서버가 공개 노출되지 않는 배포 형상인지 확인
