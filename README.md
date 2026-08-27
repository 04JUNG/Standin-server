# Standin-server

웹툰 러프 콘티 컷 1장(.png) → 가까운 **포즈 Top-K 후보**를 반환하는 Python 추론 서버.
전체 제품(Standin) 파이프라인 중 **VLM 분석 → 포즈 검색** 구간을 담당한다(담당: 도원).

> 캡처·업로드·3D 뷰어·BVH 내보내기 등 앱 쪽은 별도 저장소([Standin-client](https://github.com/04JUNG/Standin-client)).
> 이 저장소는 **입력 = 컷 이미지, 출력 = `CutResult`(JSON)** 두 끝만 맞추면 다른 팀과 병렬 개발된다.

---

## 빠른 시작

```bash
pip install numpy                 # 코어 실행에 필요한 유일한 의존성
python scripts/run_demo.py        # mock으로 6개 케이스 end-to-end 데모
python tests/test_smoke.py        # 스모크 테스트(핵심 계약 검증, pytest 불필요)
python tests/test_refine_v2.py    # 승인된 refine v2 feature-flag·안전 계약
```

**API 서버**(앱 서버 팀과의 HTTP 경계):

```bash
pip install -r requirements.txt
python scripts/build_db.py            # 합성 라이브러리 → SQLite(data/poses.db)
uvicorn api.app:app --reload          # http://127.0.0.1:8000/docs 에서 계약 확인
```

> **API 키·무거운 모델 없이 바로 돈다.** mock VLM 어댑터 + 합성 포즈 인덱스가 기본값이다.
> 실제 모델은 env 하나로 승격한다(아래 표).

---

## 서비스 경계 (한눈에)

```
[Tauri 앱] ──> [앱 서버(친구들)] ──> [도원 추론 서버 = 이 저장소]
              인증·Job·기록          POST /analyze (동기, 무인증)
                                     POST /refine   (고른 후보 → 러프에 맞춰 조정)
                                     GET  /pose/{id}/bvh
                                     POST /export-order
                                     GET  /healthz
```

이 서버는 **동기·무인증·무상태 추론 API**다. 인증·Job 비동기·버전 프리픽스는 앱 서버가 감싼다.
자세한 계약과 경계는 [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md).

Refine는 v2.5 safe aggressive가 제품 기본이다(`REFINE_V2_ENABLED=1`,
`REFINE_DEFAULT_MODE=aggressive`). aggressive 후보는 원본 대비 구조 안전과 conservative 대비
공통 metric non-regression을 모두 통과한 경우에만 반환되고, 실패하면 conservative 또는 베이스로
정확히 복구된다. 몸통은 승인 전까지 `REFINE_V2_TORSO=0`을 유지한다. 비상 복구는
`REFINE_DEFAULT_MODE=conservative` → `REFINE_V2_ENABLED=0` 순이다.

조정본은 `POST /refine` 응답의 `bvh` 본문으로만 나간다. 로컬 디스크에 남기지 않으므로 조정본
다운로드 URL은 없다(`docs/REFINE_HANDOFF.md` §3 4단계).

선택된 최종 BVH를 캐릭터 FBX로 바꾸는 출력단은 별도 내부 서비스다. 추론 프로세스는 Blender를
import하지 않으며, 변환 1건마다 Blender 5.2 child process를 새로 실행한다.

```bash
export STANDIN_MASTER_V2_URI=/absolute/read-only/standin-master-v2.fbx
export BLENDER_BINARY=/absolute/path/to/blender
uvicorn converter_api.app:app --port 8001
# GET /healthz · GET /characters · POST /convert
```

`POST /convert`는 BFF 내부망 전용 multipart API이며 `bvh` 업로드와 registry의
`character_id`만 받는다. 사용자 URL이나 서버 파일 경로는 받지 않는다. 동결 solver와 통합 정본은
[`docs/FBX_CONVERTER_V3_2_INTEGRATION_HANDOFF.md`](docs/FBX_CONVERTER_V3_2_INTEGRATION_HANDOFF.md)를
따른다.

BFF는 `refined=true`면 `/refine` 응답의 inline `bvh`, 아니면 base `bvh_url` 응답 바이트를
최종 입력으로 선택한다. Converter 성공 응답의 `X-Standin-Source-BVH-SHA256`을 BFF가 계산한
최종 BVH SHA와 대조한다. mirror는 `/convert`에서 한 번만 적용한다. Phase 3 상세는
[`docs/FBX_CONVERTER_V3_2_PHASE3_BFF_HANDOFF.md`](docs/FBX_CONVERTER_V3_2_PHASE3_BFF_HANDOFF.md)를
따른다.

Blender converter 회귀는 repo root에서 아래처럼 실행한다. `--python-exit-code 1`은 Python
traceback이 발생했는데 Blender가 종료코드 0을 반환하는 false pass를 막는다. 생성물 위치는
`CONVERTER_TEST_ARTIFACT_ROOT`로 repo 밖 임시 디렉터리를 지정한다.

```bash
export CONVERTER_TEST_ARTIFACT_ROOT=/absolute/path/to/temporary-directory
/path/to/blender --background --python-use-system-env --python-exit-code 1 \
  --python tests/converter/make_fixtures.py
/path/to/blender --background --python-use-system-env --python-exit-code 1 \
  --python tests/converter/test_convert.py
```

합성 1차 스크리닝은 다음처럼 실행한다.

```bash
python scripts/eval_refine_v2_synthetic.py --bvh-dir data/bvh --out out/refine-v2-synthetic
```

---

## 실제 모델/데이터로 승격

전부 env 하나 또는 어댑터 한 곳만 바꾸면 된다. mock↔실제가 같은 인터페이스를 구현한다.

| 바꾸는 것 | 방법 | 코드 자리 |
|---|---|---|
| VLM provider | `VLM_PROVIDER=gemini` + `pip install google-genai pillow` + `GEMINI_API_KEY` | `src/vlm/client.py::build_vlm_client` |
| 포즈 추정 | `POSE_BACKEND=rtmlib` + `pip install rtmlib onnxruntime opencv-python` | `src/pose.py::RTMPoseModel` |
| 실 라이브러리 | `BVH_DIR=<폴더> python scripts/build_db.py` | `src/library.py::load_bvh_pose` |

`build_*()` 팩토리는 실패 시 조용히 mock으로 폴백한다 → 키가 없어도 파이프라인은 항상 돈다.
설정값은 `.env`(예시: `.env.example`)로 주입. **`.env`·API 키·`data/`는 커밋 금지**(`.gitignore` 등록됨).

---

## 배포 (컨테이너)

### `APP_ENV=production`이 하는 일

개발 편의 기능이 프로덕션에서 **가짜 결과를 정상처럼 서빙하지 않도록** 막는다.

| 상황 | development | production |
|---|---|---|
| 포즈 라이브러리 없음 | 합성 라이브러리 생성 후 기동 | **기동 실패** |
| `VLM_PROVIDER=mock` 또는 `POSE_BACKEND=mock` | 그대로 기동 | **기동 실패** |
| 라이브러리가 비어 있음(`pose_count=0`) | `/healthz` 503 | `/healthz` 503 |

`build_*()`의 조용한 mock 폴백은 개발 편의를 위해 남아 있다. 프로덕션에서는
`Pipeline` 초기화 후 실제 VLM·포즈 인스턴스를 검사하므로, 실백엔드 설정에서 키나
런타임 의존성이 빠져 mock으로 폴백해도 기동에 실패한다.

### 포즈 라이브러리 공급

라이브러리(`poses.db` · `index.pkl` · `bvh/` · `thumbs/`)는 **이미지에 넣지 않는다** — Mixamo/CMU 재배포 금지 조항 때문에 `.gitignore`·`.dockerignore`로 제외돼 있다. 배포 환경에서는 기동 시 번들을 받아 푼다.

배포 중인 추론 서버에 올릴 때는 아래를 직접 치지 말고 **배포 스크립트를 쓴다.** 검증 → 압축 →
업로드 → 재기동 → 확인을 한 번에 하고, 번들이 잘못됐으면 업로드 자체를 막는다.

```bash
python scripts/deploy_pose_library.py data/
```

수동으로 번들만 만들 때는 다음과 같다. **`thumbs`를 빠뜨리지 않는다** — 빠져도 에러가 나지
않고 썸네일만 조용히 사라진다(`src/thumbnails.py`가 파일이 없으면 `None`을 돌려준다).

```bash
# 1) 번들 만들기 (루트에 poses.db · index.pkl · bvh/ · thumbs/)
tar -czf pose-library-v1.tar.gz -C data poses.db index.pkl bvh thumbs

# 2) 비공개 버킷에 올리기
aws s3 cp pose-library-v1.tar.gz s3://<bucket>/pose-library/v1.tar.gz

# 3) 컨테이너에 위치만 알려주기
POSE_LIBRARY_URI=s3://<bucket>/pose-library/v1.tar.gz
```

- `s3://`는 **boto3 + ECS 태스크 역할**로 인증한다(키를 환경변수에 두지 않는다). `requirements.txt`에서 `boto3` 주석을 해제해야 한다.
- `https://`도 지원한다(표준 라이브러리로 받으므로 추가 의존성 없음).
- 이미 `DB_PATH`에 파일이 있으면 받지 않는다 → 로컬은 `data/`에 직접 두면 그대로 쓴다.
- 번들 압축 해제 시 경로 탈출(`../`)은 거부한다.

라이브러리를 바꾸려면 새 번들을 올리고 `POSE_LIBRARY_URI`만 갱신한다. 이미지는 다시 굽지 않아도 된다.

### 이미지

- 비루트 유저(`uid=10001`)로 실행한다.
- `HEALTHCHECK`가 `/healthz`를 본다 — 라이브러리가 비면 503이라 오케스트레이터가 태스크를 교체한다.

---

## 문서 맵

읽는 순서:

1. [`CLAUDE.md`](CLAUDE.md) — 파이프라인 단계·설계 불변식·모듈 트리 (개발 전 필독)
2. [`README-vlm-search.md`](README-vlm-search.md) — 스캐폴드 실행법·모듈↔단계 표
3. [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md) — 전체 HTTP 계약 + 앱 서버 경계·불일치
4. [`docs/EXPORT_CONTRACT.md`](docs/EXPORT_CONTRACT.md) — 동원 Export 주문서(`/export-order`) 형식
5. [`docs/DECISIONS.md`](docs/DECISIONS.md) — DB·라이브러리 저장·핸드오프·앱서버(BFF) 분리 결정(ADR)
   - [`docs/BFF_DESIGN.md`](docs/BFF_DESIGN.md) — 앱 서버(BFF) 구현 설계(Python/FastAPI)
6. [`docs/SEARCH_EVAL_2026-07-14.md`](docs/SEARCH_EVAL_2026-07-14.md) — 실데이터 검색 정성평가
7. [`docs/COLLABORATION.md`](docs/COLLABORATION.md) — 브랜치·커밋·PR·소유 경계·env 규칙
8. [`docs/QA_SECURITY_RELEASE.md`](docs/QA_SECURITY_RELEASE.md) — 보안(키·라이선스·무인증 경계)·테스트·릴리스
9. [`docs/ROADMAP.md`](docs/ROADMAP.md) — 우선순위와 다음 액션

문서가 충돌하면 우선순위: `CLAUDE.md` → 기능별 계약(API/EXPORT) → `DECISIONS.md`.

---

## 테스트

```bash
python tests/test_smoke.py                    # 전체 스모크(자체 러너 내장, pytest 불필요)
python tests/test_smoke.py 2>&1 | grep FAIL   # 실패만 보기
python -m compileall src api scripts          # 문법/빌드 체크(CI와 동일)
```

CI(`.github/workflows/ci.yml`)는 PR마다 위 두 가지를 mock으로 돌린다(모델·키 불필요).

## 라이선스

`LICENSE` 참조. ⚠ **라이브러리 BVH 데이터(Mixamo/CMU)는 재배포 금지 조항**이 있어 저장소에 커밋하지 않는다(경로만 DB에 저장). 자세한 내용은 [`docs/QA_SECURITY_RELEASE.md`](docs/QA_SECURITY_RELEASE.md) §라이선스.
