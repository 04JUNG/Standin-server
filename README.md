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
                                     GET  /pose/{id}/bvh
                                     POST /export-order
                                     GET  /healthz
```

이 서버는 **동기·무인증·무상태 추론 API**다. 인증·Job 비동기·버전 프리픽스는 앱 서버가 감싼다.
자세한 계약과 경계는 [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md).

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

## 문서 맵

읽는 순서:

1. [`CLAUDE.md`](CLAUDE.md) — 파이프라인 단계·설계 불변식·모듈 트리 (개발 전 필독)
2. [`README-vlm-search.md`](README-vlm-search.md) — 스캐폴드 실행법·모듈↔단계 표
3. [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md) — 전체 HTTP 계약 + 앱 서버 경계·불일치
4. [`docs/EXPORT_CONTRACT.md`](docs/EXPORT_CONTRACT.md) — 동원 Export 주문서(`/export-order`) 형식
5. [`docs/DECISIONS.md`](docs/DECISIONS.md) — DB·라이브러리 저장·핸드오프 결정(ADR)
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
