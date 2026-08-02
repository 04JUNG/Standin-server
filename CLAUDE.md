# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 무엇을 하는 코드인가

웹툰 러프 콘티 컷 1장(.png) → 포즈 **Top-K 후보**를 반환하는 파이프라인.
전체 제품(설계문서 v2)의 파이프라인 중 **VLM→검색 구간만** 담당한다(담당: 도원).
CSP 연동·캡처·BVH 내보내기·Three.js 뷰어 등 앞단/뒷단은 이 저장소 밖(이동원/공통).

경계 계약: **입력 = 컷 이미지, 출력 = `CutResult`(JSON 직렬화 가능)**. 이 두 끝만 맞추면 다른 팀과 병렬 개발된다.

## 실행 / 테스트

```bash
pip install numpy                 # 코어 실행에 필요한 유일한 의존성
python scripts/run_demo.py        # mock로 6개 케이스 end-to-end 데모
python tests/test_smoke.py        # 스모크 테스트(핵심 계약 검증, pytest 불필요)
python tests/test_smoke.py 2>&1 | grep FAIL   # 실패만 보기

# --- API 서버(도원 추론 서비스) ---
python scripts/build_db.py            # 합성 라이브러리 → SQLite(data/poses.db)
BVH_DIR=data/bvh python scripts/build_db.py   # 실 BVH 폴더 → SQLite
uvicorn api.app:app --reload          # http://127.0.0.1:8000/docs 에서 계약 확인
#   POST /analyze (멀티파트 PNG) → CutResult / GET /pose/{id}/bvh / GET /healthz
#   POST /refine (고른 후보 1개를 러프에 맞춰 조정) → GET /refined/{handle}/bvh
REFINE_ENABLED=0 uvicorn api.app:app   # refine 비상 스위치(시연 중 이상 동작 시)
#   DB_PATH env로 위치 지정(동기화 폴더 금지 — SQLite 락). 기본 data/poses.db
```

- **API 키·무거운 모델 없이 바로 돈다.** mock VLM 어댑터 + 합성 포즈 인덱스가 기본값.
- 테스트 러너는 `__main__`에 자체 내장(`test_*` 함수 수집). pytest도 호환되지만 필수는 아니다.
- 단일 테스트만: 파일 하단 러너 대신 `python -c "import tests.test_smoke as t; t.test_face_skips()"`.

## 실제 모델/데이터로 승격하는 스위치

전부 env 하나 또는 어댑터 한 곳만 바꾸면 된다. mock↔실제가 같은 인터페이스를 구현한다.

| 바꾸는 것 | 방법 | 코드 자리 |
|-----------|------|-----------|
| VLM provider | `VLM_PROVIDER=gemini` + `pip install google-genai pillow` + `GEMINI_API_KEY` | `src/vlm/client.py::build_vlm_client` (신 SDK) |
| 포즈 추정 | `POSE_BACKEND=rtmlib` + `pip install rtmlib onnxruntime opencv-python` | `src/pose.py::RTMPoseModel` (구현됨) |
| 실 라이브러리 | ✅ 구현됨 | `BVH_DIR=<폴더> python scripts/build_db.py` (load_bvh_pose→src/bvh) |
| 별도 검출기 | YOLO/RTMDet 연결 | `src/detect.py::RTMLibDetector` (TODO) |

`build_*()` 팩토리는 실패 시 조용히 mock로 폴백한다 → 키가 없어도 파이프라인은 항상 돈다.

## Repository Tree

```
webtoon-pose-mvp/
├─ README.md              사용법·모듈↔단계 표
├─ CLAUDE.md              (이 파일)
├─ requirements.txt       코어=numpy만. 실모델 의존성은 주석으로 분리
├─ api/                   FastAPI 레이어(앱 서버 팀과의 HTTP 경계)
│  ├─ app.py             /analyze · /export-order · /pose/{id}/bvh · /healthz · /docs
│  └─ models.py          Pydantic 응답 모델 = 문서화된 계약
├─ docs/
│  ├─ PIPELINE_OVERVIEW.md 전체 파이프라인 자립 설명(외부 LLM 인수인계용 — 규약·수식·불변식)
│  ├─ API_CONTRACT.md     전체 HTTP 계약(/analyze·/pose·/healthz) + 앱서버 경계·불일치
│  ├─ MVP_RELEASE.md      작가 실사용 30분 시연 컷라인·체크리스트(현재 목표의 단일 소스)
│  ├─ REFINE_DESIGN.md    포즈 미세조정(refine) 설계 — 축소판 파라미터·안전 게이트
│  ├─ REFINE_NEXT.md      refine 개선 방향·우선순위(원인 1개 → 입자도 3개) ★ 다음 작업 시작점
│  ├─ REFINE_HANDOFF.md   이 브랜치가 안 한 것 — export 배선·뷰어 재로드(팀원 인계)
│  ├─ DECISIONS.md        DB·라이브러리 저장·동원 핸드오프·BFF 분리 결정(읽어볼 것)
│  ├─ BFF_DESIGN.md       앱 서버(BFF) 구현 설계(Python/FastAPI)
│  ├─ EXPORT_CONTRACT.md  동원 Export 주문서 JSON 형식·예시
│  ├─ SEARCH_EVAL_2026-07-14.md  검색 정성평가 리포트(실데이터 14컷)
│  ├─ COLLABORATION.md    브랜치·커밋·PR·소유 경계·env 규칙
│  ├─ QA_SECURITY_RELEASE.md  보안(키·라이선스·무인증)·테스트·릴리스
│  └─ ROADMAP.md          우선순위와 다음 액션(SEARCH_EVAL 기반)
├─ scripts/
│  ├─ run_demo.py         mock로 6개 케이스 end-to-end 데모
│  ├─ build_db.py         합성/실 BVH → SQLite 빌드(BVH_DIR env로 실 폴더)
│  ├─ mixamo_fbx_to_bvh.py  Mixamo FBX → 포즈별 1프레임 BVH(Blender)
│  ├─ bvh_contact_sheet.py  BVH 폴더 → COCO17 스틱피겨 시트(투영 검수)
│  ├─ eval_search.py       러프→RTMPose→검색 Top-K 정성평가(4순위, --use-vlm 태그검색)
│  ├─ eval_refine.py       refine 정성평가 — 베이스·조정·러프 3열 비교(숫자만 믿지 말 것)
│  ├─ eval_refine_batch.py 폴더 단위 refine 평가 — 컷별 컨택트 시트 + summary.csv + 트리아지
│  ├─ refine_top5.py       러프 1장 → Top-5 썸네일 + Top-5 전부의 조정 BVH + manifest
│  ├─ diag_refine_3d.py    refine 3D 건전성 진단 — 4개 view 렌더 + 이동량/효율(⚠ 단일 view 검증 금지)
│  └─ vlm_tag.py           러프→VLM 태그(shot/action/view/count) 측정(5순위)
├─ tests/
│  └─ test_smoke.py       핵심 계약 검증(pytest 불필요, 자체 러너 내장)
├─ data/
│  ├─ poses.db            (자동 생성) SQLite 라이브러리 = 검색 단일 소스
│  ├─ bvh/                원본 BVH 파일(동원이 /pose/{id}/bvh로 받는 실체)
│  └─ index.pkl           (레거시) 데모용 pickle 인덱스
└─ src/
   ├─ pipeline.py         오케스트레이터: process_cut() 한 컷 흐름 전체
   ├─ schema.py           데이터 타입 + Controlled Vocabulary(열거형) = 태그 단일 소스
   ├─ config.py           env 주입 설정(provider/backend/검색 파라미터)
   ├─ routing.py          [2] 3갈래 라우팅 core/bust/skip
   ├─ detect.py           [3] 검출기 + reconcile(개수 일치=신뢰도 신호)
   ├─ pose.py             [4] RTMPose Body 래퍼(mock/rtmlib)
   ├─ bvh.py              BVH 파싱+FK+관절명→COCO17 매핑(라이브러리·검수 공용 소스)
   ├─ features.py         스켈레톤→정규화 피처(쿼리·라이브러리 공용, 반드시 동일)
   ├─ refine.py           [10] 선택 포즈 미세조정 — 팔·다리 회전만, 안전 게이트로 폐기 판정
   ├─ descriptor.py       [6] VLM 태그 + 피처 결합(JSON, LLM 불필요)
   ├─ library.py          [7] 3D→다중카메라 2D 투영 색인(합성/실BVH)
   ├─ repo.py             DB 저장소(SQLite): 스키마·feature BLOB·bvh 경로 레지스트리
   ├─ search.py           [7][8][9] 태그필터→kNN(view 우선순위)→rerank
   └─ vlm/
      ├─ client.py        [2][5][9] VLM 추상화: Mock / Gemini / OpenAI + 팩토리
      └─ prompts.py       프롬프트(개수·종류·의미만, 좌표 생성 금지)
```

대괄호 숫자는 설계문서 v2 파이프라인 단계. 새 단계를 넣을 때도
`pipeline.process_cut`의 호출 순서와 이 트리의 대응을 유지한다.

## 아키텍처 (여러 파일에 걸친 큰 그림)

한 컷의 흐름. `src/pipeline.py::Pipeline.process_cut`가 오케스트레이터다.

```
VLM.analyze  ── 1회 호출로 개수·shot·action·view·relationship·대략박스 확보
  │            (검출 보정[§7]과 의미 태그[§5]가 같은 결과를 공유 — 중복 호출 안 함)
  ├─ route      face→"skip"(조기 종료) / bust→"bust"(검색 스킵) / full_half→"core"
  ├─ detect + reconcile   검출기 개수 vs VLM 개수
  ├─ pose.estimate        각 박스에서 17kp 스켈레톤
  ├─ build_descriptors    태그 + 정규화 피처 결합
  └─ search.search        태그필터 → kNN → rerank → Top-K
```

핵심 설계 불변식(수정 시 반드시 지킬 것):

1. **VLM 태그 = shot + 사람 수(제어 신호)만.** action/view/relationship는 매칭에 안 쓴다(기하와 중복). shot→라우팅(skip/bust/core), 사람 수→분기(N명→N BVH). 관절 좌표는 VLM이 생성 안 함(검출기·포즈 모델 몫).
2. **개수 일치 = 스케일 무관 신뢰도 신호.** rtmlib score는 모델마다 스케일이 달라(Body 0.1~0.2 vs Wholebody 1.4~7.5) 신뢰도로 못 쓴다. 대신 `detect.py::reconcile`이 "검출기 개수 vs VLM 개수" 이진 일치로 `high`/`low`를 낸다. 불일치=폴백 후보. 이 신호를 다른 것으로 바꾸지 말 것.
3. **얽힘·공백 = 폴백(신뢰도 분기).** 매칭은 순수 기하라 별도 얽힘 태그가 없다. 대신 `pipeline._search_one`이 스켈레톤 score 낮음(추출 실패) 또는 Top-1 거리 > `CFG.fallback_distance`(라이브러리 공백·얽힘)면 `person_confidence='low'`로 폴백(작가). 임계값은 실데이터로 보정.
4. **피처 공간의 대칭성.** 쿼리(추출 스켈레톤)와 라이브러리(3D→2D 투영)가 **반드시 같은** `features.normalize_skeleton`을 통과해야 kNN이 성립한다. 정규화는 힙 중심 이동 + 몸통 길이 스케일 + 결측 관절 마스킹(카메라·인물 크기 불변). 한쪽만 바꾸면 검색이 조용히 망가진다. 3D 포즈→피처는 `library.pose_to_feature`가 단일 소스이고 **색인과 refine이 이 함수를 공유**한다(`test_feature_space_symmetry_shared_function`이 감시).
5. **Descriptor 결합에 LLM 불필요.** VLM 태그 + 스켈레톤 피처는 `descriptor.py`에서 JSON 구조화로만 합친다.
6. **얽힘 관계는 세트로.** `Relationship.HUGGING`/`FIGHTING`은 `is_entangled` → 2인 상호작용 포즈를 한 덩어리로 검색(개별 인물 분해가 실패하는 케이스).
7. **refine은 좋아지거나, 그대로.** `refine.py`는 검색된 베이스 포즈의 **팔 회전만** 러프에 맞춰 돌린다(기본 `REFINE_LIMBS=arms`. 루트/힙 위치·척추·손목·**다리** 고정). 다리는 투영 관측 감도가 팔의 1/3.4라 손실이 못 보는 방향으로 크게 움직인다 — 3D 정규화·이동량 게이트 없이 켜지 말 것(`REFINE_DESIGN.md` §6-4). 안전 게이트 중 하나라도 걸리면 조정을 버리고 베이스를 그대로 반환한다 — refine이 결과를 나쁘게 만드는 경로는 존재하면 안 된다. 특히 **검색이 실패한 컷에는 refine을 돌리지 않는다**(틀린 베이스를 러프에 끼워맞추면 더 이상해진다). 게이트를 약화시키는 방향으로 바꾸지 말 것. 상세: `docs/REFINE_DESIGN.md`.

## 검색이 어떻게 매칭되나 (개념)

그림에서 3D를 복원(리프팅)하지 **않는다**. 이미 3D인 라이브러리 포즈를 여러 가상 카메라로 2D 투영해 색인하고(`library.py::VIRTUAL_CAMERAS`, view별 1개 엔트리), 추출한 2D 스켈레톤과 같은 피처 공간에서 **순수 기하 kNN**(`search.knn_geometric`, 거리=몸통 12관절 L2 `features.pose_distance`)을 돈다. **태그(action/view)로 필터링하지 않는다** — 기하와 중복이라 매칭에서 제외(설계 결정). cosine은 국소 차이를 못 잡아 폐기. `knn`은 같은 `pose_id`의 여러 view 중 최선 1개만 남겨 후보 다양성을 확보한다.

## Controlled Vocabulary 주의

`schema.py`의 `Shot`/`Action`/`View`/`Relationship` 열거형이 태그의 **단일 소스**다. VLM 프롬프트(`vlm/prompts.py`)의 허용값과 반드시 일치해야 한다. 라이브러리 태깅 후 이 어휘를 바꾸면 전량 재태깅이 필요하므로, 값 추가/변경은 프롬프트·인덱스 재빌드와 함께 처리한다.

## 컷 3갈래 안전마진

`routing.py`가 3갈래로 나누지만, 채택 모델 RTMPose **Body**는 흉상이 `core`로 오분류돼도 폭발하지 않는다(안 보이는 관절은 안 그림). 즉 라우팅 정확도가 완벽하지 않아도 치명적이지 않다 — 이 여유를 없애는 방향(Wholebody 도입 등)으로 바꾸지 말 것.
