# webtoon-pose-mvp — VLM→검색 스캐폴드 (도원 담당 범위)

러프 콘티 컷 1장 → 포즈 Top-K 후보. 설계문서 v2의 파이프라인 중
**[2] 3갈래 라우팅 · [3] 검출+VLM 보정 · [4] 스켈레톤 · [5] 의미 태그 ·
[6] Descriptor · [7] 검색(kNN) · [9] Rerank** 구간을 구현한다.

## 지금 상태
- **API 키·모델 없이 바로 실행**된다(mock 어댑터 + 합성 포즈 인덱스).
- 실제 모델은 어댑터만 갈아끼우면 됨: `VLM_PROVIDER=gemini`, `POSE_BACKEND=rtmlib`.

## 실행
```bash
pip install numpy
python scripts/run_demo.py     # 6개 케이스 데모
python tests/test_smoke.py     # 스모크 테스트

python scripts/build_db.py             # 라이브러리 → SQLite
uvicorn api.app:app --reload           # /docs 에서 API 계약 확인
```

## 서비스 경계 (도원 추론 서버 ↔ 앱 서버)
- `POST /analyze` (PNG) → `CutResult` JSON, `GET /pose/{id}/bvh` (동원 내보내기가 소비).
- DB=SQLite(`data/poses.db`), 라이브러리 단일 소스. 결정 근거는 `docs/DECISIONS.md`.

## 구조 (모듈 ↔ 파이프라인 단계)
| 파일 | 단계 | 역할 |
|------|------|------|
| `src/schema.py` | 전역 | 데이터 타입 + Controlled Vocabulary(열거형) |
| `src/config.py` | 전역 | provider/백엔드/검색 파라미터(env 주입) |
| `src/vlm/prompts.py` | [2][5] | VLM 프롬프트(개수·종류·의미만, 좌표 금지) |
| `src/vlm/client.py` | [2][5][9] | VLM 추상화: Mock / Gemini / OpenAI |
| `src/routing.py` | [2] | 3갈래 라우팅(core/bust/skip) |
| `src/detect.py` | [3] | 검출 + VLM 개수 보정(개수 일치=신뢰도 신호) |
| `src/pose.py` | [4] | RTMPose Body 래퍼(mock/실제) |
| `src/features.py` | [6] | 스켈레톤→정규화 피처(카메라·크기 불변) |
| `src/descriptor.py` | [6] | VLM 태그+피처 결합(JSON, LLM 불필요) |
| `src/library.py` | [7] | 3D→다중카메라 2D 투영 색인(합성/실BVH) |
| `src/search.py` | [7][8][9] | 태그필터→kNN(view 우선순위)→rerank |
| `src/pipeline.py` | 전체 | 오케스트레이터 |

## 실데이터/실모델로 승격하는 자리(TODO)
- `src/library.py::load_bvh_pose()` — BVH 파서 + 관절명→COCO17 매핑
- `src/pose.py::RTMPoseModel` — rtmlib Body 연결
- `src/vlm/client.py::GeminiVLMClient` — GEMINI_API_KEY만 있으면 동작
- `src/detect.py::RTMLibDetector` — 별도 검출기(YOLO 등) 붙일 때
