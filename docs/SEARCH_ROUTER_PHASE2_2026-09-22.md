# 라우터 다음 단계 — 카메라 규약·의미 자산 검증과 VLM 검수 준비

> 2026-09-22 · stage 2 개발 기록. 그룹 관계/하체 단독 입력 제외 유지.
> 운영 검색·refine·VLM 기본값은 변경하지 않았다. 생성물은 별도 실험 경로다.

## 1. 결과 요약

| 작업 | 상태 | 결과 |
|---|---|---|
| 실제 러프 VLM 슬롯 검수 | 부분 검수 완료 (후속 갱신) | 사용자 승인 후 Gemini 실제 호출: 17컷 중 6컷·11명 검수, 11컷은 503/429 미검수. 관계 과잉 제외·반신 지지 과신 발견. [후속 보고서](ROUGH_VLM_REVIEW_2026-09-22.md) |
| 물리 카메라 v2 | 독립 구현·로컬 검증 완료 | 좌표 반사 제거, d-z 깊이, 양의 고도=높은 눈 위치, 버전 분리. 운영 통합 전 |
| Blender 검증 | 완료 | 기준 카메라 4종 + 실제 모델 2사례. 최대 정규화 화면 오차 5.04e-7. 앞 표식 가림 확인 2/2 |
| 기존 의미 자산 감사 | 완료 | 현재 DB와 기존 문서가 불일치. 기존 빌드 그대로 사용 불가 |
| 현재 BVH 측정 갱신 | 완료 | 1,215개 측정, 기존 격리 33개, 측정 실패 0 |
| 신규 자동 테스트 | 완료 | 11/11 통과 |
| 실제 시맨틱 검색·조합·정확도 평가 | 미실행 | 개선율을 산출하지 않음 |

## 2. 실제 VLM 슬롯 검수

`scripts/eval_rough_slots.py`는 고정 프로토콜의 이미지와 응답 해시를 확인한 뒤 같은 이미지에 현재 설정된 Gemini 모델로 확장 슬롯을 요청한다. 실제 provider를 직접 초기화하며 mock 폴백을 허용하지 않는다. 요청당 60초 timeout, SDK 자동 retry는 끈다. 기존 1회 분석 요청에 슬롯을 포함하고 모델 버전·사용량·지연·원문 응답·파싱 상태를 저장한다.

새 VLM 인물 인덱스를 예전 pose 인덱스와 그대로 결합하지 않는다. coarse box와 기존 person box 사이의 양방향 최대 IoU 및 차이 조건을 확인하고, 기존 owner_verified=true인 경우에만 관절을 재사용한다. 실패하면 의미-only 관측 상태로 유지한다. 이 매칭도 사람 판정의 대체가 아니며 검토 화면에 사유를 표시한다.

### 최초 차단 이력 (이후 사용자 승인으로 해소)

기본 sandbox 실행의 첫 2건은 ConnectError로 끝났다. 네트워크 실행 요청에 대해 자동 승인 검토가 “실제 러프 이미지의 Gemini 외부 전송 목적지까지 명시적으로 승인되지 않았다”는 이유로 거부했다. 우회 경로·다른 provider로 보내지 않았으며 사용자에게 17컷 Gemini 전송(먼저 2컷 파일럿) 승인을 요청했다.

최초 단계에서는 성공한 새 VLM 결과가 0건이었다. 오류 기록은 `vlm/responses/*.attempt-01.json`에 보존했다. 이후 사용자가 실제 VLM 검수를 승인하여 실행했고, 현재 결과는 **6컷·11명 성공 / 11컷 미검수**다. 새 응답과 세부 검수는 [후속 보고서](ROUGH_VLM_REVIEW_2026-09-22.md) 및 `artifacts/rough_vlm_review_20260922/`에 기록했다.

### 최초 재개 명령 기록

```bash
.venv/bin/python scripts/eval_rough_slots.py \
  --protocol artifacts/camera_real_rough_20260921_v2/protocol.json \
  --out artifacts/rough_router_phase2_20260922/vlm \
  --limit 2 --resume --retry-errors
```

파일럿 형식·인물 귀속을 확인한 뒤 `--limit 17 --resume`으로 남은 컷을 처리한다. 새 attempt 파일을 추가하고 이전 실패 기록은 덮어쓰지 않는다. 프롬프트/프로토콜/모델 요청 식별자가 달라지면 같은 run 재개를 거부한다. 프롬프트를 변경하면 새 run에서 평가해야 한다.

## 3. 카메라 v2

새 파일: `src/experimental/camera_v2.py`.

- yaw/elevation=0에서 카메라는 원점의 +z 쪽에서 바라본다.
- 원근 깊이는 `distance - camera_z`, 화면 좌표는 x 오른쪽/y 아래쪽이다.
- Blender로의 축 변환은 `(x, -z, y)`이며 determinant가 양수다. 메시 반사나 그에 따른 normal 반전을 사용하지 않는다.
- elevation>0이면 카메라 눈이 위로 올라간다. near plane, roll, world_to_camera 직렬화를 같은 규약으로 검증한다.
- 스키마는 `pose-camera-v2-experimental`이며 기존 v1과 구별한다.
- 고도 0의 정사영은 기존 x/y 투영을 유지한다. **원근과 고도 탐색은 기존 순위와 동일하지 않으므로 별도 재평가가 필요하다.**

`scripts/render_camera_convention_v2.py`는 설치된 로컬 Blender만 사용한다. 기준 표식(앞 빨강, 뒤 파랑, 왼쪽 초록, 오른쪽 노랑)과 기존 캐시 FBX를 새 출력 폴더에 렌더한다. 원본 FBX 해시를 확인하고 수정하지 않는다.

기준 정면 정사영/원근에서 ray cast가 빨간 앞 표식을 먼저 만나는 것을 확인했다. yaw/elevation/roll이 있는 정사영/원근도 Blender와 수치 투영이 일치했다. 실제 모델은 `124637:p0 A0`, `131127:p1 A2`의 기존 포즈를 사용했다. 첫 사례는 동일한 평면 관절 배치에서 기존 등 표시가 앞면 표시로 바뀌는 것을 육안 확인했다.

이는 표시 규약 결함의 수정 증거이며, 모든 후보의 몸 방향을 올바르게 선택했다는 증거가 아니다. 카메라 v2를 라이브러리/검색/refine에 연결하거나 기존 검색 결과를 덮어쓰지 않았다.

검토 화면: [카메라 비교 HTML](../artifacts/rough_router_phase2_20260922/camera-qa/index.html).

## 4. 의미 자산 감사

현재 DB: 1,248 pose, 해시 `d30c39736fe7aea76f2a28212635905866124030a65e3db77bc320b0f7bc8360`.

기존 문서 616 unit / 1,232 member와 비교했다.

| 분류 | member 수 |
|---|---:|
| 현재 DB에 ID 없음 | 244 |
| ID는 있으나 현재 BVH 해시가 다름 | 493 |
| ID와 BVH 해시가 일치 | 495 |
| 현재 DB에 있으나 기존 문서에 없음 | 260 |

현재 DB의 BVH 파일 누락은 0이다. 원본·미러 양쪽이 모두 일치하는 기존 문서 unit은 0이며, 기존 빌드 5개 모두 현재 DB 해시와 다르다. 따라서 문서/측정값의 출처 검증을 생략하거나 manifest 해시를 교체하는 방식으로 연결할 수 없다.

`scripts/audit_search_assets.py`는 read-only 감사 후 `--refresh-facts`로 현재 DB의 BVH를 다시 측정할 수 있다. 기존 posecode 측정 함수를 재사용하고, BVH별 SHA256·DB SHA256·코드 버전·격리 정책 해시를 남긴다. 생성 도중 자산이 바뀌면 실패한다.

갱신 결과: 전체 1,248 중 기존 격리 정책 33개를 제외한 **1,215개 성공 / 측정 실패 0**. 파일은 `semantic-refresh/current-member-facts.jsonl`이다. 이는 현재 자산의 관절 관계 측정값이며 새 동작 태그나 완성된 E5 색인이 아니다. 다음 문서/색인은 이 출처에서 생성하고 v3.3의 봉인된 승격 자산과 구분해야 한다.

## 5. 변경·실행·롤백 기록

`artifacts/rough_router_phase2_20260922/`:

- `checkpoint.json`, `before/`, `changes.patch`: 이번 단계의 변경 전 파일과 변경 후 해시.
- `implementation-events.jsonl`: 단계별 실행 결과와 외부 전송 거부 기록.
- `phase2-test.log`: 자동 테스트 11/11.
- `camera-render.log`, `camera-qa/results.json`: Blender 렌더·검증 수치.
- `semantic-audit/`: 기존 의미 자산 감사.
- `semantic-refresh/`: 현재 BVH 측정 자료 및 출처.
- `vlm/`: 요청 식별자, 실패 attempt, 검수 HTML 준비 상태.
- `rollback-dry-run.json`: 복구 사전 검사.

코드 복원 사전 확인:

```bash
.venv/bin/python scripts/rollback_rough_router.py \
  artifacts/rough_router_phase2_20260922/checkpoint.json
```

실제 복원은 `--apply`를 붙인다. 이후 편집이 있으면 복원 전에 중단한다. 후속 실제 VLM 검수까지 적용된 상태에서 1차 단계까지 되돌리려면 **rough_vlm_review → phase2 → phase1 순서**로 복원한다. 산출물은 감사 자료로 보존한다. 실제 작업 트리에는 롤백을 실행하지 않고 dry-run만 확인한다.

## 6. 다음 작업

- [실제 VLM 검수](ROUGH_VLM_REVIEW_2026-09-22.md)에서 발견된 관계 쌍 처리·반신 확신도 문제 수정 후 미검수 11컷 재개. API 503/429가 남아 전체 검수는 미완료.
- 현재 측정값 기반 부위 문서/색인 생성과 A/B1 팩트 연결.
- 카메라 v2를 별도 검색·refine 실험으로 연결해 원근/고도 확장의 이득·회귀 확인.
- 그 결과가 준비된 뒤 전신/부위 retrieval을 SearchPlan 실행에 연결. 기존 응답 기본 경로는 검증 전까지 유지.
