# current-X → Human-Art M 포즈 rescue cascade 구현 명세

> 상태: 코드 구현 완료 / 실모델 shadow 검증 대기
> 갱신일: 2026-08-30
> 기준 코드: `src/pipeline.py`, `src/pose.py`, `src/pose_humanart.py`, `src/skeleton_extraction.py`

**문서 버전:** 1.2 · **전략:** current-X를 primary로 유지하고, 추출에 실패한 슬롯만
Human-Art M으로 복구

**근거:** `pose_bench/runs/threshold_sensitivity_20260829/REPORT.md`,
`pose_bench/runs/blind_scoring_20260829/RESULT_round1.md`,
`pose_bench/runs/humanart_detector_canary_20260828_codex/comparison/COMPARISON.md`

---

## 0. 문서 계약

### 0.1 목적

현재 production 기본 포즈 모델인 **current-X**가 full-image 추론과 슬롯별 crop 재시도까지
수행했는데도 특정 인물의 COCO-17 스켈레톤을 만들지 못했을 때, **Human-Art M을 전체 이미지에
딱 한 번 실행**해 그 미해결 슬롯만 복구한다.

핵심 불변식은 다음과 같다.

1. 자동 cascade는 current-X가 만든 정상 슬롯을 교체하지 않는다. 수동 요청은 지정 슬롯만 재검토한다.
2. Human-Art M은 컷당 최대 1회만 실행한다.
3. Human-Art M 후보가 애매하면 채택하지 않고 current-X 결과를 그대로 반환한다.
4. fallback 초기화·추론·배정 실패는 요청 전체 실패나 mock 대체로 이어지지 않는다.
5. rescue 슬롯은 v1에서 항상 `confidence="low"`, `refine_allowed=false`다.
6. 사람 수 계약은 VLM `num_people`이며 rescue로 새 사람 슬롯을 만들지 않는다.

### 0.2 범위

- primary: 기존 `RTMPoseModel` 기반 current-X full-image + crop
- fallback: 기존 `HumanArtPoseModel` 기반 Human-Art M full-image pose
  (current-X와 detector 계약이 같으면 동일 요청의 detector 박스를 재사용)
- 대상: current-X 복구가 끝난 뒤에도 `missing`/`invalid` 또는 `insufficient`인 VLM 슬롯
- 결과: 기존 `CutResult` 계약을 유지하며 인물별 `quality_trace`와 notes에 rescue lineage 추가

### 0.3 명시적 비범위

- action/view/relationship 태그 기반 후보 필터링
- 정상·partial·suspect 슬롯의 품질 비교 후 모델 교체
- Human-Art M을 crop 재시도 모델로 사용
- rescue 후보로 `rtm_provisional` 슬롯 생성
- rescue 스켈레톤을 이용한 자동 refine

### 0.4 현재 구현 상태

현재 저장소에는 `POSE_MODEL_VARIANT=cascade`, `CascadePoseModel`, 미해결 슬롯 전용
`rescue_slots`, pipeline/API 연결, runtime identity와 단위 회귀 테스트가 구현돼 있다.
고정 19장 local shadow와 슬롯 소유권 검수는 통과했다. production 승격 전 남은 조건은
Human-Art 라이선스 승인, 배포 하드웨어 동시성/worker matrix, live shadow와 5% canary다.

---

## 1. 설계를 고정한 측정 4개

| # | 측정 | 설계 함의 |
|---|---|---|
| 1 | current-X 실패 8건 중 **7건이 crop 재시도를 이미 거치고 `crop_no_improvement`** | 같은 모델을 crop에 다시 태우는 자리는 이미 소진됐다. 폴백을 **2단계 자리에 넣지 않는다** |
| 2 | humanart의 구조 5건이 **전부 `skeleton_source: full_image`** | 폴백은 **전체 이미지 재추론**이어야 한다. crop에 태우면 이 5건이 재현되지 않는다 |
| 3 | 2D 관절 정확도 블라인드 채점 **53.8%** (무승부 11/28, 대조쌍 0/4) | 품질 기반 트리거를 만들 근거가 없다. 트리거는 **이진 실패로만** |
| 4 | 임계값 스윕에서 실패 8건 중 **임계값 복구 가능 0건** | 폴백이 아니면 이 8명은 어떤 설정으로도 안 살아난다 |

보조: 위 비교 실험의 두 bundle은 **동일한 검출기 파일**
(`yolox_x_8xb8-300e_humanart-a39d44ed.onnx`)을 썼고, 둘 다 `self_detecting=True`라
파이프라인 경로가 같았다. production에서도 동일 detector라는 가정에 기대지 말고 양쪽
runtime identity의 detector model/hash를 계측해야 한다.

> **⚠️ 절대 수치의 외적 타당성.** 위 실험은 `FrozenVLM` 스텁으로 돌았다
> (`vlm_provider: frozen-r1-vlm`, `vlm_model: mock`). VLM 슬롯 박스가 곧 검출기 박스였고,
> `lower_body_visible`는 전부 True, `shot`은 항상 FULL_HALF, `count_confidence`는 19/19 high였다.
> **A/B 비교의 내적 타당성은 유지되지만**(양쪽이 같은 입력), "8/37 실패"라는 발동률은
> 실제 Gemini를 붙이면 달라진다. §7 계측이 필수인 이유다.

---

## 2. 결정 사항 (2026-08-30 확정)

| # | 항목 | 결정 |
|---|---|---|
| D1 | crop 재시도 예산 | **컷당 절대 최대 2회**, 심각도 순 정렬 |
| D2 | 폴백 트리거 범위 | **실패 슬롯만** — `state ∈ {missing, invalid}` 또는 `coverage == insufficient` (suspect 제외) |
| D3 | 폴백 채택 coverage 하한 | **`≥ reduced`** — 이 코퍼스에서 4명 복구 (`131056::p1`(sparse)는 채택 안 함) |
| D4 | 임계값 두 벌 | 트리거 `0.30` (current-X) / 채택 `0.35` (humanart manifest 프로필) |
| D5 | 교차 모델 duplicate 임계값 | **IoU `0.50` / 거리 `0.10`** — 별도 env로 분리 (실측 근거 §5.1) |
| D6 | fallback 모델 로딩 | **lazy** — 첫 발동 때만 초기화 |
| D7 | canary 게이트 | 기존 `POSE_CANARY_STAGE` 재사용 |
| D8 | 사용자 수동 폴백 | **v1 포함** — `/analyze`의 `rescue` form 필드 |
| D9 | 검출기 공유 | detector artifact/input 계약이 같을 때 session과 request-local bbox를 재사용 |
| D10 | 출구 조건 | 채택률 **50% 미만이면 cascade 제거**. wrong-owner 1건이면 즉시 current-X 롤백·승격 중단 |

### 2.1 운영 설정

```dotenv
POSE_BACKEND=rtmlib
POSE_MODEL_VARIANT=cascade
POSE_MODEL_URI=s3://<assets>/pose-models/humanart-m/<build>/manifest.json
POSE_MODELS_ROOT=/app/data/pose-models
POSE_MODEL_DOWNLOAD_BUDGET_SECONDS=300
POSE_CANARY_STAGE=shadow
POSE_STRICT=1

# 교차 모델 중복 제거 전용. 기존 SLOT_DUPLICATE_*와 별개다.
POSE_FALLBACK_DUPLICATE_IOU=0.50
POSE_FALLBACK_DUPLICATE_DISTANCE=0.10
SLOT_CROP_MAX_PER_CUT=2
SLOT_CROP_HARD_CAP=2
```

`POSE_MODEL_URI`는 cascade fallback인 Human-Art M의 원격 immutable manifest를 가리킨다.
기동 시 서버가 manifest와 선언된 ONNX 두 개를 staging에 받고 크기·SHA-256·runtime 계약을
검증한 뒤 `POSE_MODELS_ROOT/humanart-m/<build_id>/`에 원자적으로 공개한다. 검증 후 확정된
로컬 manifest 경로가 내부 `POSE_MODEL_MANIFEST` 값이 된다. 수동 read-only mount를 쓰는
환경은 `POSE_MODEL_URI` 없이 기존 `POSE_MODEL_MANIFEST` 절대 경로를 직접 지정할 수 있다.
운영 URI/root는 환경별 assets 버킷과 ECS task definition을 소유한 CDK/CloudFormation이
주입한다. 앱 배포 workflow는 기존 task definition의 URI를 확인만 하고 덮어쓰지 않는다.
다운로드/검증은 기본 300초 전체 예산과 S3/HTTP 요청당 최대 60초 제한을 사용하며,
`pose_model_bundle.durationMs` 로그로 staging 콜드 스타트를 실측한다. ECS health-check
grace period는 이후 current-X 초기화 시간까지 포함해 별도로 잡아야 한다.
S3 socket 제한은 botocore client `Config`가 소유한다. 작은 응답이 이미 버퍼링된 경우
하위 socket이 없을 수 있으므로 `StreamingBody.set_socket_timeout()`을 직접 호출하지 않는다.
current-X는 기존 `RTMPoseModel` runtime 설정을 그대로 사용한다.

### 2.2 두 calibration 프로필의 분리 (D4)

현재 `HumanArtPoseModel`의 직접 실행 경로는 manifest calibration과 전역 `CFG`가 정확히
같은지 검사한다. 그러나 cascade에서는 아래 두 값을 의도적으로 같이 사용한다.

| 용도 | 소유자 | threshold |
|---|---|---:|
| primary 분석·crop·기존 검색 mask | current-X / `CFG.skeleton_kpt_threshold` | 0.30 |
| Human-Art 후보 분석·채택 mask | Human-Art manifest | 0.35 |

따라서 cascade가 `validate_calibration_against_config()`를 그대로 호출하면 정상 manifest도
불일치로 실패한다. 구현 시 계약을 다음처럼 분리한다.

- `POSE_MODEL_VARIANT=humanart-m`: 기존처럼 manifest와 전역 CFG의 exact match를 요구한다.
- `POSE_MODEL_VARIANT=cascade`: manifest 자체의 hash/runtime/decoder/calibration 완전성은 그대로
  검증하되, Human-Art threshold는 manifest 값으로만 평가한다.
- current-X 설정은 바꾸지 않는다. fallback manifest의 검색 거리 임계값으로 전역 검색 설정을
  덮어쓰지 않는다.
- runtime identity에는 primary threshold와 fallback threshold/profile/hash를 함께 기록한다.

이를 위해 `HumanArtPoseModel`에 명시적 cascade 생성 경로를 두거나, bundle 검증과 전역 CFG
동등성 검증을 분리한다. 검증을 통째로 끄는 boolean은 사용하지 않는다.

---

## 3. 최종 플로우

```
process_cut
 └ _process_self_detecting                     (self_detecting=True 경로)
    1) current-X full-image  →  assign_candidates  →  슬롯 배정
  ★ 2) current-X crop 재시도   ← D1: 컷당 절대 최대 2회, 심각도 순
    3) finalize_slot
  ★ 3.5) ─── 폴백 ───────────────────────────────────────────────
         미해결 슬롯이 남아 있으면:
           humanart-M full-image pose 1회 (컷 단위)
           → 가드 3개 통과 후 대상 슬롯에만 배정
           → 배정된 슬롯만 재 finalize
    4) 검색 · unstable_search crop 재시도 (current-X 유지, 무변경)
```

**컷 단위인 이유.** 청룡성 컷은 실패 슬롯이 3개다. 전체 이미지 1회로 셋 다 복구된다.
슬롯 단위면 추론 3회. 폴백 예산은 **컷당 1회**.

### 3.1 detector session·bbox 재사용

current-X performance와 Human-Art M bundle은 같은 YOLOX-X Human-Art detector를 사용한다.
cascade는 detector artifact 경로(경로가 다르면 hash), input size, 동일 요청 lineage가 모두
맞을 때만 current-X detector session과 그 요청에서 이미 계산한 bbox를 재사용한다.

Human-Art M은 전체 이미지와 전체 이미지 좌표계 bbox를 받아 pose만 다시 추론한다.
따라서 `skeleton_source="fallback_full_image"` 계약은 유지되며 crop inference로 바뀌지 않는다.
bbox context는 모델의 공유 `last_result`에 저장하지 않고 `Pipeline` 호출 스택으로 전달하므로
동시 요청끼리 섞이지 않는다. 계약이나 lineage가 없으면 기존 전용 detector 경로로 복귀한다.

**3.5가 검색 앞인 이유.** 4단계의 `unstable_search` 재시도는 recall 실패가 아니라
검색 안정성 문제다. 트리거를 이진 실패로 한정했으므로(측정 3) 그 경로는 건드리지 않는다.

### 3.1 자동 트리거의 정확한 정의

```python
def needs_rescue(slot) -> bool:
    return bool(
        slot.slot_origin == "vlm"
        and slot.vlm_box is not None
        and (
            slot.skeleton is None
            or slot.state in {"missing", "invalid"}
            or slot.evidence is None
            or slot.evidence.coverage_class == "insufficient"
        )
    )
```

다음 신호는 **단독 트리거가 아니다.**

| 신호 | 이유 |
|---|---|
| `count_confidence="low"` | detector↔VLM 개수 불일치 신호이지 특정 슬롯 실패가 아님 |
| `state="suspect"` | 스켈레톤이 존재하며 crop·보수적 검색 경로가 이미 담당 |
| Top-1 거리 초과 | 라이브러리 공백일 수 있으며 포즈 추출 모델 교체 근거가 아님 |
| `search_stability="unstable"` | 검색 공간의 불안정이며 recall 실패와 분리해야 함 |
| entangled relationship | 2인 세트 검색 미구현 문제이며 단일인물 rescue로 해결하지 않음 |

### 3.2 상태 전이

| 시작 상태 | Human-Art 실행 | 채택 결과 | 최종 상태 |
|---|---:|---|---|
| 모든 슬롯 해결 | 아니요 | 해당 없음 | current-X 결과 그대로 |
| 미해결 슬롯 있음 | 컷당 1회 | 후보 없음/가드 실패 | 원래 실패 슬롯 그대로 |
| 미해결 슬롯 있음 | 컷당 1회 | 가드 통과 | 해당 슬롯만 `fallback_full_image`로 교체 |
| fallback 초기화/추론 오류 | 1회 시도 | 채택 없음 | current-X 결과 + 오류 lineage |

rescue가 성공해도 사람 배열의 cardinality와 기존 `slot_id`는 바뀌지 않는다. 최종
`person_index`는 기존과 동일하게 모든 처리가 끝난 뒤 왼쪽→오른쪽으로 정렬한다.

---

## 4. crop 재시도 수정 (D1)

### 4.1 현재 문제

```python
for slot in assignment.slots:                    # ← VLM이 박스를 준 순서 = 선착순
    if slot.vlm_box is not None and (slot.skeleton is None or slot.state in ("suspect","invalid")):
        try_crop(slot, "pre_search_suspect")     # crop_attempts >= 2 면 그냥 스킵
```

**청룡성에서 실제로 일어난 일:**

| | slot0 (x=106) | slot1 (x=493) | slot2 (x=4) |
|---|---|---|---|
| crop 재시도 | ✓ 1회차 | ✓ 2회차 | ✗ 예산 소진 |

응답은 `sort_slots_left_to_right`로 재정렬되어 x=4 슬롯이 `p0`으로 나온다.
2.16.52는 4명 전부 suspect인데 2명만 받았다.

예산 2회는 속도를 위한 의도된 상한이다. 해결할 문제는 **대상과 소유권**이다. 어느 슬롯이
기회를 받을지 VLM 박스 나열 순서에 맡기지 않고 심각도 순으로 고정해야 하며, crop에서 다시
검출된 인물이 실제 실패 슬롯의 소유자인지도 별도로 검증해야 한다.

### 4.2 수정

```python
SEVERITY = {"missing": 0, "invalid": 1, "suspect": 2}     # 낮을수록 먼저

needs_crop = [s for s in assignment.slots
              if s.vlm_box is not None
              and (s.skeleton is None or s.state in ("suspect", "invalid"))]
budget = min(                                         # D1: 최대 2회
    CFG.slot_crop_hard_cap,
    max(CFG.slot_crop_max_per_cut, len(needs_crop)),
)
for slot in sorted(needs_crop,
                   key=lambda s: (SEVERITY.get(s.state, 3) if s.skeleton is not None else 0,
                                  s.slot_id)):             # D1: 심각도 순, 동률은 안정 정렬
    try_crop(slot, "pre_search_suspect")
```

`slot_crop_max_per_cut`은 기본 예산, `slot_crop_hard_cap`은 **절대 상한 2회**다.
`try_crop` 안의 횟수 검사는 `budget`을 기준으로 한다. 실패 슬롯이 많아도 crop은 2회를
넘지 않으며, 나머지 미해결 슬롯은 컷당 1회의 Human-Art full-image 결과로 함께 복구한다.

crop 결과는 crop 안에서 처음 검출된 사람을 즉시 슬롯에 넣지 않는다. 각 후보를 전체 이미지
좌표로 복원한 뒤 다음 순서로 실패 슬롯 소유권을 검증한다.

1. `insufficient`·invalid·cross-slot 후보 제거
2. 이미 해결된 다른 슬롯의 스켈레톤과 중복인 후보 제거
3. 대상 VLM bbox와 모든 다른 VLM bbox에 대한 assignment cost 계산
4. 대상 슬롯이 다른 슬롯보다 `slot_assignment_ambiguity_margin` 이상 명확히 좋은 owner인지 확인
5. 동일인 중복 crop 후보는 하나로 축약하고, 서로 다른 후보가 비슷한 cost면 전부 거부
6. 기존 suspect 스켈레톤보다 구조 품질이 실제로 좋아진 후보만 `crop_retry`로 적용

이 매핑은 crop이 순차 실행돼도 먼저 복원된 인물을 이후 crop의 resolved 집합에 포함한다.
따라서 같은 사람이 두 실패 슬롯을 채우지 못하며, 애매하면 해당 슬롯은 미해결로 남아
Human-Art full-image rescue로 넘어간다. 판정값은 `quality_trace.crop_mapping`에 기록한다.

Human-Art로 채택된 슬롯의 검색은 current-X metric 설정과 분리한다. 전역 `DISTANCE`가
`angle` 또는 `hybrid`여도 `conservative_joint_mask(evidence)`를 query mask로 사용해
벡터화 `pos` 검색을 정확히 한 번만 실행한다. 반환 수도 전역 `top_k_final`과 분리해
**최대 5개 pose family**로 고정한다. base mask와의 A/B 검색이나 rescue 이후 current-X
crop은 실행하지 않으며, 후보가 없으면 X, 후보가 있으면 항상 low-confidence·refine 금지다.

> **주의.** 이 수정은 폴백과 **독립적으로 기본 경로의 crop 채택 결과를 바꾼다.**
> 별도 커밋으로 분리하고, 19장 재실행으로 before/after와 거부 사유를 기록한 뒤 폴백을 얹는다.

---

## 5. 폴백 배정 알고리즘 — 가드 3개

humanart는 컷 안의 **모든 인물**을 내놓는다(이미 잘 잡힌 a도 포함). 어느 것이 미해결 슬롯 b의
것인지는 **식별하지 않고 기존 기하 배정에 맡긴다.**

```
assignment_cost = 0.65·(1 − IoU(슬롯박스, 후보박스))
                + 0.35·min(2.0, ‖torso_center − 슬롯중심‖ / 슬롯대각선)
```

헝가리안에 슬롯마다 dummy 열이 `SLOT_ASSIGNMENT_MAX_COST=0.85`로 깔려 있어, 그보다 나쁜 후보는
배정되지 않고 `unmatched_slot`으로 남는다 — **"둘 다 b에 안 맞으면 b는 실패인 채로 둔다"가 공짜로 된다.**

### 가드 1 — 이미 가진 사람을 다시 배정하지 않기 (D5)

a'는 정의상 current-X가 이미 가진 a 스켈레톤과 거의 같은 사람이다.

```python
for cand in fallback_candidates:
    if any(bbox_iou(cand_box, s.skeleton_box) >= CFG.pose_fallback_duplicate_iou
           and duplicate_skeleton_distance(
               cand, s.skeleton, cand_box, s.skeleton_box,
               CFG.skeleton_kpt_threshold,  # 양쪽 모두 0.30
           ) <= CFG.pose_fallback_duplicate_distance
           for s in resolved_slots):
        drop(cand)                    # "이 사람은 이미 갖고 있다"
```

재측정에서 단일 `0.30/0.30`은 같은 사람 검출 96.4%, 다른 사람 오검출 16.3%였고,
모델별 `0.30/0.35`는 오검출 개선 없이 같은 사람 검출만 92.9%로 떨어졌다. 놓친 중복은
wrong-owner로 이어지는 위험한 실패이므로 **duplicate 판정만큼은 양쪽 모두 0.30**을 쓴다.
Human-Art manifest 0.35는 후보 coverage와 최종 채택 판정에만 사용한다. 공유
`duplicate_skeleton_distance` 시그니처는 변경하지 않는다.

**실패 방향이 안전한 쪽이다.** 이 가드가 놓치면(중복을 못 걸러내면) 다른 사람의 포즈가
b 슬롯으로 들어간다 — **위험**. 과하게 걸면 정당한 복구를 놓친다 — **안전**.
그래서 검출률을 높이는 방향으로 잡는다.

#### 5.1 실측 근거 (28명 겹침 집합, 2026-08-30)

| 분포 | n | 거리 p50 | p90 | p95 | IoU p50 |
|---|---|---|---|---|---|
| 같은 사람 · 교차 모델 (걸러야 함) | 28 | 0.0342 | 0.0673 | 0.0819 | 0.890 |
| 다른 사람 · 같은 컷 (걸리면 안 됨) | 43 | 0.5836 | 0.9622 | 1.0272 | 0.000 |

거리 분포는 한 자릿수 배로 갈린다. 임계값 스윕(거리 0.10 고정):

| IoU 게이트 | 같은사람 검출 | 다른사람 오검출 |
|---|---|---|
| 0.70 (기존값) | 89.3% | 11.6% |
| 0.60 | 92.9% | 11.6% |
| **0.50** | **96.4%** | **16.3%** |
| 0.40 이하 | 96.4% | 16.3% |

**채택: IoU `0.50` / 거리 `0.10`.** 놓치는 1건은 `131040::p2`(거리 0.585, IoU 0.253)로,
두 모델이 아예 다른 것을 잡은 케이스라 애초에 같은 사람으로 볼 수 없다.
오검출 16.3%는 박스가 거의 겹치는 슬롯(2.16.52의 x=1.3/1.4, x=289.6/290.1)에서 나오며,
안전한 방향의 실패다.

**기존 `SLOT_DUPLICATE_IOU=0.70` / `SLOT_DUPLICATE_KEYPOINT_DISTANCE=0.08`은 건드리지 않는다.**
일반 경로의 중복 판정이 바뀐다. 새 env 두 개로 분리:

```
POSE_FALLBACK_DUPLICATE_IOU=0.50
POSE_FALLBACK_DUPLICATE_DISTANCE=0.10
```

### 가드 2 — 유령 슬롯 생성 차단 (**가장 조용히 터질 버그**)

`assign_candidates`는 배정 안 된 후보 중 `coverage ∈ (full, reduced)`이고
`state ∈ (valid, partial)`인 것으로 **`rtm_provisional` 슬롯을 새로 만든다.**
a'는 정확히 이 조건에 해당한다(a는 잘 잡힌 인물이므로). 폴백 패스에서 해결된 슬롯을
`occupied`에 안 넘기면 `SLOT_PROVISIONAL_MAX_IOU=0.20` 검사를 통과해
**없는 세 번째 인물이 생긴다.**

→ 폴백 패스에서는 **provisional 생성을 끈다.** `assign_candidates`를 그대로 재사용하지 말고,
`pose_rescue.py`가 자체 배정을 하거나 provisional 생성을 파라미터로 비활성화한다.

### 가드 3 — 소유권 교차

```python
analyze_skeleton(cand, cand_box, fallback_kpt_thr,        # 0.35 (D4)
                 CFG.skeleton_torso_min_box_ratio,
                 owner_box=slot.vlm_box,
                 peer_boxes=[s.vlm_box for s in resolved_slots], cfg=CFG)
```

b'의 사지가 a 쪽으로 뻗어 있으면 `left_arm_cross_slot` 같은 관절 소유권 사유로 잡히고,
rescue 단계에서는 `cross_slot_ownership`으로 거부된다. 몸통 anchor 자체가 다른 슬롯에 있으면
`torso_cross_slot`이다.

### 채택 판정

```
채택 =  가드 1·2·3 통과
     ∧  assignment_cost ≤ SLOT_ASSIGNMENT_MAX_COST (0.85)
     ∧  coverage ∈ {full, reduced}                    ← D3
     ∧  assignment_margin ≥ SLOT_ASSIGNMENT_AMBIGUITY_MARGIN   ← 애매하면 거부
```

**보수적으로 가는 이유.** b가 실패하는 상황이 곧 기하 배정이 제일 약한 상황이다 — 실제 실패
데이터에 `merge_suspected`·`torso_cross_slot`·`assignment_ambiguous`가 붙어 있고, 이는 두 인물이
겹쳐 있다는 뜻이다. **잘못 살린 포즈가 3D 인형으로 나가는 것이 못 살린 것보다 나쁘다** —
사용자는 그럴듯하게 틀린 포즈를 알아채지 못한다.

채택된 슬롯: `skeleton_source = "fallback_full_image"`, `reasons`에 `"fallback:humanart-m"` 추가.

---

## 6. 파일별 변경

저장소의 모듈 경계 원칙에 따라 새 단계는 별도 모듈로 두고, pipeline은 모델 variant를
판별하지 않은 채 공통 인터페이스만 호출한다.

| 파일 | 변경 | 내용 |
|---|---|---|
| `src/config.py` | 수정 | fallback duplicate 값, crop hard cap 2, `POSE_MODEL_VARIANT=cascade` 허용 |
| `src/pose_cascade.py` | **신규** | primary 위임 + request-local detector context 전달. fallback은 lazy load (D6) |
| `src/pose_rescue.py` | **신규** | 폴백 배정 단계(§5). 순수 로직이며 모델을 모르고 `skeleton_extraction` primitive만 재사용 |
| `src/pose.py` | 수정 | current-X detector 결과를 request-local context로 반환; `cascade` 분기 |
| `src/pose_humanart.py` | 수정 | direct 검증, cascade calibration, 공유 detector session/bbox 입력 경로 분리 |
| `src/pipeline.py` | 수정 | crop 예산·순서, detector context 전달, finalize 뒤 `rescue_slots(...)` 호출. **variant if 분기 없음** |
| `src/runtime_guard.py` | 수정 | cascade의 primary/fallback identity 및 production 정책 검증 |
| `api/app.py` | 수정 | `/analyze rescue` form 필드와 health fallback 준비 상태 노출 |
| `api/models.py` | 수정 | `skeleton_source` 설명에 `fallback_full_image` 추가. 필수 필드 구조는 유지 |
| `docs/API_CONTRACT.md` | 수정 | rescue lineage·confidence·refine 차단·health 상태 문서화 |
| `standin_eval/fixtures.py` | 수정 | runtime identity와 rescue 호출 transcript로 cascade fixture 캐시 분리 |
| `scripts/run_pose_canary_eval.py` | 수정 | cascade 19장 실행과 `would_accept`/채택률·거부·지연 집계 |
| `scripts/compare_pose_canary.py` | 수정 | 채택률 50%와 wrong-owner 1건 D10 승격/롤백 게이트 |
| `.env.example` | 수정 | 새 env 주석 |
| `AGENTS.md`, `CLAUDE.md` | 수정 | 두 도구용 Repository Tree에 신규 2파일 등록 |
| `tests/test_pose_cascade.py` | **신규** | 트리거·lazy load·배정 가드·오류 복구·lineage 회귀 |

### 6.1 `src/pose.py` — 기본 no-op

```python
class BasePoseModel:
    ...
    def rescue_candidates(self, image, img_w: int, img_h: int) -> list[Skeleton]:
        """실패 슬롯 복구용 2차 후보. 기본은 없음 — 현행 모델의 동작은 바뀌지 않는다."""
        return []
```

이게 **호출부 if 분기를 없애는 핵심**이다. 파이프라인은 항상 부르고, cascade가 아니면
빈 리스트가 와서 `rescue_slots`가 즉시 no-op으로 빠진다.

### 6.2 `src/pose_cascade.py`

```python
class CascadePoseModel(BasePoseModel):
    self_detecting = True          # primary/fallback 둘 다 True — 경로 분기 없음
    model_id = "cascade"

    def __init__(self):
        self.primary = RTMPoseModel()          # current-X
        self._fallback = None                  # lazy (D6)

    # --- 1·2단계는 전량 위임: 기본 경로는 current-X 그대로 ---
    def estimate(self, image, boxes, img_w, img_h):
        return self.primary.estimate(image, boxes, img_w, img_h)

    def estimate_crop_candidates(self, image, box, img_w, img_h):
        return self.primary.estimate_crop_candidates(image, box, img_w, img_h)

    # --- 3.5단계 ---
    def rescue_candidates(self, image, img_w, img_h):
        return self._ensure_fallback().estimate(image, None, img_w, img_h)

    def fallback_kpt_threshold(self) -> float:
        """채택 판정용 임계값을 humanart 번들 manifest에서 읽는다. 하드코딩 금지 (D4)."""
        return self.bundle.calibration["skeleton_kpt_threshold"]

    def runtime_identity(self) -> dict:
        # primary/fallback 양쪽 identity를 합쳐 반환.
        # runtime_guard가 mock 폴백을 잡아내야 하므로 fallback 미초기화 상태도 표기한다.
```

`_ensure_fallback()`은 `HumanArtPoseModel()`을 만든다. `POSE_MODEL_MANIFEST`와
`POSE_CANARY_STAGE`(D7)가 필요하다.

### 6.3 API 계약

`POST /analyze`에 선택 form 필드 `rescue`를 추가한다.

| 값 | 동작 |
|---|---|
| `""` 또는 `"auto"` | 자동 실패 슬롯만 rescue |
| `"all"` | 모든 VLM 슬롯을 수동 재검토 |
| `"0,2"` | 최종 좌→우 `person_index` 0, 2만 수동 재검토 |

잘못된 형식, 중복 index, 음수, 100 이상 index는 `422 invalid_rescue_selector`다. 수동 요청도
duplicate·소유권·coverage·모호성 가드를 완화하지 않으며, `shadow` deployment에서는 배정만
계산하고 슬롯을 교체하지 않는다. 기존 `CutResult` 필수 구조는 유지하고 인물별
`quality_trace`, `quality_reasons`, `skeleton_source`, `confidence`, `refine_allowed`에 lineage를 싣는다.

채택된 인물의 예:

```json
{
  "confidence": "low",
  "skeleton_source": "fallback_full_image",
  "refine_allowed": false,
  "quality_reasons": ["fallback:humanart-m"],
  "quality_trace": {
    "pose_rescue": {
      "triggered": true,
      "model_id": "humanart-m",
      "accepted": true,
      "assignment_cost": 0.23,
      "assignment_margin": 0.18,
      "coverage_class": "reduced"
    }
  }
}
```

미채택 인물도 `pose_rescue.rejected_reason`을 남기되 내부 stack trace나 모델 파일 절대경로는
응답에 노출하지 않는다. 운영 상세 오류는 서버 로그/metric에만 남긴다.

### 6.4 `src/pipeline.py` 삽입 지점

```python
        with span("skeleton_finalize"):
            slots = [finalize_slot(slot, CFG) for slot in assignment.slots]

        # ★ 추가: 미해결 슬롯에 한해 2차 모델로 복구를 시도한다.
        with span("pose_rescue"):
            rescue = rescue_slots(slots, self.pose, image, img_w, img_h, CFG)
        if rescue.accepted:
            notes.append(f"폴백 복구 {rescue.accepted}명")

        threshold_scale = 0.7 if count_confidence == "low" else 1.0
```

`rescue_slots`는 대상 슬롯이 없으면 `self.pose.rescue_candidates`를 **부르지도 않는다**(추론 비용 0).

채택된 슬롯은 Human-Art manifest의 threshold로 다시 `analyze_skeleton`과 `finalize_slot`을
수행한다. 그 뒤 기존 descriptor/search 경로로 들어가므로 검색 피처 공간은 바뀌지 않는다.
다만 v1에서는 rescue 결과가 primary 실패에서 나온 것임을 보존하기 위해 검색 거리가 좋아도
`confidence="high"`로 승격하지 않는다. refine은 기존 `structural_refine_allowed`가 이미
`skeleton_source == "full_image"`만 허용하므로 `fallback_full_image`를 자동 차단한다. 코드 조건은
바꾸지 않고 회귀 테스트로 고정한다.

### 6.5 lazy 초기화와 오류 격리

- cascade 생성 시 current-X는 즉시 초기화한다.
- Human-Art manifest·artifact hash·runtime contract는 production에서 시작 시 검증한다.
- 무거운 Human-Art ONNX session은 첫 rescue 발동 때 lock 안에서 한 번만 만든다.
- 동시 요청이 들어와도 fallback instance는 하나만 생성한다.
- fallback 초기화나 추론이 실패하면 mock으로 바꾸지 않는다.
- 그 요청은 미해결 current-X 결과를 정상 응답하고 `pose_rescue.cut_summary.error` lineage를 남긴다.
- 오류 문자열은 분류 코드만 `quality_trace`에 기록하고 stack은 서버 로그에만 기록한다.
- health에는 `fallback_contract_ready`, `fallback_initialized`, `fallback_last_error`의
  공개 가능한 상태만 노출한다.

production 시작 시 manifest/라이선스/hash 계약이 잘못된 경우는 시작 실패다. 시작 후 일시적
추론 오류는 primary 결과를 보존하므로 요청 전체를 5xx로 만들지 않는다.

---

## 7. 계측 (v1 필수 — D10의 입력)

절대 수치가 stub VLM 기준이므로, 실제 값을 얻는 것이 v1의 주요 산출물이다.

대상 인물의 `quality_trace.pose_rescue.cut_summary`에 컷 단위로 남길 것:

```json
{
  "triggered": true,
  "trigger": "auto",
  "stage": "shadow",
  "unresolved_before": 3,
  "target_count": 3,
  "candidate_count": 4,
  "accepted": 0,
  "would_accept": 2,
  "rejected_reasons": ["duplicate_of_resolved", "ambiguous_margin"],
  "elapsed_ms": 384.2,
  "model_init_ms": 336.1,
  "error": null
}
```

집계 지표: 발동률(컷·인물 기준), **채택률**(shadow에서는 `would_accept / unresolved_before`,
canary에서는 `accepted / unresolved_before`),
거부 사유 분포, wrong-owner 수동 판정, 폴백 경로 p50/p95 지연, peak RSS,
초기화/추론 오류율.

**출구 조건 (D10):** 발동 대비 **채택률이 50% 미만이면 cascade를 제거**하고 current-X 단독으로
되돌린다. 블라인드 슬롯 소유권 검수에서 wrong-owner 채택이 1건이라도 나오면 즉시
`POSE_MODEL_VARIANT=current-x`로 롤백하고 승격을 중단한다. 원인을 수정한 뒤 고정 평가셋 전체를
다시 통과해야 재개할 수 있다.

별도로, 검색 결과 블라인드 채점에서 두 모델의 품질 차이가 없다고 나오면
**Human-Art 단독 교체**로 단순화하는 것이 더 낫다 — 그쪽이 더 빠르고 recall도 좋고 모델이 하나다.

### 7.1 2026-09-01 동일 19컷 재측정

고정 입력 19컷/37명, vectorized position search 활성화, 별도 프로세스의 current-X와
cascade shadow를 비교했다.

| 지표 | current-X | cascade shadow | 판정 |
|---|---:|---:|---:|
| request p95 | 1273.3ms | 1399.4ms | +9.9% (≤ +20%, 통과) |
| peak RSS | 1,652,211,712 | 1,754,202,112 | +6.2% (≤ +20%, 통과) |
| rescue p50/p95 | - | 33.8/46.5ms | 통과 |
| 채택률 | - | 4/8 = 50% | 통과 |
| wrong-owner | - | 0 | 통과 |

Human-Art 전용 detector session을 제거하기 전 같은 날 shadow 대비 RSS는 20.5%, rescue p95는
92.3% 감소했다. 배정 대상·candidate index·assignment cost·거부 사유는 최적화 전후 동일했다.

같이 적용된 vectorized position search는 descriptor search p50/p95를
36.9/188.0ms에서 1.2/5.2ms로 줄였다. 4,992 projection 행렬은 약 0.65MiB다.
이는 절대 지연을 낮추는 데 유효하지만 current-X와 shadow 양쪽에 동일하게 적용되므로,
Human-Art 상대 비용을 줄인 주된 변경은 detector session/bbox 재사용이다.

로컬 endpoint 동시성 측정과 라이선스/배포 하드웨어/live rollout의 현재 판정은
`docs/POSE_CASCADE_ROLLOUT_GATE_2026-09-01.md`에 고정한다. 이 측정은 deployment worker
SLO 증거가 아니며, Human-Art 라이선스가 `pending`인 동안 live shadow와 canary-5를 금지한다.

---

## 8. 테스트 · 완료 조건

**기본 경로 불변**
- [x] `POSE_MODEL_VARIANT` 미설정 시 기본 current-X 경로 유지
      (단, §4 crop 수정은 3명 이상 실패 컷에서 출력을 바꾼다 — **별도 커밋에서 before/after 기록**)
- [x] 기존 `tests/test_smoke.py`를 **수정하지 않고** 통과
- [x] `.venv`에서 `import src.pose, src.pose_cascade, src.pose_rescue` 성공

**폴백 동작**
- [x] `POSE_MODEL_VARIANT=cascade` + 미해결 슬롯 0개 → `rescue_candidates` **호출 안 됨**(추론 0회)
- [x] `POSE_STRICT=1` + 잘못된 manifest → mock 폴백이 아니라 **예외**로 죽음
- [x] fallback 첫 발동에서만 Human-Art session을 초기화하고 동시 요청에서도 1회만 생성
- [x] fallback 초기화/추론 예외 → 요청 5xx나 mock 대체 없이 current-X 결과 보존
- [x] current-X `0.30` / Human-Art manifest `0.35`를 동시에 사용해도 contract 검증 통과
- [x] direct `humanart-m` variant의 기존 exact CFG 검증은 유지
- [x] 교차 duplicate distance가 양쪽 모두 단일 0.30 threshold를 사용
- [x] 채택 하한이 `reduced`로 동작 — `131056` sparse 슬롯이 **채택되지 않음** (D3)
- [x] 19장 재실행 → 폴백 복구 **4명** (124637::p0 · 청룡성 3명)
- [x] rescue 성공 슬롯도 v1에서는 `confidence=low`, `refine_allowed=false`

**회귀 (가드)**
- [x] **유령 슬롯:** 2인 컷에서 a 성공/b 실패를 만들고, 폴백 후 인원수가 늘지 않음 (가드 2)
- [x] **중복 배정:** 폴백이 이미 해결된 슬롯의 인물을 미해결 슬롯에 넣지 않음 (가드 1, IoU 0.50/거리 0.10)
- [x] **소유권:** b'의 사지가 a 슬롯으로 뻗으면 `*_cross_slot` → `cross_slot_ownership`으로 거부 (가드 3)

**crop 수정 (D1)**
- [x] 실패 슬롯이 2개를 초과해도 crop 재시도는 **최대 2회**
- [x] 실패 슬롯 3개 이상이어도 심각도 상위 **2개만** crop 재시도를 받음
- [x] 이미 해결된 인물을 crop이 다시 검출해도 다른 실패 슬롯에 중복 배정하지 않음
- [x] crop 후보 순서를 섞어도 명확한 owner 슬롯에만 배정
- [x] 재시도 순서가 VLM 박스 순서가 아니라 심각도 순 — 박스 순서를 섞어도 결과 동일(재현성)

**기타**
- [x] fixture cache가 cascade runtime identity와 rescue transcript를 구분
- [x] `runtime_guard.ensure_production_backends`가 cascade primary/fallback 계약을 검사
- [x] `/analyze rescue` 형식 검증과 기존 `CutResult` 필수 필드 유지
- [x] `AGENTS.md`, `CLAUDE.md` Repository Tree에 신규 2파일 등록

---

## 9. 커밋 분리 · 롤아웃

**커밋 순서 (각각 독립적으로 되돌릴 수 있게)**

1. `crop 예산·순서 수정` — 폴백과 무관. 19장 before/after 기록 필수
2. `rescue_candidates 인터페이스 + no-op 기본 구현` — 동작 변화 0
3. `pose_rescue.py` 배정 로직 + 회귀 테스트
4. `pose_cascade.py` + registry
5. `/analyze rescue` 수동 경로
6. runtime identity·health·계측
7. 고정 19장 shadow 재실행 및 블라인드 슬롯 소유권 검수

**롤아웃 (D7)**

1. `POSE_CANARY_STAGE=shadow` — 폴백을 돌리고 배정까지 계산하되 결과 슬롯은 교체하지 않는다.
   발동률·가상 채택률·거부 사유·지연·RSS만 수집한다.
2. shadow 채택률이 50% 이상이고 wrong-owner가 0이면 `canary-5` → `canary-25` → …
3. wrong-owner 1건 발생 시 즉시 `POSE_MODEL_VARIANT=current-x`로 롤백하고 승격 중단
4. 각 단계에서 §7 지표 + hard fallback 비율 비회귀 확인
5. 채택률 50% 미만이면 cascade 제거

shadow 결과와 블라인드 소유권 검수 수를 함께 판정한다.

```bash
POSE_BACKEND=rtmlib \
POSE_MODEL_VARIANT=cascade \
POSE_MODEL_MANIFEST=<humanart-manifest.json> \
POSE_CANARY_STAGE=shadow \
POSE_STRICT=1 \
python scripts/run_pose_canary_eval.py \
  --variant cascade --boxes <frozen-boxes.json> --out <cascade-shadow-run>

python scripts/compare_pose_canary.py \
  --current <current-x-run> \
  --candidate <cascade-shadow-run> \
  --boxes <frozen-boxes.json> \
  --wrong-owner-count 0 \
  --out <comparison-dir>
```

`--wrong-owner-count 1` 이상이면 `promotion_decision`은
`rollback_current_x_and_stop_promotion`이고 명령은 rollback 전용 exit code `3`으로 종료한다. 운영 배포 설정을 즉시
`POSE_MODEL_VARIANT=current-x`, `POSE_CANARY_STAGE=off`로 되돌린 뒤 원인 수정 전까지 다음 단계로
진행하지 않는다. 채택률이 50% 미만이면 `remove_cascade_and_return_current_x`다.

`POSE_CANARY_STAGE`는 프로세스 내부 랜덤 샘플러로 사용하지 않는다. 5%·25% 트래픽 분리는
배포 라우팅에서 수행하고, stage 값은 해당 deployment의 정책과 runtime identity를 고정하는
표식으로 사용한다. 동일 요청이 재시도될 때 모델 경로가 무작위로 달라져서는 안 된다.

---

## 10. 하지 않는 것

- **humanart를 crop 단계에 태우지 않는다** — 측정 2. 이득이 full-image에서 나왔다
- **품질 기반 트리거를 만들지 않는다** — 측정 3. 근거가 없다
- **`unstable_search` 재시도 경로를 건드리지 않는다** — recall 실패가 아니다
- **동일 YOLOX detector를 cascade 요청 안에서 두 번 실행하지 않는다** — request-local bbox를 재사용한다
- **기존 `SLOT_DUPLICATE_*` 값을 바꾸지 않는다** — 일반 경로의 중복 판정이 바뀐다
- **임계값·후처리 튜닝을 같이 하지 않는다** — 별도 PR
- **fallback 실패를 mock 결과로 숨기지 않는다** — current-X 원본 결과와 명시적 오류 계측을 남긴다
