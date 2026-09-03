# Refine v2.5 백엔드·BFF 인계 계약

작성일: 2026-08-18  
대상: 앱 백엔드/BFF, Tauri 클라이언트, BVH export 담당  
서버 기준: `refine_version=v2.5.3`  
제품 기본: **safe aggressive** (`aggressive` 시도 후 selector가 `aggressive | conservative | base` 중 안전한 결과 선택)

> 배포 상태: v2.5.3 engineering closeout 완료. `131211:p1` single-leg extension과 불량 B0/관통
> 후보의 next Top-K 이관, 24개 D0 자동 안전·정확도·속도 gate를 통과했다. 작가 blind·실메시
> holdout·lap-contact 실표본은 대규모 운영 승격 증거로 별도 추적한다.

이 문서는 v1에서 v2.5로 바뀐 Refine HTTP 계약을 팀원에게 그대로 전달하기 위한 독립 문서다.
전체 서버 계약의 단일 소스는 `api/models.py`와 OpenAPI `/docs`이며, 이 문서는 소비자 관점의 매핑과
마이그레이션 규칙을 고정한다.

## 0. ⚠ 마이그레이션 필수 — lineage를 안 보내면 refine이 조용히 꺼진다

**이 서버 버전부터 `REFINE_V2_ENABLED=1`이 기본이다.** v2는 `/analyze`의 policy lineage를
fail-closed로 재검증하므로, 아래 조건을 **하나라도 못 채우면 요청이 전부
`refined=false`, `reason=skeleton_policy`로 떨어진다.**

```python
# src/refine_policy.py — 이 조건을 모두 만족해야 solver가 돈다
skeleton_state   in {"valid", "partial"}
coverage_class   in {"full", "reduced"}
refinable_limbs  비어 있지 않음
slot_origin      == "vlm"
skeleton_source  == "full_image"
```

기본값이 `None`이라 **필드를 생략하면 자동으로 불합격**이다. `refine_allowed`와
`refinable_limbs`만 보내던 기존 BFF는 이 배포 직후 refine이 100% 무효가 된다.

**이건 HTTP 오류가 아니라 정상 200 응답이다.** BFF가 `refined=false`를 정상 스킵으로
기록하면 로그·헬스체크·알림 어디에도 아무것도 남지 않는다. 조용히 꺼진 것을 알아채기 어렵다.

**대응 순서**

1. BFF가 §4 표의 lineage 필드를 모두 전달하도록 **먼저** 배포한다. 구 서버는 모르는 필드를
   pydantic이 무시하므로 서버보다 먼저 나가도 안전하다.
2. 배포 후 `diagnostics.mode_applied` 분포를 확인한다. 전부 `base`이고
   `reason=skeleton_policy`이면 lineage가 여전히 안 오고 있다는 뜻이다.
3. 전환 기간에는 `reason=skeleton_policy` 비율을 지표로 둔다.

---

## 1. 한눈에 보는 변경점

| 구분 | v1 | v2.5 제품 기본 | 표시 |
|---|---|---|---|
| endpoint | `POST /refine` | 동일 | **유지** |
| 기본 실행 | v1 팔·다리 미세조정 | safe aggressive | **변경** |
| 기본 config | `REFINE_V2_ENABLED=0` | `REFINE_V2_ENABLED=1`, `REFINE_DEFAULT_MODE=aggressive` | **변경** |
| 요청 mode | 없음 | `refine_mode?: conservative \| aggressive`; 생략/null은 aggressive | **추가** |
| 실행 게이트 | 큰 `search_distance`면 차단 | 거리·순위만으로 차단하지 않음 | **변경** |
| 필수 안전 lineage | 레거시 호출은 생략 가능 | `/analyze`의 policy 필드를 모두 전달, 누락 시 fail-closed | **변경** |
| 하체 가시성 lineage | 없음 | `lower_body_observed`; 명시적 true가 아니면 모든 하체 조정 동결 | **추가** |
| 하체 | 제한적/기본 비활성 | lower pair 강화, 제한적 발목 counter-rotation | **변경** |
| 손·접촉 | 개별 사지 위주 | hand pair, lap contact 목적 추가 | **변경** |
| 안전 선택 | 단일 solve 결과 | 원본 B0 구조 검사 + conservative 대비 metric selector + FINAL 충돌 재검사 | **추가** |
| 불량 asset | 선택 후 실패 가능 | 검색에서 quarantine하고 stale BVH/refine 요청은 409 | **추가** |
| 실패 처리 | base 복구 | aggressive → conservative → base exact fallback | **강화** |
| 응답 | 결과·손실·사지 진단 | mode/selector/time budget/cache/version lineage 추가 | **추가** |
| 조정본 전달 | `GET /refined/{handle}/bvh` | 응답 `bvh` 본문(무상태). `bvh_url`은 항상 베이스 | **변경** |
| 조정본 보관 | 추론 서버 로컬 디스크 | BFF의 `refined_artifacts`/S3 | **변경** |

`aggressive`라는 이름은 raw 공격 결과를 직접 반환한다는 뜻이 아니다. 서버는 먼저 conservative를 만들고,
aggressive 후보가 원본 구조 안전과 공통 metric non-regression을 모두 통과한 경우에만 채택한다.

## 2. 전체 호출 흐름

```text
Tauri/BFF                                      추론 서버
   | POST /analyze (PNG)                           |
   |---------------------------------------------->|
   | CutResult: person별 keypoints + Top-5         |
   |<----------------------------------------------|
   |                                               |
   | 작가가 person별 후보 1개 선택                 |
   |                                               |
   | POST /refine (선택 후보 + person lineage)     |
   |---------------------------------------------->|
   | RefineResponse: 조정본 bvh 본문 + 베이스 URL  |
   |<----------------------------------------------|
   |                                               |
   | bvh를 refined_artifacts/S3에 저장 후 뷰어·export가 소비 |
```

- Top-5를 먼저 모두 refine하는 계약이 아니다. 현재 제품 계약은 **Top-5 표시 → 1개 선택 → `/refine` 1회**다.
- 이미지나 RTMPose를 `/refine`에 다시 보내지 않는다. `/analyze`가 준 17개 keypoint와 score를 재사용한다.
- 조정본을 얻는 유일한 경로는 응답의 `bvh` 본문이다. 두 번째 GET은 없다.
- `bvh_url`은 결과물 위치가 아니라 **베이스 폴백 위치**다. `refined` 여부와 무관하게 항상 베이스를 가리킨다.

## 3. 관련 추론 서버 endpoint

| 메서드 | 경로 | 용도 |
|---|---|---|
| `POST` | `/analyze` | 러프 이미지에서 사람별 skeleton과 Top-5 후보 생성 |
| `POST` | `/refine` | 선택 후보 1개를 safe aggressive 정책으로 조정 |
| `GET` | `/pose/{pose_id}/bvh` | 원본 라이브러리 BVH 다운로드 |
| `GET` | `/healthz` | 서버·라이브러리·Refine capability 확인 |
| `POST` | `/export-order` | 기존 원본 pose 선택 주문서 생성. 조정본 전달 제한은 §8 참조 |

버전 prefix는 없다. 예: `/refine`이 맞고 `/v1/refine`은 아니다.

## 4. `/analyze`에서 `/refine`으로 옮기는 필드

작가가 `people[i].candidates[j]`를 선택했다고 가정한다.

| `/refine` 필드 | 소스 | 필수/정책 |
|---|---|---|
| `pose_id` | `people[i].candidates[j].pose_id` | 필수 |
| `view` | `people[i].candidates[j].view` | 필수 |
| `keypoints` | `people[i].keypoints` | 필수, 17×2 그대로 |
| `scores` | `people[i].scores` | 전달 권장, 17개 그대로 |
| `search_distance` | `people[i].candidates[j].distance` | 전달 권장, v2에서는 진단 lineage |
| `refine_allowed` | `people[i].refine_allowed` | **v2 필수 정책값** |
| `refinable_limbs` | `people[i].refinable_limbs` | **v2 필수 정책값** |
| `lower_body_observed` | `people[i].lower_body_observed` | **v2.5 필수**. true가 아니면 모든 하체 조정 동결 |
| `skeleton_state` | `people[i].skeleton_state` | **v2 필수 lineage** |
| `coverage_class` | `people[i].coverage_class` | **v2 필수 lineage** |
| `slot_origin` | `people[i].slot_origin` | **v2 필수 소유권 lineage** |
| `skeleton_source` | `people[i].skeleton_source` | **v2 필수 소유권 lineage** |
| `search_stability` | `people[i].search_stability` | partial 안정성 lineage |
| `distance_metric` | `people[i].distance_metric` | 전달 권장 |
| `confidence_threshold` | `people[i].confidence_threshold` | 전달 권장 |
| `gap_type` | 별도 평가 라벨 | 모르면 `unknown`; 실행 게이트가 아님 |
| `refine_mode` | BFF override | **생략 권장**. 생략/null이면 서버 safe aggressive 기본을 따름 |

v2.5는 policy lineage를 fail-closed로 재검증한다. `refine_allowed=true`만 보내고 나머지를 만들거나 추정하지
말고, 같은 `/analyze` 응답의 person 필드를 그대로 보낸다.

## 5. `POST /refine` 요청

```http
POST /refine
Content-Type: application/json
```

권장 요청 예시:

```json
{
  "pose_id": "Sitting Idle_01",
  "view": "front",
  "keypoints": [[120.5, 88.0], [121.4, 91.2], [119.1, 91.0], [125.0, 94.0], [115.0, 94.1], [142.0, 156.0], [101.0, 155.0], [151.0, 210.0], [92.0, 208.0], [132.0, 244.0], [111.0, 244.0], [137.0, 260.0], [105.0, 260.0], [141.0, 326.0], [103.0, 326.0], [145.0, 390.0], [101.0, 390.0]],
  "scores": [0.91, 0.87, 0.86, 0.82, 0.81, 0.94, 0.93, 0.90, 0.89, 0.88, 0.87, 0.95, 0.94, 0.92, 0.91, 0.90, 0.89],
  "search_distance": 0.21,
  "refine_allowed": true,
  "refinable_limbs": ["left_arm", "right_arm", "left_leg", "right_leg"],
  "lower_body_observed": true,
  "skeleton_state": "valid",
  "coverage_class": "full",
  "slot_origin": "vlm",
  "skeleton_source": "full_image",
  "search_stability": "not_required",
  "distance_metric": "pos",
  "confidence_threshold": 0.45,
  "gap_type": "unknown"
}
```

`refine_mode`를 생략한 것이 정상 제품 호출이다. 운영/디버그에서 conservative를 강제할 때만
`"refine_mode": "conservative"`를 추가한다.

검증 규칙:

- `view`: `front | three_quarter | side | back`
- `keypoints`: 유한한 숫자의 정확히 17×2 배열
- `scores`: 생략 가능, 전달 시 유한한 0 이상 숫자 17개
- `refine_mode`: `conservative | aggressive | null`
- `gap_type`: `near_gap | structural_gap | unknown`
- `lower_body_observed`: `/analyze` 값을 그대로 전달. 생략/null/false는 모두 하체 비관측으로 처리

## 6. `POST /refine` 응답

```json
{
  "pose_id": "Sitting Idle_01",
  "view": "front",
  "refined": true,
  "reason": "ok_partial",
  "bvh_url": "/pose/Sitting Idle_01/bvh",
  "bvh": "HIERARCHY\nROOT Hips\n...\nMOTION\nFrames: 1\nFrame Time: 0.033333\n...",
  "loss_base": 0.599,
  "loss_final": 0.094,
  "gain": 0.843,
  "backend": "scipy+numpy",
  "refine_version": "v2.5.3",
  "refine_outcome": "improved",
  "limbs": ["left_arm", "right_arm", "left_leg", "right_leg"],
  "limb_decisions": {},
  "diagnostics": {
    "mode_requested": "default",
    "mode_effective": "aggressive",
    "mode_applied": "aggressive",
    "aggressive_attempted": true,
    "candidate_status": "generated",
    "selector": {
      "version": "v2.5.3",
      "accepted": true,
      "selected_mode": "aggressive",
      "selected_variant": "full",
      "selected_alpha": null,
      "fallback_stage": null,
      "fallback_reason": null,
      "structural_checks": {"passed": true, "violations": []},
      "metrics": {}
    },
    "time_budget": {
      "total_ms": 5000.0,
      "prepare_ms": 0.5,
      "conservative_ms": 1120.2,
      "aggressive_ms": 1580.1,
      "selector_ms": 36.4,
      "final_postcheck_ms": 4.8,
      "remaining_ms": 2189.6,
      "elapsed_ms": 2737.2
    },
    "context": {
      "base_bvh_sha256": "...",
      "refine_config_sha256": "...",
      "pose_library_version": "v1",
      "deployment_version": "...",
      "feature_version": 1,
      "refine_mode_requested": "default",
      "refine_mode_effective": "aggressive"
    }
  }
}
```

소비자가 의존해야 하는 핵심 필드는 `pose_id`, `view`, `refined`, `reason`, `bvh`, `bvh_url`,
`refine_version`, `refine_outcome`이다. `limb_decisions`와 `diagnostics`는 로깅·평가·장애 분석용이며
키가 추가될 수 있으므로 BFF에서 폐쇄형 DTO로 잘라내지 말고 JSON object로 보존하는 편이 안전하다.

### mode 필드 의미

| 필드 | 의미 |
|---|---|
| `mode_requested` | 요청 값. 생략/null이면 `default` |
| `mode_effective` | 서버 config를 반영해 실제 시도한 mode. 현재 기본 `aggressive` |
| `mode_applied` | 최종 BVH의 출처: `aggressive | conservative | base` |

`mode_effective=aggressive`인데 `mode_applied=conservative` 또는 `base`인 것은 정상 안전 fallback이다.
`selector.selected_variant=global_blend`이면 full aggressive가 아니라 C→A 안전 부분 채택본이며,
`selected_alpha`는 `0.75 | 0.5 | 0.25` 중 하나다. BFF는 이 값으로 URL을 재구성하지 않고 진단 로그로만
보존한다.

### v2.5.3 하체 동결·single-leg extension·FINAL 관통 fallback

- `lower_body_observed=true`인 person만 leg block, lower-pair, leg-driving lap-contact, ankle
  counter-rotation을 시도한다. BFF가 필드를 누락하면 서버는 안전하게 하체를 전부 동결한다.
- mode 선택 후 서버가 실제 FINAL BVH를 다시 검사한다. B0 대비 신규·악화 관통 또는 검사 불능이면
  `refined=false`, `reason=final_collision_gate`, `mode_applied=base`, 원본 `/pose/.../bvh`를 반환한다.
- BFF/UI는 이 경우도 HTTP 200 정상 fallback으로 처리하고 응답 `bvh_url`만 사용한다.
- 이 검사는 BVH capsule proxy이므로 실메시 안전을 보증하는 API 신호로 해석하지 않는다.
- 단축 투시 한쪽 다리는 `single_leg_extension` 진단이 생길 수 있다. BFF가 해석하거나 별도 mode로
  호출할 필요는 없으며 성공 시 `reason=ok_foreshortened_extension`으로 기록된다.
- geometry `/analyze`는 quarantine pose를 Top-K에서 제외하고 다음 후보를 채운다. 이전 cache의 선택으로
  `/pose/{id}/bvh` 또는 `/refine`이 `409 pose_quarantined`를 반환하면 같은 person의 다음 Top-K 선택으로
  되돌리고 해당 pose를 export하지 않는다.

### `refined=false` 처리

`refined=false`는 HTTP 오류가 아니다. 안전 게이트, no-gain, timeout 또는 정책 차단으로 원본을 반환했다는
뜻이며 `bvh_url=/pose/{pose_id}/bvh`가 온다. UI와 export는 실패 팝업 대신 해당 URL을 정상 사용한다.

## 7. BFF 구현 규칙

권장 TypeScript 형태:

```ts
type RefineMode = "conservative" | "aggressive";

interface RefineRequestV25 {
  pose_id: string;
  view: "front" | "three_quarter" | "side" | "back";
  keypoints: number[][];
  scores?: number[];
  search_distance?: number;
  refine_allowed: boolean;
  refinable_limbs: string[];
  lower_body_observed: boolean;
  skeleton_state: string;
  coverage_class: string;
  slot_origin: string;
  skeleton_source: string;
  search_stability?: string | null;
  distance_metric?: string | null;
  confidence_threshold?: number | null;
  gap_type?: "near_gap" | "structural_gap" | "unknown";
  refine_mode?: RefineMode | null;
}

interface RefineResponseV25 {
  pose_id: string;
  view: string;
  refined: boolean;
  reason: string;
  bvh_url: string;          // 항상 베이스 /pose/{id}/bvh (폴백 위치)
  bvh: string | null;       // 조정본 본문(LF). refined=true일 때만. 유일한 소스
  loss_base?: number | null;
  loss_final?: number | null;
  gain?: number | null;
  backend: string;
  refine_version: string;
  refine_outcome: "improved" | "unchanged" | "reverted" | "not_attempted";
  limbs: string[];
  limb_decisions: Record<string, unknown>;
  diagnostics: Record<string, unknown>;
}
```

BFF 체크리스트:

1. 선택한 candidate와 **같은 person**의 skeleton/policy lineage 및 `lower_body_observed`를 조합한다.
2. 기본 제품 호출에서는 `refine_mode`를 생략한다. BFF 자체 기본값을 중복 보관하지 않는다.
3. `refined=true`면 응답 `bvh` 본문을 자기 영속 저장소(`refined_artifacts`/S3)에 저장한다.
   `refined=false`면 `bvh`는 `null`이므로 `bvh_url`의 베이스를 그대로 쓴다.
4. 조정본 URL을 BFF가 합성하지 않는다. 추론 서버에 조정본 다운로드 경로는 존재하지 않는다.
5. `reason`, `mode_effective`, `mode_applied`, `selector.fallback_reason`, phase timing을 로그에 남긴다.
6. 추론 solver 예산은 현재 5초다. 네트워크 여유를 포함해 BFF upstream timeout은 **최소 7초**로 둔다.
7. 추론 서버는 무상태다. 같은 선택을 다시 눌러도 재계산하지 않는 멱등성은 BFF의
   `refined_artifacts` PK `(job_id, person_index, candidate_id)`가 담당한다.
8. 서버 재배포 후 `GET /healthz`의 `refine.code_version`, `config_sha256`, `default_mode`를 배포 로그에 남긴다.

## 8. Export/BFF에서 반드시 바꿀 부분

**주의:** 현재 `POST /export-order` 요청은 `person_index`, `pose_id`, `view`만 받고 DB에서 원본
`/pose/{pose_id}/bvh`를 다시 채운다. 따라서 refine 후 기존 `/export-order`만 호출하면 조정본이 아니라
원본 BVH가 export될 수 있다.

V3.2 Converter 통합 뒤 BFF의 확정 규칙은 다음과 같다.

- `refined=true`면 `RefineResponse.bvh` 문자열을 UTF-8로 인코딩한 바이트가 최종 BVH다.
- `refined=false` 또는 refine 미호출이면 base `bvh_url` 응답의 원본 바이트가 최종 BVH다.
- `RefineResponse.bvh_url`은 refined 여부와 무관하게 항상 베이스다. `refined=true`에서 이 URL을
  최종 소스로 쓰면 조정이 조용히 사라진다.
- 현 `POST /export-order`는 **원본-only legacy 경로**로 취급한다.
- BFF는 최종 바이트 SHA256을 lineage에 기록하고 내부 Converter API에 multipart 업로드한다.

내부 Converter 성공 응답의 `X-Standin-Source-BVH-SHA256`이 BFF가 계산한 최종 SHA와 일치해야
한다. 상세 계약은 `FBX_CONVERTER_V3_2_PHASE3_BFF_HANDOFF.md`를 따른다.

추론 서버는 조정본을 저장하지 않는다. 따라서 추론 태스크 수·롤링 배포와 무관하며, 단일 태스크
운영 제약도 없다. 대신 **응답을 받은 BFF가 곧바로 보관하지 않으면 조정본은 사라진다** —
`bvh`를 버리고 나중에 다시 받으려 하면 재계산이다.

## 9. 오류·timeout 계약

| 상황 | HTTP/응답 | BFF 처리 |
|---|---|---|
| unknown `pose_id` | 404 | 사용자 재선택 또는 분석 결과 만료 처리 |
| quarantined `pose_id` | 409, `pose_quarantined` | 같은 person의 다음 Top-K로 재선택; export 금지 |
| BVH 파일 없음/파싱 실패 | 409 | 서버 데이터 장애로 기록 |
| shape, NaN/Inf, 잘못된 enum | 422 | BFF 매핑 버그로 기록 |
| solver timeout | **200**, `refined=false`, base `bvh_url` | 정상 fallback으로 소비, timeout metric 기록 |
| FINAL 신규·악화 관통/검사 불능 | **200**, `reason=final_collision_gate`, base `bvh_url` | 정상 exact fallback + 안전 metric 기록 |
| FINAL single-leg angle/ankle gate 실패 | **200**, `reason=final_extension_gate`, base `bvh_url` | 정상 exact fallback + 품질 metric 기록 |
| 정책 lineage 불충분 | **200**, `reason=skeleton_policy`, base `bvh_url` | 정상 fallback + BFF mapping 점검 |

## 10. Health check와 배포 확인

`GET /healthz`의 현재 기대값:

```json
{
  "ok": true,
  "refine": {
    "enabled": true,
    "v2_enabled": true,
    "default_mode": "aggressive",
    "selector_enabled": true,
    "config_valid": true,
    "code_version": "v2.5.3",
    "supported_modes": ["conservative", "aggressive"],
    "config_sha256": "...",
    "pose_quarantine_sha256": "..."
  }
}
```

production에서 aggressive 기본인데 selector가 꺼져 있으면 `config_valid=false`, health는 HTTP 503이다.

## 11. 운영 rollback

강도 순으로 다음 환경변수를 사용한다.

```env
# v2.5 보수적 결과만 사용
REFINE_DEFAULT_MODE=conservative

# v1로 복구
REFINE_V2_ENABLED=0

# refine 전체 비활성, 항상 원본 사용
REFINE_ENABLED=0
```

> ⚠ **`REFINE_V25_SELECTOR_ENABLED=0`을 단독 비상 스위치로 쓰지 말 것.**
> production에서 `기본 aggressive + selector off`는 안전하지 않은 구성이라
> `/healthz.refine.config_valid=false` → **HTTP 503**이 된다. ALB가 전 태스크를 unhealthy로
> 보고 교체 루프에 들어간다. selector를 꺼야 한다면 `REFINE_DEFAULT_MODE=conservative`를
> **반드시 함께** 설정한다. 애초에 selector off는 raw aggressive를 우회 반환하는 경로라
> 허용하지 않으며, 강도를 낮추려면 mode를 내리는 것이 정상 경로다.

변경 후 프로세스를 재시작하고 `/healthz.refine.config_sha256`가 바뀌었는지 확인한다.

## 12. 팀 인계 완료 체크리스트

- [ ] BFF가 `/analyze` person lineage와 `lower_body_observed`를 빠짐없이 `/refine`에 전달한다.
- [ ] **§0을 읽었고, lineage 누락 시 refine이 조용히 꺼진다는 점을 배포 순서에 반영했다.**
- [ ] `reason=skeleton_policy` 비율을 전환 기간 지표로 두었다.
- [ ] 생략된 `refine_mode`가 safe aggressive라는 점을 반영했다.
- [ ] `refined=false`도 성공 응답으로 처리하고 `bvh_url`을 사용한다.
- [ ] BFF upstream timeout이 7초 이상이다.
- [ ] 최종 export가 `/refine` 응답의 `bvh` 본문을 잃지 않고 Converter에 전달한다.
- [ ] `/export-order`의 원본-only 제한을 backend/export 담당이 인지했다.
- [ ] BFF 계산 final BVH SHA와 Converter source BVH SHA를 대조한다.
- [ ] refined export artifact와 최종 FBX는 BFF가 자기 저장소 기준으로 조립한다.
- [ ] 배포 시 `/healthz.refine` capability를 검증한다.
- [ ] 장애 시 conservative → v1 → disabled rollback 순서를 공유했다.

관련 소스:

- HTTP 모델: `api/models.py`
- endpoint·cache·health: `api/app.py`
- v2.5 selector: `src/refine_selector.py`
- 전체 상세 계약: `docs/API_CONTRACT.md`
- Refine 설계: `docs/REFINE_V2_DESIGN.md`
- 기존 export 계약: `docs/EXPORT_CONTRACT.md`
