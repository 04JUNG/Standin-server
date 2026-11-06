# Top-5 이전 source 후보 리파인과 선택 후 체형 적용·최종 검증

작성: 2026-11-06 KST. 개정: 사용자 체형 적용 시점 정정 반영. 상태: 구현 지시용 설계이며 운영 승격·v3/v4 제품 수용·속도 목표 달성을 선언하지 않는다.

## 1. 최신 사용자 결정과 흐름

> “체형을 입히는건 사용자 선택 이후 단계로 하자. 즉 최초검색->리파인 단계에서는 체형을 입히지 않는거야.”

이 지시가 이전 설계보다 우선한다. **Top-5 전 현재/기본/자동 target 체형을 후보에 입히던 제안은 취소한다.**

**최초 검색 → 체형 미적용 source 후보별 리파인 → Top-5 → 사용자 포즈 선택 → 실제 체형·수치·손·범위 적용 → 필요한 target 보정·조합/메시/파일 검증 → 최종 미리보기·저장.**

Top-5에서 고르는 것은 러프에 보정된 source 포즈다. 실제 체형의 최종 인체 표면은 선택 이후에 보여준다. 체형 비례·target 보정에 따른 외관/자세 차이를 숨기지 말고 최종 미리보기를 갱신한 뒤 그 동일 결과를 저장한다. source 포즈를 최종 target 메시라고 표시하지 않는다. 앱/플로팅 바는 같은 결과·상태를 사용한다.

시간 최적화는 PNG 굽기 제외, 분석 근거 1회 재사용, source 계산·표시 재사용, 중복 작업 제거·제한 병렬, 선택한 후보 하나의 target/출력 재사용 순으로 한다. 안전 기준을 낮추지 않는다.

## 2. 조사한 기준과 실제 차이

읽기 전용 조사에서 확인한 최신 원격 refs:

| 저장소 | main | develop |
|---|---|---|
| 서버 | `153ddb2d7c0cdb3b3bb7b814d7ccab949a51fdbd` | `f260b1e6843996935ab287cdfad63465c18f7266` |
| BFF | `b6a13e0ee4aa8b44220cf93f4e7c8881461c8468` | `b4a943019de97d661c574216a505e2407a613c33` |
| 클라 | `ef56d4f5dbc58489f205ebc078fbf0f7064cd7f8` | `5934f0d2799d64074cb5836b9d7bab48d7246f1e` |

운영 추론 image tag는 서버 main `153ddb2…`, 운영 BFF image digest의 ECR 태그는 `b6a13e0…`였다. 스테이징 BFF는 `b4a9430…`, 추론 runtime 소스에는 `efa8d68…` 태그가 있으며 최신 서버 develop과 동일하다고 간주하지 않는다. GitHub 최신 정식 앱 릴리스는 `v0.2.8`, target `3b228910160e20d6fab3519bc450638b186e64ea`였으며 해당 코드도 선택 후보만 보정한다. 모든 사용자 설치가 최신 릴리스라는 뜻은 아니다.

- 운영 source refine은 v2.5.3, develop은 v2.5.4. BFF `jobs/runner.ts`는 분석 결과·관측을 저장한 뒤 completed를 기록하며 Top-5 전체 리파인은 하지 않는다.
- 클라 `useSelectionReview → useRefineSelection`은 선택 후보만 호출하고 현재 인물 간 동시 호출 상한은 3이다. 이 상한을 5후보×인물 수로 곱하지 않는다.
- 기존 `/refine`은 후보 BVH·러프 관절·카메라·정책으로 동작하고 target 체형·손을 입력받지 않는다. v3의 source 실험과 target 프로필/메시 수용은 구분된다. develop의 v3 runtime은 off/shadow이며 실제 적용 인증이 없다.
- v4는 단순한 `/refine` 버전 교체가 아니다. 장면 worker와 명시 지시를 받는 native parts HTTP 경로가 있고 현재 parts는 검토한 현재 FBX·수정 지시를 요구한다. scene/parts API는 단일 슬롯이며 busy에 503/Retry-After를 반환한다.
- 운영·스테이징 추론 ECS는 각각 1 vCPU, 6GiB, 서비스 running 1이었다. 병렬 수를 늘렸다고 지연이 줄었다고 주장할 수 없다. 실제 v4 worker 자원은 별도로 측정한다.
- 기존 후보 썸네일/정적 모델 경로와 선택 후 refined artifact resolver는 별개다. 계산만 앞당겨도 카드가 보정본으로 바뀌지 않는다.

참조: `api/models.py::RefineRequest`, `api/app.py::refine`, `src/refine_v3_contracts.py`, `src/refine_v3_runtime.py`, `docs/REFINE_V3_AUTO_FLOW_DESIGN_2026-10-22.md`, `docs/REFINE_V4_PARTS_HTTP.md`, BFF `src/refine/{service,store,eligibility}.ts`, `src/interactive-state/{selection,static}.ts`, 클라 `useRefineSelection.ts`.

## 3. 선택 전 source 보정 — target 체형 적용 금지

러프 분석·관절 추출·가림·접촉/지지 근거는 한 번만 확보한다. 각 적격 후보의 원본 BVH SHA와 검색 카메라를 고정해 수용된 source solver를 실행한다. source 골격 FK의 기준 길이·좌표계를 사용하는 것과 사용자 target 체형 적용은 구분한다. 관측 불가 관절·좌우 귀속·하체 가림·격리·참고 후보 정책은 그대로 유지한다. 같은 이미지를 후보별 VLM에 재전송하지 않는다.

사전 prepare 입력에 target asset/profile/body/shape/hand 선택·사용자 export scope·appearance를 받지 않는다. 사용자 체형 설정을 source 조건으로 고정하지 않는다. source 결과는 피부 메시 안전이나 최종 FBX 인증이 아니다.

v3/v4 중 관측·source 골격만으로 처리 가능한 부분을 source 어댑터로 재사용한다. 실제 body/skin/current FBX를 요구하는 v3 target/v4 native parts는 사용자 선택 후에 둔다. 기본 target을 미리 입히거나 user_confirmed를 허위로 만들어 이 경계를 우회하지 않는다. 현재 동작 가능한 v2.5.4부터 연결하되 v3 shadow·미수용 엔진의 capability를 apply로 위장하지 않는다.

### Top-5 포즈 표시

보정한 source 골격/관절 packet 또는 source 기반 3D 데이터를 표시한다. 공통 표시 모델이 필요하면 한 번 로드해 재사용하는 **display-only 기준 모델**이다. 사용자 개인/최근/선택 체형이 아니며 user shape·target 손을 적용하거나 실제 체형별 retarget/메시 보정을 수행하지 않는다. pose 표시 때문에 후보마다 Converter 인체 FBX/PNG를 생성하지 않는다.

source와 표시 rig의 회전·관절 대응·좌표계·카메라 mapping을 명시적으로 검사한다. 정적 library 원본 썸네일을 refined로 표시하지 않는다. 표시용 모델을 target 메시 안전 근거로 쓰지 않는다. 스킨·외곽선 등 외양은 pose identity와 분리하고 별도 appearance 작업의 파일·PR을 덮어쓰지 않는다.

## 4. 선택 후 체형 적용·target 보정·최종 검증

선택한 prepared source의 hash·정책·관측·카메라·소유권·현재 source revision을 재확인해 재사용한다. source solver를 다시 호출하지 않는다. 그다음 기존 사용자 설정/기본/자동 규칙으로 체형·수치·손·범위를 정하고 **선택한 후보 하나에만** target을 적용한다. 다인은 인물별 선택 결과만 처리한다.

수용된 target-aware v3/v4 프로필·연산이 있으면 이때 호출하고 실제 retarget 결과·관절/표면·접촉/간섭을 검사한다. source 오차 개선을 실제 skin 안전 인증으로 바꾸지 않는다. 미지원/실패는 동일 target 조건의 수용된 기존 결과로 복구하며 불가능한 안전 보장을 추가하지 않는다. v4 사용자 지시형 parts의 명시 지시 흐름은 유지한다.

타인과 손잡기·포옹·기대기는 선택 상대와 배치가 정해진 뒤 영향 있는 연결 집합만 검사/필요 보정한다. 선택 전 모든 5^N 조합을 생성하지 않는다. 관측에서 얻은 자기 접촉·바닥 지지 근거는 재사용하되 최종 target/상대 조합 검사와 구분한다.

같은 source에서 체형·손·범위만 바꾸면 검색·관측·VLM·source solve를 유지하고 target 결과만 갱신한다. 바뀐 target preview revision의 검증 파일만 저장한다. 이미 검증한 bytes는 재사용하고 canonical 상태만 있으면 최종 export·독립 재임포트로 **선택 후 target 미리보기**와 correspondence를 확인한다. 기존 허용치를 완화하지 않는다. Top-5 source 포즈와 최종 target이 같은 표면이라고 보장하지 않는다.

.clip 저장·재열기/CSP 내부 재관절편집/카메라 FBX bake·CSP 초기 시점 자동 설정은 이번 수용 조건에 몰래 추가하지 않는다.

## 5. 후보 상태·Top-5 공개 장벽

| 상태 | 의미 | 처리 |
|---|---|---|
| queued / running | source 후보 준비 중 | 완료로 표시하거나 과거 결과로 대체하지 않음 |
| ready_refined | 현재 source 보정본이 정책·검사를 통과 | 정확한 source 포즈 표시·선택 후 재사용 |
| ready_base | source 비악화/안전 판정으로 원본 유지 | 원본 유지 결과 표시, 보정 성공으로 집계하지 않음 |
| unsupported | 해당 source/엔진/정책 미지원 | 이유·허용 대안 명시 |
| failed_retryable | 시간 초과·일시 장애 | 재시도/원본 유지 정책 명시, 영구 성공 cache 금지 |
| invalidated / cancelled | 관측/후보 변경·삭제·취소 | 과거 결과의 최신 묶음 재게시 금지 |

기본 UX는 모든 후보의 **확정된 source 처리 결과**를 준비한 뒤 Top-5를 공개한다. raw 카드부터 보여주고 나중에 몰래 자세를 바꾸는 대체는 하지 않는다. 간단한 진행 표시가 가능하다. 보정 불가 후보를 억지로 변형시키지 않고 한 후보 실패로 유효 후보 전부를 잃지 않게 한다. 제한시간 후 실제 원본 유지/재시도 정책을 분명히 표시하고 완료를 위장하지 않는다.

**source_only는 사전 단계의 정상 품질 범위다.** target 체형 미수용 때문에 정상 source 후보를 실패/unsupported로 바꾸거나 Top-5 전체를 막지 않는다. source 영수증은 최종 target 파일 권한이 아니다.

전체화면·플로팅 바는 같은 source context·결과·진행률을 사용한다. 캡처/이미지 검색부터 구현하고 자연어/일반 참고 생성 후보의 기존 정책을 유지한다. 원래 순위·ID를 보존하며 새 순위/후보 보충을 별도 평가 없이 추가하지 않는다.

## 6. source cache와 선택 후 target cache 분리

source context/cache는 설치·작업·인물 소유권, source candidate-set revision/후보 ID, 러프 관절·관측/귀속/정책 hash, 원본 BVH SHA, 검색 카메라·원근·좌우 mapping, source solver/selector/라이브러리/설정 버전을 묶는다. **body/shape/hand/profile/export scope/appearance는 제외한다.**

source 관측 coverage·허용 사지와 사용자의 export crop 범위는 다르다. 관측 변경은 source를 무효화하지만 출력 범위 변경은 target/출력만 무효화한다. source revision과 target selection/output revision도 분리해 체형 변경이 source 묶음을 간접 무효화하지 않게 한다.

동일 source 요청은 진행 중 작업까지 합친다(single-flight). 정상 no-gain/안전 원본 유지와 timeout/일시 장애를 구분하고 일시 장애를 영구 성공 cache에 넣지 않는다. 사적 query 결과는 설치/작업/보존기간에 귀속하며 다른 사용자에게 전역 cache로 노출하지 않는다. source 변조·격리·엔진 변경은 reuse에서 다시 검사한다.

선택 후 cache는 prepared source SHA에 selection revision, 실제 body/profile/shape/hand/범위, target solver/retarget/mesh-check 버전을 추가한다. 외양은 별도 presentation/output identity다. 기존 job/person/candidate PK만으로 target을 재사용하지 않는다. 새 저장 구조/migration은 구버전 경로를 보존한다.

클라의 source 후보별 cache와 선택 후 target 상태를 분리한다. 기존 refineByPerson projection은 호환 adapter로 유지할 수 있다. prepared source 선택 시 useRefineSelection이 같은 source를 다시 풀지 않게 하고, 구버전 서버/클라는 기존 선택 후 리파인으로 호환한다.

## 7. 시간 최적화·제한 병렬

선택 전 critical path에서는 PNG·각도별 썸네일·후보별 target FBX·실제 체형 적용을 제외한다. 기존 분석/VLM/관절·가림·접촉 근거는 후보 전체가 재사용한다. BVH 파싱/불변 FK·검색 카메라·공통 수치 준비를 재사용하되 입력/결과 buffer는 후보별로 격리한다. source single-flight/cache, 재진입·앱↔바 전환·선택 후 source 재계산 제거를 우선한다. 공통 display rig/geometry와 다운로드/GPU buffer를 공유하고 후보별 불필요한 WebGL context를 늘리지 않는다.

선택 후에는 선택한 source 하나만 target에 적용한다. 승인 model/rig/body baseline·손 라이브러리를 불변 cache로 재사용한다. warm worker는 정확한 복구·독립 검증을 통과할 때만 재사용한다. “미리 준비”라는 이유로 선택 전에 사용자 target을 후보 다섯 개에 입히지 않는다.

- Promise.all(5N)로 폭주시키지 않는다. 전체 worker 용량·설치/작업 간 공정성·큐 길이·인물 수를 제한한다. 1vCPU에서 C=1/2를 같은 조건으로 비교하고 기본값은 실측으로 정한다.
- I/O와 CPU, BLAS 내부 thread와 외부 후보 병렬도를 함께 조절한다. 요청마다 전역 환경변수/CFG·카메라를 변경하지 않고 요청별 projection context를 유지한다.
- 한 Blender scene/bpy를 여러 thread로 수정하지 않는다. target은 선택 후 격리 프로세스 또는 복구 검증된 단일 warm worker로 처리한다. v4 단일 슬롯은 선택 후 작업 큐로 보호하고 503 무한 fan-out 재시도를 하지 않는다.
- 현재 선택의 최종 준비/검증을 과거 무효 작업보다 우선한다. 관측/후보 변경·작업 삭제 시 불필요 source를 취소한다. 체형·손 변경이나 앱↔바 전환만으로 source를 취소/재계산하지 않는다.
- 큐/source/target/export 기한을 구분하고 실제 process/thread가 끝나기 전에 슬롯만 반환하지 않는다. 자기 process group만 정리한다.
- 내부 batch 상한은 bounded chunk 제한이다. 3명 이상에서 11번째 후보부터 누락하지 말고 같은 job/source revision의 제한된 chunk와 전체 barrier를 관리한다.

측정하지 않은 “5개 몇 초”를 SLA로 약속하지 않는다. 기존 source 후보당 5초/BFF9초를 batch SLA로 해석하지 않고 baseline으로 명시 budget을 정한다. 끝나지 않은 결과를 완료로 처리해 지연 수치를 줄이지 않는다.

선택 전: 입력→검색 완료, source batch→전체 terminal, source parse/solve/check/전송, cache/in-flight hit, 실제 첫 보정 preview, 최대 동시 수·CPU·memory·timeout/복구. 선택 후: 선택→target preview, 선택→검증 FBX 준비, target load/적용/보정/메시/serialize/reimport, 같은 source solve 재호출 0. 기존 raw Top-5와 새 refined Top-5의 작업량/품질 차이를 표시한다.

target/성능 담당이 정량 benchmark를 조율한다. C=1/2, cold/warm/cache hit은 같은 CPU quota/메모리·worker 수·부하에서 비교한다. 다른 세션의 무거운 빌드/Blender가 섞인 수치를 무부하와 비교하지 않는다. throughput을 첫 화면 지연 개선으로 바꾸어 보고하지 않는다. 표본 수·실제/합성·초기화 포함 여부를 명시하고 소수 percentile을 운영 p95 보장으로 해석하지 않는다.

## 8. 서버·BFF·클라 계약

엔진이 공통 source machine contract를 먼저 공개한다. 최소 개념은 source context/fingerprint, batch/chunk ID·source revision, 원본/보정 BVH SHA, 엔진/정책 버전, 후보별 상태/reason/timing, 실제 source pose/preview 데이터다. **선택 전 prepare는 target/body/shape/hand/appearance 입력을 받지 않는다.** 이전 초안의 optional target과 사전 target-provider 호출은 제거한다. target 권한을 source 영수증에 넣지 않는다.

BFF는 후보·관측·카메라를 서버에 등록한 뒤 source를 준비한다. 기존 loadCandidate가 completed result JSON에서 camera를 읽는 의존성을 고려해 내부 authoritative 입력을 구성하고 공개 전 필요한 정보를 원자 저장/명시 전달한다. confirmed selection·실제 노출/선택 계측은 미리 만들지 않는다. 캡처1건의 quota를 후보별로 중복 차감하지 않는다.

긴 UI 연결보다 기존 job/poll과 호환되는 prepare/status/result를 사용한다. 소유권/source revision을 검사해 늦은 응답이 최신 묶음을 덮어쓰지 못하게 하고 후보별 중복 polling 대신 묶음/delta를 쓴다. 실제 body/hand/target context는 **사용자 선택 후** 별도 endpoint/작업에서 구성한다.

클라는 source 3D만 Top-5에 표시하고 선택한 prepared source를 이후 target 경로로 넘긴다. 체형·손은 source 준비/표시의 전제 조건이 아니다. 최종 target 영수증/hash·현재 선택·소유권·lease/보존기간은 파일 전달 때 다시 검사한다. 제품 UI는 사용자 문구를 쓰고 엔진/hash/개발 용어를 노출하지 않는다.

## 9. 구현 순서·검증

1. 최신 develop/의존 PR·source/target 경계를 확인하고 고정 관측 fixture baseline·source 계약 확정.
2. source refine를 사전 후보 batch로 연결: cache·원본 유지·늦은 응답·source3D·선택 후 source reuse.
3. 선택한 source에 실제 body/손을 적용하고 필요한 target-aware v3/v4·메시/조합/FBX 검증 연결. 미수용 엔진/프로필/연산 제한.
4. 전/후 단계별 cold/warm·C=1/2·cache·선택 후 source 재계산0을 측정하고 UI 샘플·실제 파일/한계·새 PR/CI 보고.

필수 사례: 1인5·2인10·3명 이상chunk; 가림/저신뢰/부분 관측·비허용 후보; no-gain·timeout·crash·원본 변조/cache손상/엔진 변경; 설치·인물 분리; 앱/바 전환·재열기·늦은 완료·취소; 동일 source 선택 시 solve0회; **체형/손 변경에도 source key/결과/완료 상태 유지**; source 표시와 target 안전 분리; 선택 후 body/shape/손/범위/appearance의 정확한 target 무효화; 실제 조합 접촉·독립 재임포트; warm 복구·슬롯/process 정리.

기존 로컬 관측/평가 자료로 정면, 굽힘/앉기·접촉, 가림, 다인을 포함하고 source/target 성능을 따로 기록한다. 유료 VLM/클라우드 평가를 실행하지 않는다. 모델/토큰/원본 사용자 이미지/대용량 산출물을 Git에 넣지 않는다.

## 10. 병렬 세션·의존성·배포 금지

| 역할 | 소유 범위 |
|---|---|
| 서버 엔진 | api/source batch·engine adapters·공통 source contract/schema. 사전 target-provider 호출 금지 |
| 선택 후 체형 검증·성능 | converter/선택한 prepared source의 body/손 적용·필요 target 보정/메시/최종 FBX 검증, 전/후 성능 조율. 사전 target batch 금지 |
| BFF | source 등록/job/single-flight/권한/source cache·노출 계측; 선택 후 target 별도 조율·재사용/migration |
| 클라 | Top-5 source 상태/cache/포즈 표시·앱/바 장벽; 선택 후 체형 적용·최종 확인/저장, 중복 source refine 제거·샘플 |

각자 최신 origin/develop의 독립 codex 브랜치·워크트리를 사용하고 원본 dirty checkout을 reset/stash하지 않는다. frozen solver/retarget·자산·허용치는 수정하지 않는다. 외곽선/스킨/큰미리보기/개인모델 기존 PR은 변경/병합하지 않는다. 미병합 기능은 pinned 의존성/stacked PR로 구분한다.

직전 검증 heads: 서버#170 70b8a6f37d748d5a81801e0c1d632f7c512ba517, BFF#161 54f14a1edd33d915e01b08e22734fb730642d5cd, 클라#162 9962f497c26ef049272c7ff9462ee7c1021830c4. 원격 상태 재확인. #170 마지막 독립 커밋은 exact 승인 남녀 ID+SHA의 Head를 보존하는 importer adapter, #161은 Operation 전체를 ownerBinding으로 해시하던 문제 수정이다. 선택 후 경로에 필요하면 최소 의존성과 회귀를 명시적으로 가져오고 안전 우회로 재해결하지 않는다.

공유 폴더: /Users/dowon/dev/Standin-server/artifacts/pretop5-refine-20261106/. 엔진은 source CONTRACT.md/schema를 소유하고 target 담당은 **선택 후** 별도 계약을 제안한다. 각 status/measurement/dispatch에 이 정정을 반영한다. 엔진은 이 개정 설계를 새 PR에도 포함한다.

이번 승인은 구현·로컬 검증·새 PR 작성이다. 운영/스테이징 배포·전역 활성화·보호 브랜치 직접 push/병합·정책 우회·클라 릴리스·실제 AWS 증설·유료 AI 호출은 하지 않는다. 서버 병합 전 최소1명 리뷰 규칙도 유지한다.
