# main → develop 공통 코드 동기화 (2026-11-13)

기준은 명시적 refspec fetch로 확인한 main `f31558c305c8bb4c1687ab8fe4e15f001ca26f24`, develop `e9de78d6c7bd99ed3e918a98046e0783d3ae144d`이다. squash 이력의 커밋 수 대신 두 tree의 파일/patch를 비교했다. 전용 경로 `/private/tmp/standin-sync-20261113/Standin-server`, 브랜치 `codex/sync-main-develop-20261113`.

| 판정 | 기능/파일 묶음 | 근거 |
|---|---|---|
| 반영 | #231 binary v2, v1 rollback/협상, binary body base, 공유 reader/fixture | converter/interactive*, solved_state*, converter_api/interactive, Dockerfile에 필요한 파일 포함 |
| 반영 | #229 metadata/health digest 캐시, #228 health stage timing, #225–227 준비/검증/appearance 추적과 published preview 재사용 | body_metadata/verified_files/health_assets, snapshot 검증과 요청별 binding 유지 |
| 반영 | #234 body/solve cache capacity 복구·오류 구분 | work_queue/work_middleware, interactive API와 회귀 |
| 반영 | #222 ob_083 검수 허리·발 | source_body/shape_bake/reviewed_pose_frames.json의 main 검수값 그대로, 원본 자산·임계값 임의 변경 없음 |
| 반영 | #220 Recall 손 screening 의존 제거·기본 후보 오류 복구 | search_recall engine/verify_hypothesis_rank와 analysis_stream/회귀 |
| 반영 | #233 실제 Blender v1/v2 CI | converter-ci에 prepare → solve → 독립 export/reload 검사 추가, 배포 workflow 미변경 |
| 이미 포함 | #203/#211/#212 기본 FBX 재사용·prepared neutral, #221 중복 Blender 제거, #223 검증 skin identity 재사용, composed skin | 동일 공통 파일은 develop에 이미 존재. tree 차이가 없는 코드는 중복 cherry-pick 안 함 |
| 환경별 유지 | .env.example, inference/Converter deploy workflows, production parity/pin/refine/initializer/Recall 배포 helper 및 관련 배포 tests | staging New Refine OFF/기존 aggressive 정책·기존 pin·배포 권한 그대로 |
| 별도 소유/제외 | add_prepared_bodies_to_release.py 및 release helper/tests/docs, #235/#236 프리셋20종, 메모리 장면 전달 실험 | 준비·패키징 세션 및 실험 세션 소유. 열린 PR/실험 코드/자산 미반영 |
| 생략 | src/refine_v4/runner.py 끝 빈 줄, 과거 운영 승격 문서 | 동작 차이 없음 또는 운영 기록 |

## 호환·번들 의존성

Converter → BFF → client 순서로 검토/병합한다. Converter는 v1/v2 모두 지원하며 BFF는 client token을 받은 경우에만 binary를 협상하고 구 Converter의 v1 응답도 허용한다. 기존 최종 FBX 검증/receipt 권한은 유지한다.

`prepared_body.identity`는 shape_bake.py 등 구현 hash와 Blender build/OS/architecture를 key에 넣는다. 이번 shape_bake.py 반영으로 이전 develop prepared-neutral key가 바뀐다. 기존 번들은 정확한 key가 불일치하므로 안전 생성 경로로 폴백하며 새 head 기준 재생성 전에는 prepared 채택을 기대하면 안 된다. #235 준비 세션은 동기화 head로 재배치하고 해당 runtime별 번들을 재생성/재검증해야 한다. interactive runtime revision도 갱신되어 이전 private body/solved cache를 재사용하지 않는다. 자산 bundle/pin·카탈로그는 수정하지 않았다.

## 검증 증거

main의 #233에 포함된 과거 검증과 이번 실행은 구분한다. 이번 워크트리에서 Blender 5.2.0/build fbe6228777e7(macOS arm64) 합성 fixture를 새로 생성했고 verify_interactive_binary.py의 v1/v2 body 준비·solve·fresh-process export/reload를 실행했다. 논리 상태, topology/joint/skin 및 receipt model digest 동등성 통과. 출력은 별도 tmp 아래만 저장하며 실제 사용자 자산은 사용하지 않았다. Linux/amd64 검증은 PR CI 대상이다. 그 외 회귀 결과는 아래 최종 검증 기록에 추가한다.

## 인프라 검사

인프라 main `1345a833522c425d49af69674e756a6596f8d998` / develop `33d4fdb409552b26eb02ea7181db81052edcd5b7`의 전체 tree diff를 별도 워크트리에서 확인했다. #89/#90/#91과 production parity, prod inference/BFF capacity, ALB timeout, image/library/skin/prepared pin, production-only initializer/asset promote helper가 전부 운영 전용 변경이다. body-assets 모듈 차이도 운영 appearance 허용 확장이다. staging에 필요한 공통 runtime 수정이 없어 인프라 코드는 변경하지 않고 PR도 만들지 않는다. #92/#93 및 replay #75는 제외했다. 인프라 변경이 없으므로 synth/deploy를 실행하지 않았다.

어느 저장소도 main/develop 직접 변경·병합·배포·릴리스·AWS/SSM/ECS/자산 업로드를 하지 않는다.
