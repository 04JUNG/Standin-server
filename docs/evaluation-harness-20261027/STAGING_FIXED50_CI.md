# 스테이징 고정 50컷 CI

## 실행과 범위

**2026-10-31 사용자 요청으로 유료 자동 평가를 계속 꺼둔다.** 저장소·staging·운영(beta)의 `STAGING_EVAL_ENABLED=false`를 적용했다. 배포 workflow의 평가·평가 자격증명 갱신·평가 Summary·평가 artifact 단계를 제거한다. develop 배포나 향후 main 머지로 모델 평가가 자동 실행되지 않는다. 환경 변수를 실수로 true로 바꿔도 workflow에는 평가 호출 자체가 없다.

`scripts/run_staging_evaluation_ci.py`도 `PAID_EVALUATION_ENABLED=False`로 고정하며, 환경 변수로 풀 수 없다. `--preflight-only` 외의 실행은 AWS 접근·태스크 실행 전에 거부한다. 재활성화는 사용자의 새로운 명시적 요청과 코드 변경이 필요하다. 아래 실행 구조·보고서·2026-10-29~30 기록은 중단 전 이력이다.

GitHub runner는 모델을 실행하지 않는다. `scripts/run_staging_evaluation_ci.py`가 스테이징 서비스의 현재 task definition을 읽고 **동일 이미지·라이브러리·모델 설정**의 고정 50컷을 17·17·16컷으로 나눈 최대 3개의 별도 Fargate task를 실행한다. inference 서비스 자체는 수정하지 않는다. task 안의 API는 127.0.0.1에만 열며 Discord 알림 설정은 제거한다. converter는 스테이징 내부 서비스를 사용한다. 이는 BFF 인증·사용자 클릭·최종 FBX 저장까지의 제품 E2E 검증이 아니다.

최대 task 실행 시간은 3,000초이고 CI step은 60분이다. 3개 묶음은 같은 배포 이미지로 병렬 실행하며 50개 고유 컷이 모두 정상 완료된 경우에만 합산 보고서와 기준선을 발행한다. 어느 묶음이 실패해도 기존 기준선은 유지하고 묶음별 진단 파일을 보존한다. 원본 50회 분석, 반환 인물별 Top-1의 허용된 리파인 최대 60회, 두 arm의 AI 판정(각 shard/arm 최대 120회, 전체 최대720회 예산)이 상한이다. comparison arm은 첫 arm에서 동결된 목표 자세를 재사용하고, 다음 버전에서도 이전 목표 추정을 재사용한다. 실제 비교 모델은 staging 환경 변수 `STAGING_EVAL_JUDGE_MODEL`로 고정한다. 최대 3개 평가 태스크/렌더/AI 비용이 발생한다. converter를 공유하는 고정 3-way 부하 조건이므로 시간은 이 CI 조건의 관측값이다.

## 고정 데이터·기준선

사용자가 확정한 `standin-rescore-v3.2.zip`, SHA-256 `be85f93b2e3f4bc276d221ce8dd5c716b3360f546802c5dcf65a703d7bcf00b3`를 스테이징 비공개 assets bucket의 `evaluation/fixed50/v3.2/<sha>.zip`에 보관한다. 매 실행 실제 ZIP 바이트와 입력 이미지 해시를 검증한다. 기존 최종 출력과 점수를 새 버전 결과로 재사용하지 않는다. 원본 50컷, 기대 인원·원본 박스, framing만 사용하며 박스/인원은 기존 AI 기준이라는 한계를 유지한다.

평가 전용 코드·루브릭·모델 이름·데이터 해시가 같은 실행끼리만 같은 비교 series다. 패키지에 함께 들어간 제품 검색·리파인 알고리즘 파일의 변경은 series를 바꾸지 않는다. 첫 완료 실행이 baseline이며 이후 마지막 완료 실행과 비교한다. 인프라·렌더·AI 판정 실패는 이전 baseline을 갱신하지 않는다. 성능이 나빠졌지만 정상 실행된 결과는 그대로 다음 비교 기준이 되며, 이전 모든 보고서를 버전별 보존한다. 평가 코드/모델 변경 시 새 series를 생성하고 잘못된 전후 차이를 제시하지 않는다.

프로덕션 버킷을 쓰지 않는다. 평가 task는 제한 시간의 서명 URL로 필요한 ZIP만 읽고 자기 보고서 경로에만 업로드한다. URL/키는 보고서에 넣지 않는다. GitHub CI 역할의 필요한 추가 권한은 `evaluation/ci/staging-evaluation-policy.json`에 명시되어 있다. 평가 task role 자체에는 새 권한을 부여하지 않는다. 현재 배포 역할은 RegisterTaskDefinition/Describe/PassRole 권한을 이미 가지고 있다.

## 보고서

GitHub Actions Summary에 점수·개선/악화·제외 변화 표가 나오며, `staging-fixed50-<sha>-<attempt>` artifact는 30일 보관한다. ZIP을 내려받아 `report/report.html`을 열면 search/refine 각각의 원본·최종 이미지·부위별 이유를 볼 수 있다. 원본/실행 산출물이 포함되므로 저장소 접근 권한이 있는 검수자용이다. S3에는 run별 report.zip과 마지막 완료 baseline.json을 보관한다.

- search: 동결된 Top-1 후보를 converter로 렌더링해 v3.2로 평가.
- refine: 해당 실행의 **같은 후보·같은 카메라 회전**에 리파인을 적용해 평가. 허용되지 않은 후보는 베이스 유지. converter의 평가 렌더는 직교 투영이며 검색의 원근 거리까지 재현하는 UI E2E가 아니다.
- 전신100/상반신45 평균과 평가 가능 수, 동일 평가 가능 컷의 변화 범위, 확실한 개선/악화·동일·불확실 범위 중첩·제외 변화.
- 동일 실행 내 search → refine 전후 비교, 분석/렌더/리파인 오류, 시도 횟수와 총 소요 시간.

두 버전의 점수 범위가 겹치면 개선이나 악화를 확정하지 않는다. 평균의 분모가 달라지면 그대로 비교하지 않고 대응 가능한 컷 평균과 제외 변화를 함께 제시한다. Top-5 정답률, 인간 검수 완료 점수, 메시/FBX 안전성, 검색/리파인 변경의 인과적 기여로 해석하지 않는다. 후보가 버전 사이 달라지면 refine arm의 버전 비교는 전체 파이프라인 비교다. 실제 평가 AI의 정확도 검증과 사람 최종 검수는 별도다.

## 현재 허용된 실행: 읽기 전용 사전 확인

유료 50컷 실행은 차단되어 있다. 이전 실행 명령을 그대로 사용하면 오류로 종료하며 환경 플래그로 우회할 수 없다. AWS의 현재 배포 SHA와 고정 ZIP 메타데이터만 확인하는 명령은 다음과 같다. 모델 API와 평가 태스크를 실행하지 않는다.

```bash
python scripts/run_staging_evaluation_ci.py \
  --revision DEPLOYED_STAGING_SHA --run-id preflight-only \
  --judge-model gemini-3.5-flash --preflight-only
```

배포 workflow에는 평가를 켜는 옵션을 두지 않는다. 기존 파일 검증·집계와 저장된 보고서 열람은 사용할 수 있다.

## 검증

비교 범위·분모·상반신 분리, 다인 박스 대응, dynamic 무결과의 legacy 대체 차단, 리파인 권한·원본 후보 전달, 알림 제거와 task 정의 불변성, develop 전용 workflow, 서비스/AI를 주입한 실제 50컷 수집→양 arm 채점→보고서 생성 통합 테스트를 포함한다. 실제 스테이징 태스크 실행과 CI 권한 활성화 여부는 배포 기록에서 별도로 확인한다.

2026-10-29 전체 검증: **2787 passed, 8 skipped, 31 subtests passed**. 스테이징 전용 회귀 24개에는 3개 묶음의 50컷 집계 코드 통합, 누락·중복 거부, 제품 검색 코드 변경 시 기준선 유지, 평가 기준 변경 시 새 series, 스테이징 초기화 이미지 고정이 포함된다. compile 및 diff 검사와 PR checks/worker를 통과했다. 처리 시간 `total_seconds`는 50컷 작업 시간의 합이며 병렬 실행의 벽시계 경과 시간과 다르다.

사용자 명시 승인 후 `standin-github-deploy`의 `standin-staging-fixed50-evaluation` 정책을 적용했고, staging `STAGING_EVAL_ENABLED=true`를 확인했다. IAM 시뮬레이션에서 해당 평가 task definition의 RunTask 권한이 allowed다. 고정 ZIP과 스테이징 리비전 `612be944d387c3dd2543ee2eed6ae95183c101bb`의 preflight를 통과했다. CI 코드의 develop 반영은 PR 머지 후 이뤄진다.

실제 기동 검증에서 초기화 컨테이너의 과거 digest가 ECR에 없어 새 태스크를 만들 수 없는 문제를 확인했다. `prepare_staging_asset_initializer.py`는 인라인 Python 코드와 manifest 환경을 유지하고 이미지 참조만 이번 immutable inference SHA로 맞춘다. develop 배포 단계와 평가 태스크에 적용하며 main 배포에는 적용하지 않는다. 평가 태스크는 컨테이너에 들어 있던 이전 하네스 대신 업로드한 평가 코드를 명시적으로 먼저 로드한다. 현재 서비스 task definition 자체는 수정하지 않았다.

## 첫 실제 실행 결과 (2026-10-29)

`bootstrap-20261029-shards-v1`은 스테이징 리비전 `612be944d387c3dd2543ee2eed6ae95183c101bb`으로 50컷 분석을 마쳤다. 분석 오류 0건, 산출물 오류 4건(CB019/CB020/CB042/CB058), 리파인 오류 1건(CB027), 두 arm의 AI 판정 실패 23건을 기록했다. 모든 태스크 종료 및 파생 정의 정리를 수행했고, 기준선은 발행하지 않았다. 이 실행의 점수를 개선 수치로 사용하면 안 된다.

실측 후 AI 호출에 고정 JSON 스키마를 전달하고, 리파인 바인딩이 없는 숫자형 검색 카메라의 회전도 평가 렌더에 적용하도록 보완했다. 실패 실행의 컷별 오류 표도 Actions Summary에 남긴다. 이 변경은 회귀 테스트 대상이며, 수정 후 외부 모델 50컷 재실측을 통과한 것으로 간주하지 않는다. 첫 정상 기준선 확보는 아직 남아 있다.

## 2026-10-30 develop 통합 재검토

기준 develop은 `292b9151148e82d75cba6a61cfe11a8f114082a7`, 기존 PR head는
`2c1f150f2da930ea0b57892a77bdfb2877b9685a`다. 유일한 병합 충돌은 deploy workflow의
초기화 단계다. develop의 `refresh_staging_body_initializer.py`(staging family·고정 이미지·
`body_assets_ready` 계약 검사)를 먼저 실행하고 기존 평가용 inline Python 계약 검사를 이어 실행한다.
두 단계 모두 이미지 참조만 바꾸며 manifest·command·서비스 설정은 유지한다. main에는 적용하지 않는다.
검색/refine v4·관측 API·Converter 및 배포 설정은 develop 병합 결과를 유지한다.

### 실패 원인과 확인 수준

`60a4fb54b`의 JSON schema 고정, 숫자 카메라 회전 보존, 실패 Summary 보완은 이미 구현되어
회귀 검사로 확인했다. 이는 외부 모델/Converter의 모든 실패가 해소됐다는 증거가 아니다.
로컬에 남은 첫 실행 shard ZIP을 외부 호출 없이 재사용했다. 이 shard는 complete=false/failures=6이며
CB019·CB058의 ThumbnailRenderFailed, CB042의 HTTPError, CB036·CB073(search)·CB063(refine)의
judge ValueError를 포함한다. 현재 실패 보고서로 여섯 사례가 다시 표시되고 부분 shard의 합산/기준선
발행은 거부됐다. 실제 CB019의 숫자 카메라와 CB058의 카메라 없는 Top-1 선택도 회귀 사례로 추가했다.
CB058에서 다른 refinable 후보로 대체하지 않는 것이 의도한 계약이다.

전체 첫 실행의 4개 산출물/1개 refine/23개 judge 실패는 과거 실행 관측이다. 저장된 이 shard만으로
CB020·CB027의 현재 해결 여부를 확정할 수 없고, 오류 클래스만 남은 렌더/HTTP 실패의 상세 원인도
확정할 수 없다. 숫자 회전 수정은 렌더 오류 전체의 해결 증거가 아니며 JSON schema 수정도
23개 실판정 성공의 증거가 아니다. 현재 정상 baseline은 아직 확보하지 않았다.

### 평가 계약과 남은 승인

- 고정 50컷의 완료 여부와 품질 평균의 분모는 별개다. 누락·실패·미실행은 고정 집합에서 사라지지
  않으며, 활성 unobservable/not_run·인물 누락은 점수 평균에서 제외하고 제외 수를 보고한다.
  uncertain은 점수 범위다. 전신100/상반신45는 따로 집계한다. 관측 joint mask와 가림 추정은 분리한다.
- 실시간 CI는 동결 Top-1과 같은 후보의 refine 렌더 비교다. 최종 FBX 저장/receipt 검증과 동일하지 않다.
  최종 FBX는 별도 artifact lineage/export 하네스에서 후보·인물·선택 revision·출력 SHA·저장 SHA·receipt를
  대조한다. 증거가 없으면 unknown이며 렌더 점수로 FBX 안전성이나 저장 성공을 대체하지 않는다.
- AI target/판정은 `ai_assisted_unreviewed`다. 사람 승인·수정 이력과 별도 보관한다. 초안 GT와 AI 점수는
  검증된 정답 정확도가 아니다. 사람 GT 검수, AI 판정 대조, 별도 holdout/추가 러프 준비는 후속 작업이다.
- dataset ZIP hash·평가 코드 fingerprint·루브릭 hash·judge model이 series를 정한다. 제품 검색/refine 변경은
  같은 series에서 비교한다. 첫 정상 실행 및 이후 마지막 정상 실행만 baseline이 된다. 실패/부분 실행은
  진단 artifact만 남기고 조건부 baseline 쓰기에 도달하지 않는다.

### 자동 실행·알림·비용 경계 (2026-10-31 갱신)

기능 브랜치 push, PR 검사, develop·main 배포에는 유료 평가 호출이 없다. 저장소와 staging/beta의 평가 플래그는 false이며 CLI도 고정 차단한다. 과거 true 설정·IAM 부여·자동 실행 설명은 중단 전 이력으로만 읽는다. 기존 S3 데이터와 평가 코드는 기록·오프라인 검수용으로 보존한다. 일반 서비스 분석 요청의 모델 사용과 별개로 고정50컷 일괄 평가를 중지한 것이다.

평가 태스크의 Discord 환경 설정을 제거해도 ECS 실패 종료에 대한 인프라 알림은 별도 경로로 올 수 있었다. 실행이 실패하더라도 이미 수행된 모델 API 사용은 비용을 소비할 수 있으므로, 사용자의 새로운 요청 없이 유료 검증을 재개하지 않는다.

### 이번 로컬 검증

- 전체 `tests/`(converter 제외): **2901 passed, 10 skipped, 31 subtests passed**.
- staging/초기화 집중 회귀: **34 passed**. 스모크 **54/54**.
- compileall src/api/scripts/standin_eval/pose_curation 및 diff whitespace 검사 통과.
- 로컬 모의 HTTP 소켓이 필요한 10개 검사는 샌드박스 제한을 해제한 전체 재실행에서 통과했다.
- 저장된 shard ZIP 실패 보고서 재생: 산출물3/판정3 표시 및 부분 집합 baseline 차단, 외부 모델 호출0회.
- Converter Blender/FBX 실측·Docker 이미지 빌드·새 staging 실행은 로컬에서 수행하지 않았다.
