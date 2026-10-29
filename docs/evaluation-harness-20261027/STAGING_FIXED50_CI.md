# 스테이징 고정 50컷 CI

## 실행과 범위

`.github/workflows/deploy.yml`의 develop 배포가 안정화된 뒤 staging 환경 변수 `STAGING_EVAL_ENABLED=true`이면 자동 실행한다. 모든 develop 배포를 평가하므로 검색·리파인 변경 배포도 포함한다. main/beta에서는 실행하지 않는다. 기존 배포 concurrency를 공유하며 배포가 바뀌는 동안 평가가 겹치지 않는다. 평가 실패는 배포를 되돌리지 않으며 CI 실패와 보고서로 알린다.

GitHub runner는 모델을 실행하지 않는다. `scripts/run_staging_evaluation_ci.py`가 스테이징 서비스의 현재 task definition을 읽고 **동일 이미지·라이브러리·모델 설정**의 별도 Fargate task를 실행한다. inference 서비스 자체는 수정하지 않는다. task 안의 API는 127.0.0.1에만 열며 Discord 알림 설정은 제거한다. converter는 스테이징 내부 서비스를 사용한다. 이는 BFF 인증·사용자 클릭·최종 FBX 저장까지의 제품 E2E 검증이 아니다.

최대 task 실행 시간은 3,000초이고 CI step은 60분이다. 원본 50회 분석, 반환 인물별 Top-1의 허용된 리파인 최대 60회, 두 arm의 AI 판정(각 arm 최대 120회 예산)이 상한이다. comparison arm은 첫 arm에서 동결된 목표 자세를 재사용하고, 다음 버전에서도 이전 목표 추정을 재사용한다. 실제 비교 모델은 staging 환경 변수 `STAGING_EVAL_JUDGE_MODEL`로 고정한다. 별도 평가 태스크/렌더/AI 비용이 발생한다.

## 고정 데이터·기준선

사용자가 확정한 `standin-rescore-v3.2.zip`, SHA-256 `be85f93b2e3f4bc276d221ce8dd5c716b3360f546802c5dcf65a703d7bcf00b3`를 스테이징 비공개 assets bucket의 `evaluation/fixed50/v3.2/<sha>.zip`에 보관한다. 매 실행 실제 ZIP 바이트와 입력 이미지 해시를 검증한다. 기존 최종 출력과 점수를 새 버전 결과로 재사용하지 않는다. 원본 50컷, 기대 인원·원본 박스, framing만 사용하며 박스/인원은 기존 AI 기준이라는 한계를 유지한다.

평가 코드·루브릭·모델 이름·데이터 해시가 같은 실행끼리만 같은 비교 series다. 첫 완료 실행이 baseline이며 이후 마지막 완료 실행과 비교한다. 인프라·렌더·AI 판정 실패는 이전 baseline을 갱신하지 않는다. 성능이 나빠졌지만 정상 실행된 결과는 그대로 다음 비교 기준이 되며, 이전 모든 보고서를 버전별 보존한다. 평가 코드/모델 변경 시 새 series를 생성하고 잘못된 전후 차이를 제시하지 않는다.

프로덕션 버킷을 쓰지 않는다. 평가 task는 제한 시간의 서명 URL로 필요한 ZIP만 읽고 자기 보고서 경로에만 업로드한다. URL/키는 보고서에 넣지 않는다. GitHub CI 역할의 필요한 추가 권한은 `evaluation/ci/staging-evaluation-policy.json`에 명시되어 있다. 평가 task role 자체에는 새 권한을 부여하지 않는다. 현재 배포 역할은 RegisterTaskDefinition/Describe/PassRole 권한을 이미 가지고 있다.

## 보고서

GitHub Actions Summary에 점수·개선/악화·제외 변화 표가 나오며, `staging-fixed50-<sha>-<attempt>` artifact는 30일 보관한다. ZIP을 내려받아 `report/report.html`을 열면 search/refine 각각의 원본·최종 이미지·부위별 이유를 볼 수 있다. 원본/실행 산출물이 포함되므로 저장소 접근 권한이 있는 검수자용이다. S3에는 run별 report.zip과 마지막 완료 baseline.json을 보관한다.

- search: 동결된 Top-1 후보를 converter로 렌더링해 v3.2로 평가.
- refine: 해당 실행의 **같은 후보·같은 카메라**에 리파인을 적용해 평가. 허용되지 않은 후보는 베이스 유지.
- 전신100/상반신45 평균과 평가 가능 수, 동일 평가 가능 컷의 변화 범위, 확실한 개선/악화·동일·불확실 범위 중첩·제외 변화.
- 동일 실행 내 search → refine 전후 비교, 분석/렌더/리파인 오류, 시도 횟수와 총 소요 시간.

두 버전의 점수 범위가 겹치면 개선이나 악화를 확정하지 않는다. 평균의 분모가 달라지면 그대로 비교하지 않고 대응 가능한 컷 평균과 제외 변화를 함께 제시한다. Top-5 정답률, 인간 검수 완료 점수, 메시/FBX 안전성, 검색/리파인 변경의 인과적 기여로 해석하지 않는다. 후보가 버전 사이 달라지면 refine arm의 버전 비교는 전체 파이프라인 비교다. 실제 평가 AI의 정확도 검증과 사람 최종 검수는 별도다.

## 설정·실행

staging 환경에 `STAGING_EVAL_ENABLED=true`, `STAGING_EVAL_JUDGE_MODEL=gemini-3.5-flash`를 설정한다(모델은 현재 스테이징 설정에서 확인한 값). IAM 정책 승인·적용 후 활성화해야 한다. CI 코드가 develop에 반영된 다음 배포부터 자동 실행된다. 정책 변경이나 PR 머지는 자동 승인됐다고 가정하지 않는다.

직접 실행 시 현재 스테이징 SHA를 반드시 지정한다:

```bash
python scripts/run_staging_evaluation_ci.py \
  --revision DEPLOYED_STAGING_SHA --run-id UNIQUE_RUN_ID \
  --judge-model gemini-3.5-flash --output out/staging-evaluation
```

`--preflight-only`는 모델/태스크를 실행하지 않고 배포 SHA·데이터 존재를 검증한다. `STAGING_EVAL_ENABLED=false`로 자동 평가만 중지할 수 있다. task의 healthCheck와 외부 알림만 평가 실행용으로 제거하고, 종료 시 task를 중지하고 파생 task definition을 비활성화한다.

## 검증

비교 범위·분모·상반신 분리, 다인 박스 대응, dynamic 무결과의 legacy 대체 차단, 리파인 권한·원본 후보 전달, 알림 제거와 task 정의 불변성, develop 전용 workflow, 서비스/AI를 주입한 실제 50컷 수집→양 arm 채점→보고서 생성 통합 테스트를 포함한다. 실제 스테이징 태스크 실행과 CI 권한 활성화 여부는 배포 기록에서 별도로 확인한다.

2026-10-29 로컬 검증: 전체 **2778 passed, 8 skipped, 31 subtests passed**, 변경 관련 **117 passed**, compile 및 diff 검증 통과. 스테이징 리비전 `612be944d387c3dd2543ee2eed6ae95183c101bb`과 업로드된 데이터의 preflight 통과. IAM 추가 정책은 별도 승인 대상이며, 승인 전 `STAGING_EVAL_ENABLED=false`로 설정한다. 실제 Fargate/평가 AI 50컷 실행 완료를 의미하지 않는다.
