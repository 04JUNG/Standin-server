# develop → main 승격 점검 · 2026-11-12

기준: main `371bbed1632d64819babd19b3aceb12d1adf7f97`, develop `fd2212e123ddc07bb53e8abb10967d96cd7d0273`.

이전 승격이 squash로 쌓여 두 브랜치를 직접 merge하면 이미 운영 중인 코드까지 충돌한다. 마지막 main→develop 동기화 당시의 main `38fc3cc1a06ace6c65f6144bc70315950afa1aa7`를 비교 기준으로 삼아 develop 변경과 main 전용 수정사항을 검토하고, 현재 main에서 분기한 승격 브랜치에 결과를 올렸다. develop/main 원본 브랜치는 수정하지 않았다.

- develop의 v2.5.4 관측성 수정·새 포즈/손/FBX 보존·검색/진행 스트림·새 기능 코드를 승격한다.
- main의 production Recall 모델 버킷/빌드 옵션, 본체와 initializer 이미지 동기화, converter 수동 beta 배포를 보존한다.
- production의 설명 prefetch 기본 ON을 workflow에서 명시한다. staging의 선택적 설정은 유지한다.
- main 배포 직전 `prepare_production_refine.py`가 legacy v2 aggressive와 안전 selector를 고정하고 RC4 worker URL을 제거한다.

검증: 전체 inference 테스트의 3,569개가 sandbox에서 통과했다. loopback bind 제한으로 실패한 11개는 권한이 있는 로컬 가짜 HTTP 서버에서 재실행해 모두 통과했다. 11 skipped. 별도로 refine/smoke/deployment 126 passed, 1 skipped와 최종 production 정책/Recall/converter 배포 계약 28 passed. 실제 유료 모델 호출·Blender 실메시 검수는 새로 실행하지 않았다.

## refine와 develop 전용 기능 최종 점검

| 항목 | 이번 production 승격 정책 |
|---|---|
| legacy refine | `REFINE_ENABLED=1`, `REFINE_V2_ENABLED=1` |
| 알고리즘/요청 기본값 | `v2.5.4`, `REFINE_DEFAULT_MODE=aggressive` |
| aggressive 안전 selector | `REFINE_V25_SELECTOR_ENABLED=1` |
| 제한 몸통 | `REFINE_V2_TORSO=0` |
| New Refine RC4 | inference/BFF/worker `REFINE_RELEASE_ENABLED=0` |
| v3 실험 | inference `REFINE_V3_MODE=off`, evidence OFF; BFF shadow OFF |
| staging 전용 배포 | scene worker, anatomy initializer/preset pin, interactive transport 준비는 staging guard를 유지 |

`services/refine_release`의 전용 worker 이미지에는 실행용 ON 기본값이 있지만, production inference 이미지나 BFF가 그 worker로 바뀌지 않는다. 전용 worker 생성/배포도 이번 PR 범위가 아니다. 신형 refine 코드는 포함하되 production에서 광고하거나 호출하지 않는다.

검사 시점 실제 production은 inference task revision 53, BFF revision 59, analysis-worker revision 25였다. `REFINE_ENABLED=1`/BFF 노출 true였고 RC4·v3 override는 없었다. 당시 inference 이미지의 legacy 버전은 v2.5.3이므로 **이 PR 병합·배포 전부터 live가 v2.5.4라고 해석하지 않는다**. 이 문서와 PR 생성은 배포 실행이 아니다.

전체 저장소 점검: server/BFF/client/infra에 승격할 코드가 있다. 랜딩 `04JUNG/Standin`의 develop 트리 `295790a4b4670995233befcba7dd775f2e667f42`는 이미 과거 main `cb2a74bdd6d078a7200ff4d7d3819521426c8144`에 포함돼 새 승격 PR이 필요 없다. `Standin-master-docs`, `demo-repository`에는 develop이 없다.
