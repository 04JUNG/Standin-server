# 생성 포즈 02~10 통합 검토 및 실제 FBX 검증

2026-10-20 KST. 사용자 최신 지시에 따라 01(크게 보기/확대)은 제외한다.
사진 클릭과 `이 포즈 선택`으로 선택하는 기존 동작을 유지한다.
기능 브랜치 `codex/generated-recovery-qa`만 사용하며 기본 checkout은 수정하지 않는다.
자동 적용/refine, 인물 통합/v4, 검색/생성 알고리즘과 검사 기준 변경은 없다.

## 범위 대조

| 항목 | 구현/검증 근거 |
| --- | --- |
| 01 | 사용자 취소. 새 확대 버튼/모달 제외, 사진 선택 유지 |
| 02~03 | BFF 생성 source/null poseId 타입·nullable DB·소유 작업의 비공개 원본·해시·기존 검사 근거 |
| 04~05 | 생성-only/혼합 후보의 수동 권한·메인/바 인물 집계·선택 확정·선택 보존 |
| 06 | 기존 Converter에 체형/수치·양손·범위를 같은 입력으로 전달 |
| 07~08 | 최종 확인 paired FBX의 동일 바이트/해시 저장·설정 변경/오래된 응답 거절·만료 시 재확인 |
| 09 | 원본 소실·정보/해시 불일치·검사 탈락·설정 변경·변환 실패별 한국어와 복구 |
| 10 | 앱/바 fixture 회귀 + 아래 실제 생성·현재 체형·손·범위 변환/재열기. 설치 앱/CSP는 미검증 |

BFF/Client 구체 계약 및 테스트는 각 저장소 `docs/GENERATED_MANUAL_EXPORT.md`와
`docs/qa/GENERATED_RECOVERY_QA_2026-10-20.md`(Client)를 참조한다.

## 실제 생성과 현재 등록 체형 검증

- Blender 5.2.0 LTS, build fbe6228777e7. 로컬 실제 리그 `2hand Idle_01`의
  계층/OFFSET으로 기존 TextConstraintPoseGenerator를 실행했다. 이전 동작 프레임은
  재사용하지 않는다. 지시는 양팔 V자로 서기, 결과 preview_ready 1개다.
- 운영에서 받은 작업이 아닌 로컬 실제 생성기 실행 결과다. 생성 BVH·러프·인증값을
  저장소/문서/로그에 포함하지 않았다. 결과 자산은 비공개 임시 폴더에만 보관한다.
- 현재 로컬 등록 소스 male 0.7.2 / female 0.3.4의 registry/FBX/.blend/controller 해시를
  검증하고, 실제 `BlenderRunner.convert` → 기존 worker의 shape/hand/framing 경로를 사용했다.
  원격 배포 설정이나 다른 체형 정리 브랜치는 수정/통합하지 않았다.

| 실제 실행 | 적용 | 결과 |
| --- | --- | --- |
| male-base-v0-7-2 | height=182, muscle=.2, 왼 주먹/오른 힘 뺀 손, full | 변환/체형 정보/FBX 해시 일치, 2,030,300바이트 |
| female-base-v0-3-4 | height=172, physique=.1, 왼 편 손/오른 주먹, half | 변환/체형 정보/FBX 해시 일치, 1,065,356바이트 |
| male-base-v0-7-2 재현 스크립트 | height=182, muscle=.2, 왼 힘 뺀 손/오른 편 손, full | 2,030,444바이트, 몸 본 보존·손 자체 겹침 없음·체형 수치/해시 일치 |

- 실제 산출 FBX를 기존 shape importer로 다시 열었다. 남녀 각각 52본, 1메시,
  24,988 / 13,678정점, 유한 좌표, 재열기 전후 바이트 해시 동일이다.
- worker가 실제 최종 FBX를 다시 읽어 만든 PNG를 열어 양팔 포즈·현재 체형·양손 차이와
  full/half 범위를 육안 확인했다. 전체 메시가 터지거나 사라지는 현상은 보이지 않았다.
- 남녀 산출물의 기존 반대손 피부 교차 검사는 clear_screen(교차 0)이다.
  이 검사는 양손 사이만 검사하며 전체 충돌 인증이 아니다.
- 주먹을 적용한 경우 기존 캡슐 기반 손 자체 겹침 경고는 남았다. 남성의 동일 체형/수치/손을
  원래 라이브러리 포즈에도 적용했을 때 동일하게 왼손 6건/오른손 0건이었다.
  생성 경로에서만 발생한 회귀로 보이지 않지만, 주먹을 포함한 모든 손을 문제없음으로 인증하지 않는다.
  손 알고리즘이나 검사 조건을 바꾸거나 실패를 다른 손/체형으로 숨기지 않았다.

## 앞선 로컬 진단과의 관계

초기 plain convert 호출에서는 오래된 0.6.7/0.2.2 BASE가 필수 head 누락으로 거절됐고,
별도 옛 master 실메시로 full/half만 검증했다. 이는 현재 체형의 정상 shape worker 경로를
검증한 결과가 아니었다. 이번 검토에서는 기존 worker의 real terminal bone 보존 정책과
해시가 일치하는 현재 등록 소스를 사용해 위 체형·수치·손 검증을 추가 완료했다.

## 재현과 남은 확인

`scripts/verify_generated_fbx_roundtrip.py`를 Blender 내부에서 실행한다.
기본 인자 `--bvh PRIVATE --character FBX --out PRIVATE --scope full|half|bust|shoulder_head|head`.
현재 체형은 `--shape-registry LOCAL_REGISTRY --character-id ID --control height=182` 등으로
검증하며 `--left-hand`/`--right-hand`를 명시한다. 기존 shape 원본 해시와 제어값 검사를 재사용하고,
실패 시 다른 자산으로 대체하지 않는다. 결과 JSON은 모델/손/체형/재열기 요약을 분리한다.

- 이 실험은 BFF→설치 앱 native 저장까지의 end-to-end 실행이 아니다. 그 구간의 동일 파일
  전달·설정 변경·소유권/해시 거절은 BFF/Client fixture 테스트로 별도 검사했다.
- Standin Staging.app 0.1.1-dev.268 존재/버전만 확인. 설치본을 업데이트/교체하지 않았으며
  이번 변경 포함 설치 앱의 native 저장·드래그·CSP 불러오기, Windows 실앱 확인은 남아 있다.
- 기본 서비스 생성 리그 webtoon_combat_03 로컬 파일은 없었다. 다른 운영 작업을 읽거나
  재생성하지 않았다. 원격 생성/배포별 자산 설정은 이 로컬 결과가 대신 인증하지 않는다.
- 통합/v4는 기능 범위 밖이며 생성 인물이 포함된 컷은 개별 저장만 지원한다.

## 자동 검증

- BFF 전체 654 passed, 2 skipped. Client 전체 966 passed, 1 skipped.
- Client build/typecheck/lint 성공(기존 Vite 대형 번들 warning).
- 최신 server develop 750fbe567 기준 생성/표시/shape-hand/shape-registry/framing/runner 오류
  관련 pytest 185 passed. smoke 54/54, compileall도 별도 확인한다.
- 위 자동 테스트의 Converter fixture 결과와 실제 Blender 결과를 섞어 보고하지 않는다.
