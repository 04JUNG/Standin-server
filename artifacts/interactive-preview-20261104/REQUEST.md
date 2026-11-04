# 앱 내 3D + 비동기 FBX 구현 공동 요청

사용자 요청: 설계대로 worktree를 나눠 별도 세션에서 구현한다. 운영/main/develop 병합·배포는 이번 요청에 포함하지 않는다. 기존 dirty checkout과 다른 세션 변경을 보존하고 각 저장소 최신 원격 develop에서 분기한다.

단일 설계: [설계 문서](../../docs/NEXT_SPRINT/INTERACTIVE_POSE_PREVIEW_ASYNC_FBX_DESIGN_2026-11-04.md)

작업 분담:
- Converter/Inference: interactive-pose-model-v1, SolvedPoseState, 공유 손 규칙, 실제 solve/export 경계, 동일 상태의 FBX 검증. converter-contract-v1.json 소유.
- BFF: 설치 소유 비동기 operation/HTTP/event/revision/receipt, 최신 선택 검사, 기록 이미지 보관. bff-http-contract-v1.json 소유.
- Client: Top-5 공통 3D/카메라/조작, 손 임시 반영/확정 패킷, 비동기 다운로드 상태, 작업기록 캡처·자연어 목록 정적 유지, 사용자 UI 검수 샘플.

계약 파일은 이 디렉터리에 공유하고 각자 자기 파일만 쓴다. 각 계약 파일에 draft/frozen, schema version, commit/source 및 변경 기록을 명시한다. BFF는 Converter 상태 계약을 참조하고 Client는 두 계약을 참조한다. 계약을 기다리는 동안 현황 감사·측정·뷰어·단위 검증은 진행한다. 원격 PR에 동일 스키마를 반영하고 테스트로 교차 검증하기 전 fixture만으로 통합 완료를 주장하지 않는다.

공통 불변식:
1. Top-5 때문에 서버 변환/PNG bake를 새로 하지 않는다. 최초 시점은 검색된 projection/rotation/distance/좌표계이며 orbit은 관찰용이다. 자연어 목록과 작업기록은 정적 이미지 유지, 자연어 상세 이후는 공통3D.
2. 현재 GLB는 정적 표면이다. 편집용 skin/bone 계약은 신규 버전으로 추가하며 기존 검증기를 완화하지 않는다. 손 즉시 반영은 지원 리그/규칙이 검증된 경우만.
3. 리파인 계산과 필요한 안전 검사를 생략하지 않는다. 완료 상태를 파일 출력보다 먼저 보여 주는 것이 목표다. 현행 worker의 초기/중간 FBX 의존 비용이 남으면 명시한다. frozen solver와 SHA를 조용히 바꾸지 않는다.
4. preview/export는 같은 immutable stateHash/revision/source/body/shape/hand/camera/scope를 사용한다. 늦은 응답·연속 손 변경·refine 경합·삭제·앱↔바 이동 후 옛 파일을 저장하지 않는다. 검증한 최신 FBX만 다운로드한다.
5. 로컬 예측·서버 확정·파일 검증을 구분하고, 파일이 준비되면 GPU/PNG 실패가 다운로드를 막지 않는다. 배경 파일 준비를 사용자 다운로드/디스크 저장 완료 지표로 세지 않는다.
6. 원본/개인 이름/경로/토큰을 replay·telemetry에 새로 노출하지 않는다. 사용자 앱의 수집 설정·동의 상태를 바꾸지 않는다.
7. 로컬 승인 자산/fixture를 활용한다. 새 유료 VLM/API 평가·운영 접속/배포·AWS 변경·PR 병합 금지. UI 최종 검수는 사용자에게 실제 화면을 보여 준 뒤 받는다.

각 담당은 계약 초안 → 구현 → 필요한 검증 → 제품/fixture/실제 파일 증거 구분 → 범위에 맞는 PR과 후속 항목을 남긴다. 공개/재배포 검토 전 원본 모델·BVH/FBX·생성 자산·runtime/secret을 Git에 포함하지 않는다.

각 담당의 진행 기록은 client-status.md / bff-status.md / converter-status.md에만 작성한다. 다른 담당의 파일이나 원본 작업을 덮어쓰지 않는다. 사용자에게 보고할 때 전체 설계의 완료 여부와 본인 단계의 완료 여부를 구분한다.
