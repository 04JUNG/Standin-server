# 기본 FBX 재사용과 appearance 변경 — 2026-11-12

포즈·체형·shape·rig·scope·mirror·카메라·실제 손 선택이 같으면 이미 계산된 private core bundle을 재사용한다. owner/person/candidate도 일치해야 한다. 스킨·외곽선 변경으로 operation/revision이 바뀌어도 Blender prepare/solve를 반복하지 않는다. `.blend`·staged FBX는 복사해 old operation 취소/삭제와 새 operation 수명을 분리하고, 새 canonical state/packet/stateHash를 발행한다. base lock은 동일 core의 중복 생성을 막는다. 기존 TTL/디스크 예산이 적용된다.

첫 최종 FBX는 기존 독립 export/reimport 검증을 그대로 수행한다. 동일 core의 다음 operation은 검증된 bytes의 SHA·크기·model SHA·runtime·owner binding을 재검사한 뒤 새 operation/revision/identity에 맞는 receipt를 발행한다. staged FBX는 canonical state 생성 **이전**에 작성되며 exporter에 operation/stateHash가 전달되지 않는다. frozen exporter의 custom property export는 기본 비활성이고 state/receipt는 FBX 외부에 보관하므로 파일 내 authority metadata를 낡게 유지하는 방식이 아니다.

스킨/외곽선 후처리는 source FBX SHA·모델/UV/asset manifest/runtime·선택 스킨/폭 등 모든 content 입력을 키로 한 최대 128 MiB process-private LRU를 사용한다. 같은 입력의 동시 요청은 singleflight로 처리하고 검증 완료된 bytes만 저장한다. cache hit에서도 현재 BFF binding을 새 receipt에 부여하며 BFF의 권한/현재 선택/취소/receipt 검사는 완화하지 않는다. 다른 스킨이나 폭은 필요한 후처리만 다시 수행한다. 재시작은 캐시를 폐기한다. 다운로드 권한을 캐시가 부여하지 않는다.

검색 serving의 hand quality gate, 손 self-collision 품질 gate, 선택한 손 triangle screening은 더 이상 실행하지 않는다. 과거 ON 배포값도 OFF로 정리한다. 손 라이브러리 선택/변형은 유지하며, source/hash/rig/finite transform·body bone 보존·FBX round-trip 일치 검사는 파일 무결성 계약으로 유지한다. 새 solved state는 `safety.handChecksPassed=false`로 손 품질을 검사하지 않았음을 명시한다. 과거 true packet도 읽을 수 있다. bodyAccepted 및 finalFileVerified 계약은 유지하고, 미검사를 안전 인증으로 취급하지 않는다. 명시적 진단 도구는 serving과 별도로 남긴다.

미리보기는 앱에서 로컬 appearance를 먼저 변경하고, 검증된 다운로드는 별도 background 작업 완료 뒤 허용한다. 이 변경은 운영 배포/자산 교체를 수행하지 않는다. 단위 검증은 동일 core의 두 revision에서 prepare/solve가 한 번만 호출되는지, owner/source/shape/실제 손 변경 miss, old 취소와 새 bundle 수명 분리, 변조 파일 거부, receipt 재바인딩, appearance singleflight 및 메모리 제한을 확인한다. 실서비스 초 단위 개선은 별도 실측이 필요하다.
