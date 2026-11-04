# 오프라인 예시 보고서

`python scripts/demo_evaluation_extension.py --output out/eval/extension-demo-new-version --public-reports out/eval/public-example-reports`로 재현한다. 입력은 저장소의 합성 회귀 fixture다. 실제 323장, provider, Blender, 설치 앱, CSP는 사용하지 않았다. 사람 라벨은 없으며 활용성은 unknown으로 남긴다. 일부 구조/무결성 차원의 fixture pass는 실제 FBX 승인과 다르다.

- experiment-unknown: 대상 2인 × arm 3개 = 6행. 1인은 후보 누락, v3 adapter 미설정, F25 정확한 기권 복구. 품질 라벨 없음 → INCONCLUSIVE.
- experiment-failure: `refined=false`인데 BVH가 바뀐 exact recovery 계약 위반 → FAIL.
- export-unknown: 사전 과제 2개 중 1개만 synthetic 관측, 실제 사람 평가 없음 → INCONCLUSIVE.
- export-failure: native 저장 노드의 SHA가 download와 다름 → FAIL.
- pipeline-unknown: 사전 실험과 최종 과제 2개의 연결은 고정했으나 실제 export 측정 없음. 분모 2 유지 → INCONCLUSIVE.
- save-contracts: 50/50 오프라인 프로토콜 계약 통과, 실제 제품 품질 → INCONCLUSIVE.

이 예시의 timeout/정책 수치는 regression fixture용이다. 제품 SLO·출시 표본 수로 사용하지 않는다. 원본 이미지나 private 경로는 보고서에 포함하지 않는다.
