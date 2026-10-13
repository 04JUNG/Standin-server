# 스켈레톤 전신 우선·부위 보완 실험 — 2026-10-13

## 결론과 범위

스켈레톤 검색에도 상체/하체 분리 검색을 적용할 수 있다. 다만 이번 구현은
`src/experimental/geometric_parts.py`의 **오프라인 참고 후보 실험**이며 Pipeline/HTTP/운영 경로에는 연결하지 않았다.
전신 후보를 덮어쓰거나 confidence를 승격하지 않고, 합성 BVH를 만들지도 않는다.
부위 후보는 `observed_part_geometry_reference`, low confidence, refine/release 금지다.

## 전신 먼저, 부족한 부위만

1. 기존 추출·전신 검색을 그대로 실행한다. 정상 high-confidence 전신 결과가 있으면 부위 검색 0회.
2. 전신이 없거나 신뢰 부족이면 관측된 각 부위만 검사한다. 상체는 양 어깨·팔, 하체는 양 골반·다리다.
3. 기존 유효 관절/뼈 마스크를 사용한다. 해당 부위 뼈 최소 3개, 양쪽 anchor, 크기/길이/소유 인물 검사를 통과해야 검색한다.
4. 쿼리와 기존 라이브러리 투영을 같은 anchor 중점·너비로 정규화한다. 상체 단위는 어깨 너비, 하체 단위는 골반 너비다.
5. 전신 후보가 있으면 그 후보의 부위 거리와 새 후보의 **같은 부위 거리**만 비교한다.
   절대 개선 .1 및 상대 개선 25% 둘 다 넘는 경우에만 보완 참고 후보를 추가한다.
   전신 거리와 부위 거리 또는 상체와 하체 거리를 직접 비교하지 않는다.

기하 순위에 VLM 자연어/동작/시점 사전확률을 넣지 않는다. 빈 관절을 추정해 채우지 않는다.
기존 head/entangled 차단·quarantine·family dedup·refine 정책을 유지한다.
DB/원본 BVH/태그/feature schema 및 production config를 변경하지 않았다.

## 같은 러프 전후 비교

승인된 T008/T009/T011/T012/T013/T014/T028/T044를 current-X로 다시 추출했다.
이전 실험은 좌표를 저장하지 않았으므로 기존 실제 Gemini 실험의 **최종 인물 박스**와 body-scope 메타데이터를 fixture로 사용했다.
이는 원래 VLM 박스/관계 메타데이터의 정확한 재현이나 새 Gemini E2E 실험이 아니다.
action/view는 neutral, 관계는 SOLO/TALKING fixture이며 이전 분석의 원래 관계 판정을 재사용한 것이 아니다.
공정한 전후 비교는 이번 재추출 좌표·유효 마스크·전신 후보를 양쪽이 동일하게 사용한다는 범위다.
추가 VLM 호출 0회. T008/T009와 T011/T012는 유사한 그림의 변형이라 독립 동작 8개로 해석하지 않는다.

| 러프·인물 | 전신 검색 | 부위 보완 후 | 해석 |
| --- | --- | --- | --- |
| T008 1 | 없음 | 없음 | 관측 스켈레톤 없음 |
| T009 1 | 없음 | 없음 | anchor/크기/뼈 증거 부족 |
| T011 1 | 없음 | 하체 Top5 참고 | 추출기의 다리 라벨 기준 검색. 관절 위치/라벨은 육안상 의심, 정답 인정 금지. 1위 combat_ual2_Shield_Dash_RM_f00200 |
| T012 1 | 없음 | 없음 | anchor/크기/뼈 증거 부족 |
| T013 1 | 없음 | 없음 | 부위 증거 부족 |
| T013 2 | 없음 | 없음 | 상체 중심이 다른 인물 박스에 들어가 소유 인물 모호 → 차단 |
| T013 3 | 없음 | 없음 | 관측 스켈레톤 없음 |
| T014 1 | 없음 | 없음 | anchor/크기/뼈 증거 부족 |
| T028 1 | 없음 | 하체 Top5 참고 | **실제 V자 팔을 RTM이 다리로 라벨링. 잘못된 해석을 검색이 따라감** |
| T028 2 | 없음 | 상체 Top5 참고 | 잘린 옆 인물의 해부학적 라벨 자체가 의심스러움. 정답 인정 금지 |
| T044 1 | 기존 Top5 | 동일 Top5 | 부위 검색 0회. ID/view/distance 보존 |

8컷·11슬롯: 정상 전신 1명 그대로, 부위 참고 추가 3명(하체 2/상체 1), 나머지 7명 보완하지 않음.
전신 후보 인물 1 → 전신 또는 부위 참고가 있는 인물 4는 **정확도 상승이나 완성 포즈 4개가 아니다**.
전체 전신 후보 동일, 부위 검색 3회, 합성 BVH 0개.
실제 사례에서는 보완된 세 슬롯 모두 전신 후보가 비어 있었다.
전신 후보의 일치한 하체를 보존하고 틀린 상체만 보완하는 분기는 단위 fixture로 검증했고,
실제 러프에서 해당 분기의 품질 개선은 아직 검증하지 않았다.

## 다음 개선

구조적 유효성만으로 관절 이름의 의미적 정확성을 보장할 수 없다.
T028의 팔↔다리 전도 및 잘린 인물의 환각 관절을 우선 진단해야 한다.
T011도 손목/발목 위치가 가까이 겹치는 등 관절 의미·위치가 의심스럽다. 부위 검색 실행 성공을 품질 성공으로 보지 않는다.
의심 슬롯은 낮은 신뢰 참고에 머물고 시맨틱 fallback/보류를 유지한다.
관절 라벨을 자연어로 임의 수정하거나 안전 임계값을 낮춰 통과시키지 않는다.
다인 소유 검사는 현재 보수적인 peer-box 중첩 차단이므로 정확한 person crop/소유 판정 검증이 필요하다.
현재 검색은 기존 4개 투영(front/side/three_quarter/back)이며 새로운 최적 카메라 탐색이 아니다.
상하체 후보를 실제 합성하려면 별도 retarget·고정 부위 보존·손목/관절 제한·실제 mesh 충돌 검사까지 필요하다.

## 재현

모델 및 캐시가 준비된 환경에서:

```sh
python scripts/capture_geometry_snapshot.py --reference <archived-live-results.json> --images <approved-image-root> --db <poses.db> --out <new-capture-dir>
python scripts/eval_geometric_parts.py --snapshot <new-capture-dir>/capture.json --db <poses.db> --out <new-comparison-dir>
python -m pytest tests/test_geometric_parts.py -q
```

캐시 경로는 실행 환경의 TORCH_HOME으로 지정한다. capture CLI는 실제 current-X를 사용하며 Gemini client를 만들지 않는다.
이번 근거: `tmp/geometric-parts-current-X-capture/capture.json`, `tmp/geometric-parts-eight-v1/results.json`.
두 파일은 로컬 ignored 실험 산출물이다. 논리·재현 스크립트·테스트는 버전 관리한다.

## 검증

부위 검색 단위 23 passed. 서버 회귀 916 passed / 8 skipped / 22 subtests passed
(Blender 전용 `tests/converter` 제외), 스모크 54/54, compileall 및 diff-check 통과.
기존 FastAPI/httpx 및 tar extraction deprecation 경고 2개는 남아 있다.
전후 비교의 핵심 4컷·7슬롯 모든 후보 선택과 원본 이미지 decode,
320/360/736px 레이아웃 및 스크립트 오류 없음 확인.
