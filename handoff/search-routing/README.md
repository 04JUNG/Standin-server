# 검색 라우팅·자연어 검색 참고본

이 브랜치는 2026-10-01 로컬 작업 트리에서 **검색 관련 코드와 문서만 선별한 참고용 워크트리**다. 기준 커밋은 344adfb이고, 원래 작업 트리의 refine·배포·리그 수정은 가져오지 않았다.

## 어떤 검색인가

| 경로 | 살펴볼 코드 | 상태 |
| --- | --- | --- |
| 컷의 기본 라우팅 | src/routing.py → src/pipeline.py → src/search.py | 기존 운영 경로. shot에 따라 skip/bust/core를 고르고 core에서 기하 kNN 실행 |
| 사용자 문장 검색 | api/app.py의 POST /semantic-search, src/semantic_service.py, src/semantic_search.py | 기존 opt-in API. 별도 E5 모델·검증된 빌드가 필요하며 이미지 라우터의 입력이 아님 |
| 이미지 기반 다이나믹 라우터 | src/vlm/rough_slots.py, src/experimental/rough_router.py, rough_router_shadow.py | 인물별 슬롯을 G(기하), S(의미), F(관측 사실), D(대안) 채널로 **계획**. 기존 후보는 바꾸지 않음 |
| 러프 슬롯의 의미 검색 분기 | src/experimental/rough_semantic.py, rough_semantic_v2.py, rough_semantic_service.py | 별도 E5 색인으로 S 채널을 실행한 로컬 실험. 이 브랜치에는 API 훅이 없음 |
| 다중 시점 설명 실험 | src/experimental/rough_multiview.py, pose_captioning.py | 시범 구현. 전체 포즈 설명·색인은 이 브랜치에 없음 |
| 기하+의미 후보 결합 | src/experimental/intent_fusion.py | 오프라인 shadow 결합 실험. 운영 최종 랭킹은 아님 |

src/routing.py의 세 갈래와 rough_router.py의 인물별 검색 채널은 서로 다른 단계다. 사용자 텍스트 /semantic-search도 이미지 전용 SearchPlan과 분리되어 있다. 부위별 기하 검색·상하체 조합·공통 카메라 보정·refine은 이 참고본에서 완료된 기능으로 간주하지 않는다.

## 먼저 실행해 보기

저장소 루트에서:

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install numpy
python examples/search_routing_demo.py
python tests/test_search_routing_handoff.py
~~~

예제는 합성 슬롯과 관절로 SearchPlan만 출력한다. API 키, BVH, DB, E5 모델을 쓰지 않는다. S 채널은 질의 계획을 뜻하며 예제 실행 자체가 실제 포즈 검색은 아니다.

기존 사용자 텍스트 API는 [BFF 계약](../../docs/API_BFF_SEMANTIC_HANDOFF_2026-08-18.md)과 [서비스](../../src/semantic_service.py)를 참고한다. 실제 호출에는 SEMANTIC_ENABLED=1, 같은 geometry DB에 맞는 pinned semantic build, E5 모델 파일이 필요하다. 현행 규칙 파서는 짧거나 모호한 문장을 clarify로 돌릴 수 있으므로, 차기 v3.3 설계를 현행 동작으로 설명하지 않는다. 이 워크트리에는 모델·DB·BVH·실제 러프 이미지·평가 아티팩트를 넣지 않았다.

## 읽는 순서

1. [현행 라우터 코드](../../src/experimental/rough_router.py)와 [9/22 초기 규칙](../../docs/SEARCH_ROUTER_RULES_DRAFT_2026-09-22.md): 조건별 G/S/F/D 분기, 인물 귀속, 가시성. 과거 그룹 제외 규칙은 9/26 실행 기록보다 오래됐다.
2. [러프 의미 검색 실행 기록](../../docs/ROUGH_SEMANTIC_PIPELINE_2026-09-26.md)과 [구조 검토 개선](../../docs/ROUGH_SEMANTIC_FIX_2026-09-26.md): S 채널을 실제 BVH member까지 연결한 실험.
3. [기하·의미 결합 실험](../../docs/SEARCH_INTENT_AB1_INTEGRATION.md): 후보 출처와 A/B1 경계.
4. [전체 목표와 미구현 단계](../../docs/SEARCH_DYNAMIC_ROUTING_IMPLEMENTATION_PLAN_2026-09-22.md): P0–P7 로드맵. 초기 제안은 완료 상태가 아니며 위 실행 기록을 우선한다.
5. [사용자 텍스트 검색 계약](../../docs/API_BFF_SEMANTIC_HANDOFF_2026-08-18.md)과 [차기 v3.3 설계](../../docs/SEMANTIC_SEARCH_DENSE_FIRST_V3_DESIGN.md). v3.3은 현재 API 런타임이 아니다.

## 참고 시 경계

- 이 워크트리는 현재 실험 코드를 **원래 경로 그대로** 옮긴 참고본이다. src/config.py, src/pipeline.py, src/vlm/client.py, src/schema.py, api/models.py에 있던 로컬 연결 수정은 다른 기능 수정과 섞여 있어 가져오지 않았다. 따라서 ROUGH_SEMANTIC_ENABLED=1만 지정해도 여기의 /analyze에서 새 러프 의미 후보가 나오지는 않는다.
- 실험 평가 스크립트는 원본 artifacts/, data/poses.db, BVH, E5 파일을 요구한다. 이 워크트리에서는 인수인계용 소스다. 예제와 테스트는 자산 없이 실행 가능하다.
- S 검색의 임베딩 유사도는 자세 정답률이 아니다. 일부 로컬 실험의 구조 검사 통과율은 동일 규칙으로 후보를 정렬·판정한 수치이며 독립 작가 평가가 아니다.
- 새 후보는 전신 BVH member이고 refine_allowed=false다. 부위별 문서 검색이 상·하체를 실제로 조합했다는 뜻은 아니다.

## 팀원에게 전달

브랜치: codex/search-routing-handoff

로컬 워크트리: /Users/dowon/dev/Standin-search-routing-handoff

원격에 브랜치가 올라간 뒤에는 git fetch origin codex/search-routing-handoff로 받아 이 README부터 읽으면 된다. 원본 작업 트리의 미커밋 수정이나 개인 자산은 브랜치에 포함되지 않는다.
