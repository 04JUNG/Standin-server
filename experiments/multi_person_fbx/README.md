# 04J-289 다인 FBX 자동 수정 실험

러프의 인물별 포즈·몸통/머리 방향·출력 범위와 상대 배치를 추정해 한 컷의 여러
인물을 하나의 FBX에 담는 로컬 실험 코드다. **서비스의 ‘3D 선택’ 버튼과 추론 API에
연결된 제품 기능은 아니다.** 통합 변환 API의 범위는
[FBX_MULTI_PERSON_SCENE.md](../../docs/FBX_MULTI_PERSON_SCENE.md)를 참고한다.

## 코드 구성

- `sources/04j-289-set3-orientation-v3`: set3 13컷·34명 방향/범위/기본 자세 실험.
- `sources/04j-289-scene-interactions-v4`: p060·p220·p0718의 장면 관계, 떨어진 손의
  소유자, 양팔 리파인 및 기존 배치 보존 실험.
- `sources/04j-289-set3-layout-v2/layout_math.py`: 화면상 머리 위치·크기 맞춤과
  실루엣 겹침 최소 보정.
- `sources/04j-289-set34-refine-scenes/align_refine.py`: 라이브러리 BVH의 몸 방향 정렬.
- `sources-manifest.json`: 실제 로컬 실행 소스의 원본 SHA-256. 스냅샷은 당시 코드
  그대로 보존했으므로 로컬 경로와 런타임 경로가 들어 있다.
- `stage_sources.py`: 검증한 스냅샷을 개인 작업 공간에 복사하고 실행 파일·체크아웃·
  폰트 경로만 치환한다. 기존 스크립트는 덮어쓰지 않는다. 포즈 정책은 변경하지 않는다.
- `evidence/`: 측정한 파일 정합성·배치 검증 수치. 이미지 품질 합격 자료는 아니다.

공통 후보 통합/방향/범위/접촉 측정은 `src/experimental/`의 opt-in 모듈에,
기하 기반 엎드림·포갠 팔 보정은 `converter/contact_rest.py`와
`converter/folded_forearm_rest.py`에 있다. 이 모듈들은 기존 서비스 경로에 자동 연결되지 않는다.
기존 retarget 핵심 코드는 변경하지 않는다.

## v4 처리와 결과

전체 장면과 인물 단독 관측을 새로 분석하고, 머리 방향과 몸통 표면을 따로 판단한다.
기존 crop 밖의 큰 손은 머리/손 크기 관계로 소유자를 추정한다. 원본 손·관절과 장면의
상대 인물 위치에서 팔 목표를 만들며, 컷별 각도·관절 좌표·포즈 ID를 지정하지 않는다.
기존 좋은 v2 머리 중심과 앞뒤 순서를 우선 보존한다.

| 컷 | 반영한 변경 | 남은 차이 |
| --- | --- | --- |
| p060 | 오른쪽 몸통 왼쪽 옆모습, v3의 불필요한 배치 이동 복구 | 왼쪽 양손/휴대폰 동작과 소품 |
| p220 | 앞 인물의 뻗는 팔과 상반신, 뒤 인물의 전신 뒷모습 | 걸음/소품, 손가락 모양 |
| p0718 | 서로 향하는 양쪽 방향, 가운데 전신 서기/양손 제지 | 팔 높이·정확한 접촉·대립/제지 강도 |

3컷·7명의 좋은 v2 머리 중심 추가 이동과 앞뒤 순서 반전은 0이다(부동소수 오차 제외).
p220 앞 인물은 얼굴 일부를 전체 머리로 보던 크기 추정을 교정해 화면 크기와 추정 깊이가
변했다. 같은 위치/크기가 모두 고정된 실험으로 해석하면 안 된다. 깊이는 실제 거리 복원값이 아니다.

긴 변 1200px 검사에서 최종 실루엣 겹침은 3/3컷 0px다. 모든 인물은 52본을 유지한다.
Blender와 Three.js로 raw/final 통합 FBX 6개를 검사했다. Three.js 최대 오차는
스킨 0.192mm, 원시 정점 0.223mm, 원본 재투영 0.0401px다. 이전 0.1mm 단일 검사는
실패했고 최종 월드 허용오차는 1mm, 재투영은 0.1px로 명시했다.
동작 의미·원본 일치도 및 CSP 직접 불러오기는 별도 검증이 필요하다.

## 실행 전제와 재현 범위

새 clone만으로 러프를 자동 다운로드하거나 전체 실행하지 않는다. 저작물인 러프,
라이선스가 있는 BVH/FBX, 모델 가중치, 생성된 FBX와 이미지 내장 HTML은 Git에 포함하지 않는다.
팀 공유용 HTML은 별도 파일로 전달한다. 모든 모델 추론은 로컬이며 외부 API 업로드 단계가 없다.

실험 작업 공간은 다음의 **권한 있는 로컬 자산과 이전 자동 산출물**을 요구한다.

- `outputs/04j-289-set34-refine-scenes`: `contact-index.json`, 컷별 `people.json`, 기존 선택 기록.
- `outputs/04j-289-set3-layout-v2`: 원본 입력 목록, 컷별 `layout.json`, 비교용 이미지/FBX.
- v3 결과의 `selection.json`, `geometry.json`, `layout.json` 및 비교 이미지/검증 기록.
- v4 입력의 `inputs.json`, `identities.json`, 머리 ID가 표시된 `scene-context.png`.
  장면 viewport는 인물 bbox 합집합에서 세로 여유를 주고 전체 가로폭을 유지한 자동 crop이다.
  이 ID 목록은 동일 인물 비교 기준이며 새 인원 검출의 정답률 검증은 아니다.
- `tmp/standin-local-routing/latest-bound-library.json`과 연결된 라이브러리, 원본 BVH,
  `outputs/04j-289-fresh-multicut-experiment/library-body-v2.json`.
- `pose-head-contact-repair-20261015/mesh-review/assets/standin-master-v2.fbx` 및
  `master-head-axes.json`. 현재 공용 인체로 체형·옷·머리카락은 복원하지 않는다.
- HumanArt manifest/가중치, Qwen3-VL-8B-Instruct-4bit와 CLIP의 오프라인 캐시.
  기본 캐시 위치는 각 스크립트의 `HF_HOME`을 따른다.
- `routing_repo`: semantic285 어댑터와 fullbody-routing 코드가 있는 별도 체크아웃.
  이 PR이 그 라우팅 브랜치까지 병합하는 것은 아니다.

Python의 numpy/scipy/Pillow/OpenCV, 기존 서버 의존성 및 HumanArt 환경이 필요하다.
Qwen은 Apple Silicon MLX-VLM, CLIP은 torch/transformers 환경을 별도로 사용한다.
Blender는 bpy가 포함된 실행 파일로 호출하며 Three.js 설치 디렉터리를 지정한다.
입력 JSON의 절대 경로는 팀 작업 공간에 맞게 준비해야 한다.

```sh
python experiments/multi_person_fbx/stage_sources.py --runtime /private/path/runtime.json --validate-only
python experiments/multi_person_fbx/stage_sources.py --runtime /private/path/runtime.json
```

`runtime.example.json`을 개인 경로로 채운다. 빈 작업 공간에 소스를 먼저 배치하고
위의 권한 있는 자산/메타데이터를 준비한다. 원본 실험 폴더 덮어쓰기는 금지한다.

v4의 추론 단계는 `analyze_scene.py` → `verify_scene.py` → `fresh_keypoints.py`,
`contrastive.py` → `find_hands.py` → `localize_hands.py`다. 각 모델에 맞는 환경을 사용한다.
일부 VLM 출력이 객체 대신 배열로 반환되었으므로 다음 계획 단계 전에
`surfaces-review.json` 등에서 `analysis.people`와 `analysis.relations` 구조 및 인물 ID를
검증해야 한다. 당시 원문은 보존하고 JSON 구조만 정규화했다. 장면 설명의 환각은 정답으로 쓰지 않았다.

그 다음 staged `run_experiment.py`는 계획 → 인체 변환 → 배치 → 팔 리파인 → 재배치 →
통합 FBX → Three.js 검사 순으로 실행한다. `build_review.py`가 결과 HTML과 묶음을 만든다.
분석 스크립트는 기존 출력이 있으면 건너뛰므로 **새 추론은 새 실행 폴더와 비어 있는 분석 결과**로
실행해야 한다. 사용자 피드백 파일은 추론 입력에서 차단한다.

경로 치환본은 문법·설정·덮어쓰기 방지 검사를 했으며 새 작업 공간에서의 전체 모델 재실행은
하지 않았다. `sources-manifest.json`과 `evidence/`는 원래 로컬 실행 결과에 대한 기록이다.
