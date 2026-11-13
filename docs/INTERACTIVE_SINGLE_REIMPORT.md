# 인터랙티브 경로: staged FBX를 한 번만 다시 불러온다

## 무엇이 중복이었나

이전 흐름에서는 같은 `staged.fbx`를 두 번 다시 불러왔다.

1. **solve 자식(같은 프로세스)**
   - 원점 하나를 얻으려고 전체 모델을 캡처한다(약 3초).
   - FBX를 저장하고, `reset_scene` 후 다시 불러와 캡처해 화면 패킷을 만든다.
2. **export 시점(새 Blender 프로세스)**
   - 같은 바이트를 다시 불러와 구조를 검증하고 캡처한다.
   - 1번의 패킷과 비교해 영수증을 발행한다.

실제 크기(정점 139,474개) 프로파일에서는 FBX 저장·불러오기 자체가 짧았다(저장 0.85초, 불러오기 0.77초, 검증 불러오기 1.5초). 비용은 순수 Python 루프인 전체 캡처(약 3초씩)와 비교(1.7초), 그리고 export용 Blender 프로세스였다.

## 바꾼 흐름

```
prepare (그대로)
solve 자식    : 체형 장면 대응 검사(그대로) → 손 → 범위 → staged.fbx 저장 → solve.json
publish 자식  : 새 Blender 프로세스. staged.fbx를 한 번 불러와 구조 검증(final_artifact.validate)
                → 캡처 = 화면 패킷 → state·snapshot·verified/ 영수증 → packet.json (마지막에 게시)
export        : verified/ 영수증을 현재 operation/revision에 다시 묶기만 한다(rebind_export)
                실패하면 기존 독립 검증기(export_bundle)로 돌아간다
```

- **원점 계산:** `scene_origin()`은 `capture()`와 같은 장면 검사(뼈대 1개, 애니메이션 없음, rest에 구워짐)와 같은 mathutils 식으로 계산한다. 값은 비트 단위로 같다.
- **영수증:** 형식은 기존 검증기와 같다(`verified_receipt`). `packetModelSha256`과 `reimportModelSha256`은 같은 값이다. 패킷이 곧 그 재로드 결과이기 때문이다.
- **재사용:** 핵심 재사용(core) 캐시에 `verified/`가 함께 복사된다. 외형만 바꾼 다음 revision도 영수증을 다시 묶기만 한다.
- **로컬 도구:** `solve_from_body(..., publish=True)`는 기존 단일 프로세스 동작(같은 프로세스에서 다시 불러오기, 영수증 없음)을 유지한다.

## 안전 조건

**유지한 것**
- 체형 base의 장면 대응 검사(`load_body`)
- 새 프로세스에서의 FBX 구조 검증
- 영수증은 새 프로세스 검증기만 발행한다는 원칙
- export 시점의 revision·취소 펜스
- BFF의 소유권·영수증 바인딩 검사

**빠진 것:** 같은 바이트를 두 프로세스에서 각각 불러와 서로 비교하던 교차 확인 하나다. 이제 화면 패킷이 새 프로세스의 그 재로드 결과 자체이므로, "보이는 것 = 파일을 다시 불러온 결과"는 구성상 성립한다.

## 함께 고친 것

v1(typed JSON) 상태의 export 검증기가, 이미 해석한 모델이 있는데도 상태 전체를 다시 해석해 모델 해시를 계산하고 있었다(#231 이후 약 10초 추가). 이제 메모리의 모델로 계산한다.

## 검증

- **실제 Blender**(`tests/converter/verify_interactive_binary.py`, Converter CI의 이미지 단계에서 실행): 새 흐름 10개 항목을 포함해 25/25 통과.
  - v1·v2 모두 게시된 패킷이 이전 방식(같은 프로세스 재로드) 패킷과 **바이트 단위로 같다**.
  - 원점이 같다.
  - 미리 발행한 영수증이 다시 묶인다.
  - 독립 검증기도 게시된 묶음을 통과시킨다.
- **단위 테스트**(`tests/converter/test_interactive_publish.py`)
  - 영수증 형식
  - `safety`를 표면 해석 없이 읽을 수 있음
  - export가 검증기를 다시 돌리지 않음
  - 변조 시 독립 검증기로 대체
  - core 재사용 시 새 revision으로 재바인딩
- **실제 크기 로컬 측정**(Docker linux/amd64, `male-base-v0-8-0`, 라이브러리 BVH 1개, 1회 측정):

  | | 이전 | 새 흐름 |
  |---|---:|---:|
  | v2 미리보기 준비 | 17.5 s | 20.6 s |
  | v2 export | 14.0 s | 1.2 s |
  | v2 다운로드 가능까지 | 31.5 s | **21.8 s** |
  | v1 다운로드 가능까지 | 71.0 s | **53.8 s** |

  미리보기 준비는 약 3초 늘어난다(새 프로세스 시작 약 1.6초와 구조 검증 1.5초가 미리보기 쪽으로 옮겨짐). 대신 export에 Blender 프로세스가 더 이상 필요 없다. 운영 수치는 배포 후 `interactive_stage`의 `blender_solve`·`blender_publish`·`export_published`로 비교한다.
