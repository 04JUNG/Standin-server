# Interactive solved state 바이너리 인코딩 (v2)

## 왜 바꾸나

운영 미리보기(2026-11-12 19:31~21:10 KST 로그)에서 같은 51.7MB 상태 패킷을 매번 다시 해석했다.
실수 하나를 `{"$f64":"16hex"}` JSON 객체로 담는 v1 형식 때문이다.

| 위치 | 단계 | 운영 실측 |
|---|---|---:|
| Converter API | `body_metadata` (체형 base.json 전체 디코드) | 5~9 s |
| Converter API | `validate_before_cache` (상태 디코드·검증) | 6 s |
| Converter API | `core_rebind` (재사용 시 디코드·재인코딩) | 17~19 s |
| BFF (0.5 vCPU) | `artifact_packet_validation` | 15~33 s, 검증 중 폴링 정지 |
| Blender 자식 | 상태 생성·영수증 모델 해시 | 측정 안 됨 (로컬 수 초) |

캐시가 있는 미리보기 약 40초에는 Blender 계산이 없다. 전부 이 해석과 검증 비용이다.

## 형식

논리 내용은 v1과 같다. 무손실이며, v2를 디코드하면 v1 payload와 같은 float·int가 나온다.
테스트가 `canonical(v2.payload) == v1 바이트`를 확인한다.

```
"STNDSPS2" | u32 S | 상태 헤더 (v1 canonical JSON: encoding, schema, identity, safety, dependencies) | 0-pad
"STNDMDL2" | u32 M | 모델 헤더 (v1 canonical JSON: schema, coordinates, capabilities, bones[{name,parent}],
                    meshes[{name, vertices, normals, triangles, influences}]) | 0-pad
bones f64[B*3*16]   (restLocal, poseLocal, inverseBind 순)
메시마다: positions f64[V*3] | normals f64[N*3] | triangles u32[T*3] | normalTriangles u32[T*3]
          weightOffsets u32[V+1] | weightBones u16[W] | weightValues f64[W]
```

- 모든 정수는 little-endian이다. 각 구간은 8바이트 정렬이고, 0 padding도 canonical 형식에 포함된다. 뒤에 남는 바이트는 허용하지 않는다.
- `encoding` = `standin-binary-f64le-v2`, `/model` media type = `application/vnd.standin.solved-pose-state-v2`.
- `stateHash` = SHA256(전체 바이트).
- `modelSha256` (= 영수증 `packetModelSha256`) = SHA256(`STNDMDL2`부터 끝까지).
- 검증은 v1 `validate_model`과 같은 계약이고, canonical 조건이 추가된다.
  - 정점별 bone id는 강하게 증가해야 한다(중복 불가).
  - padding은 0이어야 한다.
  - 헤더는 canonical JSON이어야 한다.
- 구현:
  - Python: `converter/solved_state_v2.py`
  - JS: `converter/shared/interactive_pose_v2.mjs`. BFF·Client에 byte-for-byte로 vendoring한다.
  - JS는 typed array view를 반환하고, 기존 스킨 핀용 `triangleIndexJson`과 호환용 `toV1Model`을 제공한다.

## 협상과 하위 호환

- Converter는 v1과 v2를 모두 지원한다. BFF가 `POST /interactive/v1/solve`에
  `X-Standin-State-Encoding: standin-binary-f64le-v2`를 보낸 작업만 v2로 만든다.
  - 헤더가 없으면 지금과 같은 v1이다.
  - 모르는 값은 400 `UNSUPPORTED_STATE_ENCODING`이다.
  - 응답 헤더 `X-Standin-State-Encoding`과 `packet.canonicalEncoding`이 실제 인코딩을 알려준다.
- v2 solve 응답의 `packet`에는 `canonicalState`가 없다. 대신 `stateBytes`, `modelSha256`이 들어간다.
  BFF는 `GET /snapshots/{id}/model`로 바이너리를 받는다.
- v2 스냅샷의 `/variants`는 디코드 없이 바로 409 `VARIANTS_UNAVAILABLE`을 돌려준다.
  - v1 손 변형 atlas는 변형마다 전체 typed JSON 상태를 담는다.
  - 운영 체형(51.7MB 상태)에서는 이미 60MB 예산을 넘어 항상 409였다.
  - 따라서 v2에서 기능이 줄지 않는다.
- 재사용 키(요청 해시, core 디렉터리)에 인코딩이 들어간다. 같은 선택이라도 v1과 v2는 다른 묶음이다.
- 운영 스위치:
  - `CONVERTER_INTERACTIVE_BINARY_STATE=0` (기본 1): v2 요청에도 v1을 돌려준다(긴급 복구).
  - 값이 잘못되면 기동에 실패한다.

## 체형 base 내부 저장 (계약 영향 없음)

- `prepare`는 `base.json`을 작은 인덱스로 쓴다: `interactive-body-base-v2`, binding, runtime, `snapshotSha256`, `modelSha256`.
  모델은 `base.model.bin`(v2 모델 컨테이너)으로 따로 쓴다.
- API 프로세스는 카메라용 bone만 디코드한다. 전체 모델을 typed JSON으로 디코드하던 5~9초가 없어진다.
- Blender 자식의 장면 대응 검사(`load_body`의 `compare`)는 같은 데이터로 그대로 수행한다.
- 리더는 v1과 v2를 모두 읽는다. `CONVERTER_INTERACTIVE_BODY_FORMAT=v1`이면 이전 형식으로 쓴다(되돌리기).

## 배포 순서

1. **Converter (이 변경).** 기본 동작은 v1 그대로이고, 체형 base만 v2로 저장한다.
2. **BFF.**
   - 클라이언트가 v2를 지원한다고 알리고, 운영 스위치가 켜진 작업만 v2를 요청한다.
   - v2 검증은 공유 모듈로 한다.
   - `snapshotProof.modelSha256`은 v2 모델 해시가 된다. Converter 영수증도 같은 정의를 쓰므로 일치한다.
   - 스킨 `trianglesSha256` 핀은 `triangleIndexJson`으로 그대로 확인한다.
3. **Client.** v2 지원을 알리고 `/model` media type으로 분기한다. v1 경로는 남긴다.

어느 단계에서 멈춰도 이전 구성 요소와 호환된다. v2 요청을 받은 구 Converter는 v1을 돌려준다.
BFF는 `canonicalEncoding`을 보고 처리한다.

주의:
- 클라이언트는 BFF보다 먼저 배포하면 안 된다. 현재 운영 BFF는 프로토콜 헤더가 정확히 `pose-body-only-v1`일 때만 인터랙티브 경로를 연다.
  `pose-body-only-v1, solved-state-binary-v2`를 보내는 클라이언트는 구 BFF에서 레거시 경로로 떨어진다.
- 새 클라이언트가 배포된 뒤 BFF를 이 변경 이전으로 되돌릴 때도 같은 일이 생긴다.
- v2 작업이 이미 있는 상태에서 BFF를 되돌리면, 그 작업의 `/model`이 409로 실패한다. 작업 만료(1시간)로 영향 범위가 한정된다.
- 긴급 시에는 되돌리기 대신 `INTERACTIVE_BINARY_STATE=0`(BFF) 또는 `CONVERTER_INTERACTIVE_BINARY_STATE=0`(Converter)을 쓴다.

## 검증

- `tests/converter/test_solved_state_v2.py`:
  - v1과의 무손실 일치(-0.0, 비정규 수, 극값 포함)
  - 해시 정의
  - 변조·비정규 바이트 거부(Python): magic, 잘림, 뒤 바이트, NaN, 인덱스 범위, offset, bone id, 가중치, padding, 헤더, 안전 필드
  - Node에서 공유 JS 리더가 Python 바이트를 비트 단위로 같게 읽음. 위 변조 중 7종을 거부하고, 정렬이 어긋난 pooled `Buffer`도 복사해서 읽음
  - 재바인딩 시 모델 바이트 재사용
  - 체형 base 두 형식
  - 이미지 COPY 목록으로 import 가능함
- `tests/converter/test_interactive_transport.py`:
  - 협상 헤더, 운영 스위치
  - v2 `/model` 바이트와 media type, 변조 시 409
  - v2 `/variants`의 즉시 409
- 운영 단계를 로컬에서 재현한 측정. 조건: 합성 모델(정점 139,474개, 삼각형 278,944개, 법선 160,000개, bone 54개),
  Windows/Python 3.12. 이 PC는 v1 기준으로 운영보다 약 2배 느리다.

  | 단계 (운영 로그 이름) | 운영 v1 | 로컬 v1 | 로컬 v2 |
  |---|---:|---:|---:|
  | `body_metadata` | 5~9 s | 11.5 s | 0.07 s |
  | `validate_before_cache` | 6 s | 14.6 s | 0.12 s |
  | `core_rebind` | 17~19 s | 43.1 s | 1.0 s |
  | 영수증 모델 해시 (`export_rebind`/검증 자식) | 미측정 | 12.1 s | ~0 s |
  | 상태 생성 (solve 자식) | 미측정 | 18.1 s | 0.57 s |
  | 상태 크기 | 51.7 MB | 49.2 MB | 18.0 MB |

  같은 상태를 Node 22에서 공유 리더로 읽은 시간: v1 `readSolvedState` 4.0~4.1 s, v2 `readSolvedStateV2` 0.04~0.06 s.
  v1 리더는 해시·JSON 파싱·float 복원만 한다. BFF의 v1 경로는 여기에 zod 검증과 canonical 재직렬화가 더해진다.

  운영 수치가 아니다. 배포 후 같은 단계 로그(`interactive_stage`, `artifact_packet_validation`)로 전후를 비교한다.
- 실제 Blender 왕복(prepare → solve → export 검증)은 Converter CI의 이미지 단계에서 확인해야 한다.
  이 PR의 로컬 검증에는 포함되지 않는다.
