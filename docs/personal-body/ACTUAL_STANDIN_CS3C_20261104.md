# 실제 Standin 남성 CS3C 적용 테스트 — 2026-11-04

**결과: 현재 CS3C 경로는 실제 Standin 남성 0.7.6을 등록하지 못한다.** 네 원본 모두 재질 검사에서 `UNSUPPORTED_MATERIAL`로 거절되어 포즈 적용과 CS3C 기반 FBX 출력은 실행되지 않았다. 이전 성공은 다른 관찰된 CS3C 샘플의 범위이며, 이 인체 모델 지원으로 확대할 수 없다.

현재 앱에 포함된 revision4 런타임을 사용했다. client `a044bcf`, converter `1cb6b5dbc`, runtime manifest SHA256 `a3eabed90e4100f5eb7896cb72fb6d01a4362d315c148a58abdeaeab8c439e03`. 네 파일을 한 host에서 검사했으며 최초 host 시작은 7.26초였다. 원본은 수정하지 않았고 부분 등록 카드/미완료 staging은 0개다.

| 실제 원본 | 등록 소요 | 결과 | 포즈 적용 |
| --- | ---: | --- | --- |
| standin-male-0.7.6-mapped-review.cs3c | 3.68초 | UNSUPPORTED_MATERIAL | 미실행 |
| standin-male-0.7.6-arm-raised-review.cs3c | 2.10초 | UNSUPPORTED_MATERIAL | 미실행 |
| standin-male-0.7.6-recorded-and-applied-review.cs3c | 2.00초 | UNSUPPORTED_MATERIAL | 미실행 |
| standin-male-0.7.6-recorded-pose-review.cs3c | 1.99초 | UNSUPPORTED_MATERIAL | 미실행 |

## 확인된 차단 원인

모두 노드 53개, 메시 1개, texture 0개, 재질 1개다. 원본 테이블을 읽기 전용으로 검사한 결과 diffuse는 약 `(0.48, 0.53, 0.59)`, specular는 `(0.12, 0.1325, 0.1475)`, emission은 `(0,0,0)`, opacity는 `1`, shininess는 약 `10.24`다. 현재 CS3C adapter는 specular와 emission이 모두 0인 경우만 인증하므로 이 재질을 거절한다. 재질 검사를 끄거나 반사광 값을 지워 성공으로 만들지 않았다.

이 조건은 `converter/local/cs3c.py`의 material admission 검사와 현재 specular를 0으로 설정하는 변환 구현에 해당한다. 실제 모델 지원에는 반사광 정보를 보존하는 CS3C material mapping과 독립 FBX roundtrip 검증이 필요하다. 이번 요청에서는 제품 지원 범위를 임의로 변경하지 않고 현재 동작을 시험했다.

## 같은 모델의 FBX 대조군

`01-male-0.7.6-neutral.fbx`를 같은 런타임에 등록한 뒤, 당시 저장된 선택 포즈 BVH와 CMU 검수 BVH를 적용했다. 등록·두 pose apply·FBX 출력·출력 SHA 검사가 모두 통과했다. 이것은 동일 인체의 FBX 대조군 결과이며 **CS3C 변환 성공이 아니다**. 두 출력은 각각의 검증을 통과했지만, CS3C가 재질 다음 단계까지 통과하는지는 아직 검증되지 않았다.

실제 CS3C가 등록 단계에서 차단되어 이번에 CSP에 넣을 CS3C 기반 출력은 생성되지 않았다. CSP 가져오기·live BFF 전체 E2E 성공을 주장하지 않는다. 기존 CSP 종료 문제의 원인을 이 재질 문제로 단정하지도 않는다.

- [원본 hash·재질 값·실행 증거](verification-actual-standin-cs3c-20261104.json)
- 제품 코드·원본 모델·재질 guard 변경 없음. 원격 업로드·유료 추론·merge·배포 없음.
