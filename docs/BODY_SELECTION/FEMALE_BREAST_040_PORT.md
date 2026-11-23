# Female 0.4.0 breast7 port

2026-11-23 후속 요청으로 검증된 코드 변경만 main에도 반영한다. 아래의 main 미병합 기록과 validation JSON은 최초 검수 완료 시점의 이력이다. 이 코드 반영으로 새 자산을 활성화하거나 운영 manifest/task pin, 앱 릴리스·업데이트 피드를 변경하지 않는다. develop의 다른 작업은 포함하지 않는다.

현재 서비스의 `standin-female-body-0.4.0`에 원 breast7의 가슴 조형과 잔차 합성 정책을 이식한 별도 검수 후보다. 운영 활성화, S3 쓰기, main 병합, 앱 릴리스 및 업데이트 피드 변경은 포함하지 않는다. 기존 카드·캐릭터·소스·저장 파일과 기본 선택은 유지한다.

## 계보와 이식 범위

원 변경은 서버 [#145](https://github.com/04JUNG/Standin-server/pull/145), 클라이언트 [#163](https://github.com/04JUNG/Standin-client/pull/163)이며 이미 병합됐다. 원 후보 바이너리를 찾고 문서의 SHA-256과 대조했다. 원 조형 생성기를 다시 실행해 비슷한 모양을 만든 결과가 아니다.

| 입력 | SHA-256 |
| --- | --- |
| 현재 0.4.0 Blend | `5a62d2f3fb49d77fdc04d1c78a361edf9fbab55e71e0e47d3b6b393fae14d3a9` |
| 현재 0.4.0 controller | `1164dff9a4c6069d0da37b80dd26c61544bc6f9eb9c9f3469c33d457e893e9af` |
| 원 breast7 Blend | `c2d3a81e52eda54c0fed3d23aa3e45375f99989796f10fab9f2578e0d98a843a` |
| 원 breast7 controller | `1cdb04c45e34ecf054da0769479f9f28c5e62a8c6f5154340b1b28a25789ae2d` |

2026-11-23 AWS 읽기 전용 재조회에서 Converter의 manifest `f0169b4813d70877e924965e0cc9f212b499aabac52c1cf6a4848e41858968a5`와 위 현재 소스 해시를 확인했다. 원 후보는 24,994정점·52본, 현재 모델은 105,430정점·54본이다. 가슴의 유효 지지 정점은 좌표가 직접 대응한다. 1µm 초과 Small 잔차 1,745개와 Extended 1,133개가 모두 정확히 대응한다. 비대응 경계의 최대 Small 전송량은 0.000153mm이며, 그 경계만 원 표면의 삼각형 보간을 사용한다. 새 조형이나 임의의 스케일·스무딩을 추가하지 않았다.

`BODY_BreastSmall`은 현재 Small에 원 후보의 Small−SmallReference 잔차를 더한다. `BODY_BreastSmallReference`는 현재 모델의 기존 Small을 정확히 보존한다. `BODY_BreastExtended`는 원 후보의 추가 볼륨을 사용한다. Large와 기본 0~+1은 유지한다. 크기는 −1~+2, 간격은 .01이다. 기존 지방·근육 계산을 기준 형상으로 먼저 평가하고, 원 후보와 같은 warp/fold 잔차 합성과 자세 reference 정책을 적용한다. 현재 Basis·토폴로지·UV·본·frozen rest·웨이트·재질 및 가슴 외 모든 BODY 키는 유지한다.

## 별도 자세 보정 후보

원 이식본 `.breast7.2`는 극단 체형에서 팔을 내린 한 사례에 새 교차 1건이 있었다. 사용자 요청에 따라 중립 키를 그대로 둔 자세 보정 후보를 분리했다. 최대 지방·지방 볼륨, 가슴 −1, 간격 −1, 돌출 +1에서 겨드랑이 접촉이 재현됐다.

최종 규칙은 `standin-female-body-0.4.0.breast7.4`다. 새로 생긴 교차에 닿는 가슴 잔차 영역만 기준 자세 쪽으로 물린다. 실제 armature의 변위 Jacobian을 측정하며 팔·등·손·본은 이동하지 않는다. 0.1mm 이내의 작은 자세 보정, 면 뒤집힘 없음, 기준 대비 새 교차 없음이 모두 성립해야 적용한다. 원 중립 Small/Extended를 재조형하지 않는다. 정상 0~+1은 기존 경로를 그대로 사용한다.

초기 자세 후보 `.3`은 극단 18건을 통과했으나 추가 .01 자세 경로의 −0.99에서 수치 경계 문제가 발견됐다. `.4`는 이분 탐색 후 추가 이동을 하지 않고 검사에 통과한 float32 메시를 정확히 보존한다. 이전 후보와 실패 기록은 별도 폴더에 보존한다.

## 실제 자산과 미리보기

여성 11종의 새 FBX, sources, catalog, shape-bakes, characters, 중립 PNG, UV/skin, Linux/amd64 prepared neutral과 receipt를 같은 후보에 연결한다. `build_shape_release.py --retain-declared-controls`는 현재 카탈로그의 정확한 기본값과 slider grid를 보존한다. Blender FloatProperty에서 키 기본값이 0.000005cm 달라지는 반올림을 카탈로그에 옮기지 않는다. 다른 체형 슬라이더의 공개 범위는 현재 카탈로그에서 그대로 가져오고 가슴 최대만 +2로 바꾼다.

클라이언트 runtime은 105,430정점 전체를 계산한다. SSR1의 명시적 `indexDtype: uint32`와 필드별 `valueDtype: float32`를 사용한다. 기존 u16/int16 파일은 그대로 읽고, SSR2의 65,535정점 한도는 바꾸지 않는다. 기존 rule alias, source hash 우회, LOD runtime 승인 복사는 없다. 양자화된 정수 필드는 극단 체형 한 사례의 fold 분기를 바꿨으므로 새 후보 필드는 float32로 보존한다. 20,020,176바이트이며 기존 32MiB 한도 안이다. 새 클라이언트와 동일 source/runtime/catalog를 함께 검토해야 한다.

`package040.py`는 기본 회귀·극단 자세·독립 전체 메시 검사와 추가 301개 자세 경로가 모두 완료돼야 검수 묶음을 만든다. `body-models.review.json`, `shape-bakes.review.json`, `characters.review.json`은 이전 항목을 보존한 검토용 합본이다. 배포용 aggregate manifest나 기존 prepared 개수·파일/전체 용량 제한을 우회하는 활성화 파일이 아니다. 실제 활성화는 별도 검토한 body-assets 릴리스에서 기존 제한을 검증해야 한다.

새 UV는 실제 FBX 재로드에서 정점·삼각형 순서·웨이트가 동일한지 확인한 뒤 기존 UV 코너를 그대로 연결한다. 텍스처 파일은 현재 바이트를 유지한다. 기존 CSP 인증을 새 캐릭터 해시에 복사하지 않는다. default FBX의 대표 실제 CSP 확인과 다섯 skin의 새 자산 인증은 구분한다.

## 검증과 실행

수치와 최종 상태는 `FEMALE_BREAST_040_VALIDATION.json`에 기록한다. 기존 .01 중립 경로 301개, 체형 조합 150개, 기본 자세 52개, 극단 자세 18개, 실제 fixed-FBX export/reload 및 클라이언트 전체 메시 129개를 다시 검사한다. 추가 검사는 문제가 재현된 극단 체형의 팔 내림 자세를 −1~+2 전체 .01 경로 301개로 대조한다. inherited 교차는 별도로 남기며 전신에 교차가 전혀 없다는 뜻으로 보고하지 않는다.

최종 `.4`는 위 검사를 모두 통과했다. 추가 자세 경로에서 기준 대비 새 교차는 0건이고, 최대 보정은 가슴 잔차 소유 정점 13개의 0.065841mm다. 전체 메시 runtime의 최대 정점 오차는 0.000083769mm이며 0.05mm 한도를 유지했다. Linux/amd64의 11종 prepared 및 실제 Converter의 −1/0/+1/+2 변환도 완료했다. 클라이언트 Node 24 전체 1,710개 테스트와 서버 계약 66개가 통과했다.

격리 검수 앱에서 11종을 실제 미리보기하고 최종 source의 −1/+2 FBX를 HTTP 다운로드했다. 각각 CSP에 가져와 `.clip` 저장·닫기·재열기를 확인했다. 파일·source 해시와 실제 화면은 전용 산출물 폴더의 `evidence/final4-csp-acceptance.json`에 남겼다. 이는 default FBX 대표 확인이며 다섯 skin의 새 CSP 인증과 설치된 Tauri→운영 BFF 검증은 남아 있다. 작가 조형 검수와 별도 활성화 승인은 완료로 취급하지 않는다.

```sh
blender -b --disable-autoexec --python scripts/female_breast/port040.py -- \
  --pose-contact-guard --blend "$CURRENT_BLEND" --controller "$CURRENT_CONTROLLER" \
  --donor-blend "$BREAST7_BLEND" --donor-controller "$BREAST7_CONTROLLER" \
  --registry "$CURRENT_REGISTRY" --out "$NEW_CANDIDATE"
blender -b --disable-autoexec --python scripts/female_breast/verify.py -- \
  --candidate "$NEW_CANDIDATE" --parent-blend "$CURRENT_BLEND" --parent-controller "$CURRENT_CONTROLLER"
blender -b --disable-autoexec --python scripts/female_breast/pose_stress.py -- \
  --candidate "$NEW_CANDIDATE" --parent-blend "$CURRENT_BLEND" --parent-controller "$CURRENT_CONTROLLER"
blender -b --disable-autoexec --python scripts/female_breast/verify_contact_path040.py -- \
  --candidate "$NEW_CANDIDATE" --parent-blend "$CURRENT_BLEND" --parent-controller "$CURRENT_CONTROLLER"
blender -b --disable-autoexec --python scripts/build_shape_release.py -- \
  --retain-declared-controls --source "$NEW_CANDIDATE/model" --out "$NEW_CANDIDATE/release"
```

runtime은 companion 클라이언트의 `build_runtime.py --random 40 --dump ...`로 생성하고 `check_full_runtime.ts`의 실제 클라이언트 parser/evaluator로 독립 비교한다. 작은 fixture는 `build_breast040_fixture.py`로 재생성한다. 11종 prepared는 pinned Linux/amd64 Blender에서 생성하며 다른 플랫폼의 receipt를 승인 pin으로 쓰지 않는다.

격리 검수 화면은 실제 `mountShapeBody`와 동일 evaluator를 사용한다. `serve040.py`는 source/shape digest를 검증한 실제 컨트롤러의 fixed-FBX writer와 재로드 검사를 호출하고 HTTP 다운로드를 제공한다. localhost 검수 앱의 neutral 다운로드/CSP 확인과 설치된 Tauri 앱→운영 BFF의 전체 경로를 구분한다. 운영 앱 식별자·키체인은 사용하지 않는다.

## 경량 여성 v4와의 경계

이 후보의 `.breast7.4`는 가슴 포팅의 수정 번호이며 94,901정점 경량 여성 v4가 아니다. 경량 작업 채팅 `01a1f9f9-97da-7093-9e1c-a3e9df2f6d69` 및 서버 #244 / BFF #208 / 인프라 #95 / 클라이언트 #234의 자산·브랜치·승인 pin은 수정하지 않는다.

향후 경량 v4에 옮길 것은 세 가슴 키, 기존 reference를 먼저 평가하는 국소 잔차 합성, 필요 시 같은 자세 접촉 정책이다. 경량 소스의 정확한 해시와 대응을 새로 증명하고 새 rules/asset/controller 버전을 만들어야 한다. 현재 0.4.0의 토폴로지·UV·웨이트 인증이나 전체 메시 runtime을 경량 모델에 재사용하지 않는다. 새 11종 neutral/FBX/source/runtime/UV/skin/receipt, .01 경로·체형·자세·교차·실제 다운로드/CSP를 다시 검사한다. 경량 SSR2의 preset 0.5 / few 5 / basic 10 / all 20mm 및 32MiB 기준도 그대로 적용하고, 이 작업을 경량 runtime 승인으로 취급하지 않는다.
