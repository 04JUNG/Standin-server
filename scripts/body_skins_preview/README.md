# 교체형 인체 스킨 검수 초안 0.1.3

실제 남·여 FBX 표면에 기본 / 회색 가이드 / 살색 / 살색 가이드 / 근육선 강조의
5종을 교체하는 독립 검수 도구다. 제품 API, BFF, 검색, 앱 선택 UI, 운영 설정은 변경하지 않는다.

## 원본과 기본 재질

- 시작점: `origin/develop` `612be944d387c3dd2543ee2eed6ae95183c101bb`.
- `source_assets.json`에 남성 `male-base-v0-7-2`, 여성 `female-base-v0-3-4`의 SHA를 고정한다.
  최신 develop의 `docs/BODY_SELECTION/FINE_RECOGNITION_CRITERIA.md`와
  `ROUGH_PROJECTION_RESULTS.md`가 이 버전을 확정 체형 자산으로 명시한다.
  로컬 Converter 등록 release의 `characters.local.json`과 실제 바이트도 일치했다.
- 이것은 문서와 로컬 등록 자산의 확인이다. 운영 서비스의 현재 registry를 조회한 것은 아니다.
  추적 중인 `config/characters.example.json`의 기존 기본은 `standin-master-v2`이고,
  `config/body_presets.v2.json`은 이전 버전의 draft다. 최신처럼 보이는 폴더명으로 선택하지 않았다.
  로컬 `v3.3.1` 남성은 `standin-master-v2`와 SHA가 같았다.
- 기본 스킨은 `converter/preview_model.py`의 기존 PBR 값
  `baseColorFactor=[.72,.74,.76,1]`, roughness `.8`, metallic `0`을 유지한다.
  소스 FBX의 저작용 원래 재질도 `.blend` 안에 보존한다. 기존 제품 경로는 변경하지 않는다.

## 클튜 기준 가이드선 0.1.3

클튜 Ver.2 데생인형의 실제 T 포즈 정면을 직접 관찰하고, 사용자가 제공한 측면과 후면 화면으로
가이드선의 색/개수/연결을 확정했다. 자료 SHA와 명명한 배치는 `guide_spec.json`에 기록한다.
클튜 원본 메시/텍스처를 추출하지 않고 기존 Standin 표면에 선을 새로 작성한다.

- 주황: 머리·얼굴·목·몸통·골반의 중앙선, T 포즈 팔의 위/아래 선, 다리 앞/뒤 축에서 발까지.
- 회색: 머리 눈높이 둘레와 옆 세로선, 목 밑 둘레, 가슴/허리/골반 3개 단면,
  앞뒤 몸통의 세로선, T 포즈 팔 앞/뒤 선과 어깨·위팔·팔꿈치·아래팔·손목 5개 단면.
- 몸통 회색 세로선은 허벅지 시작 단면에서 주황색 다리 축으로 전환한다.
  다리의 회색 옆 세로선과 허벅지 시작/위 허벅지/아래 허벅지/위 종아리/아래 종아리/발목의
  6개 단면을 포함한다. 무릎 중심을 가로지르는 이전 단면은 제거했다.
- 주황 `#f39850` 전체 폭 2.6mm, 회색 `#707070` 전체 폭 2.0mm/농도 62%.
  화면의 조명과 원근 때문에 참고 이미지의 픽셀 RGB/폭과 항상 같은 것은 아니다.

두 원본에는 UV가 없다. 복사본에 `StandinSkinUV_v1`을 만들고 4096² RGB 거리 마스크로 굽는다.
전체 단면은 실제 3D 표면의 교차선으로 만들며, 남녀의 뼈 기준 위치로 단면 높이와 팔 길이를 맞춘다.
앞뒤 몸통/다리의 연결 곡선과 근육 경계는 실제 삼각형 표면으로 투영한 뒤 3D 거리로 폭을 계산한다.
화면 오버레이나 wireframe이 아니다. 기존 메시, 뼈, 가중치, 체형 비례는 바꾸지 않는다.

R=주황 가이드, G=회색 구조선, B=기존 주요 근육 경계다. 살색 `#FADFCA`와 근육선은 유지한다.
참고 인형과 Standin은 형상이 다르므로 동일한 선 배치와 연결을 Standin의 해부학적 위치에 대응시킨다.
숨은 부위나 전체 체형을 픽셀 단위로 동일하다고 주장하지 않는다.

## 리그와 검수 범위

UV 생성 전후 정점 위치·폴리곤 인덱스·모든 가중치·rest bone 행렬의 해시가 동일해야 빌드가 계속된다.
원본의 shape key는 둘 다 없다. 이후 원본 데이터나 체형 비례를 바꾸는 스킨 로직은 없다.

브라우저 포즈 계산은 4가중치 제한으로 원본과 달라질 수 있다. 검수용 GLB 10개는
Blender가 **전체 원본 가중치**로 계산한 실제 표면을 저장한다. 모델/자세 안에서는 모든 스킨이
동일 geometry를 공유한다. 별도 rigged GLB는 모든 influence를 내보내고 `.blend`도 리그를 유지한다.
검수용 정적 GLB를 동적 리그 자산으로 오인하면 안 된다.

- 남/여 기본형 각각 T 포즈, 기본, 팔 굽힘, 앉은 자세를 확인했다.
- 체형 변형은 `Spine1` local scale `[1.12,1,1.08]`의 진단 복사본 한 종류다.
  승인된 전체 body-template 또는 제품 shape-controller 검증을 뜻하지 않는다.
- 정면/45도/측면/후면 40장, T 포즈 8장, 자세·변형 12장, 확대 8장, 비교 2장 = 실제 WebGL 렌더 70장.
- 브라우저 오류 0, 마우스 회전/확대 및 스킨 전환 시 카메라 유지 검사 통과.
- 10개 FBX의 Blender 재임포트: 정점/폴리곤/52개 뼈 보존, 가중치 최대 오차 0,
  정점 최대 오차 약 0.00032mm, bone 행렬 최대 오차 6.24e-6.
- UV overlap은 픽셀 래스터 진단도 남긴다. 모든 seam의 무결성·모든 체형/포즈를 인증한 결과는 아니다.
- **내보낸 스킨의 CSP 가져오기는 미검증.** 클튜 기본 인형의 선 배치 관찰과, 생성 FBX/PNG의 Blender 왕복을 구분한다.
  동적 포즈에 대한 제품 Converter 전체 보정, CSP에서의 보이기/재질·텍스처 연결은 후속 검증 대상이다.

## 재현

Blender 5.2.0 LTS (`fbe6228777e7`), NumPy 2.3.5, Pillow 12.3.0,
Three.js 0.186.1, Chrome + Playwright 1.62.1에서 검수했다.
제품 frozen Converter의 실행 버전을 바꾸거나 동일하다고 가정하지 않는다.
원본 모델·대용량 생성물·vendor 라이브러리는 Git에 넣지 않는다.

```bash
python -m pip install -r scripts/body_skins_preview/requirements.txt
npm install --prefix scripts/body_skins_preview
python scripts/body_skins_preview/build.py \
  --source-root /path/to/local/Standin-server \
  --output /path/to/skin-preview \
  --three-root scripts/body_skins_preview/node_modules/three \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
python3 -m http.server 8791 --bind 127.0.0.1 --directory /path/to/skin-preview
# 다른 터미널에서:
node scripts/body_skins_preview/capture_review.cjs http://127.0.0.1:8791 /path/to/skin-preview
python scripts/body_skins_preview/make_sheets.py /path/to/skin-preview
python scripts/body_skins_preview/validate_assets.py /path/to/skin-preview
```

`make_sheets.py`는 macOS의 Apple SD Gothic Neo 폰트를 사용한다. 다른 OS에서는 해당 폰트 경로를 바꾼다.
미리보기는 GLB/PNG를 HTTP로 읽으므로 `file://`로 열지 않는다. 네트워크 CDN 없이 로컬 vendor를 사용한다.
HTML의 5종 선택, 남/여, 검수 자세, 4시점, 드래그 회전/스크롤 확대, 상체 확대, 나란히 비교를 사용할 수 있다.

## 검증 명령

```bash
python tests/test_smoke.py                 # 54/54 passed (기존 서버 가상환경)
python -m compileall -q src api scripts    # passed
node --check scripts/body_skins_preview/viewer.js
python scripts/body_skins_preview/validate_assets.py /path/to/skin-preview
```

`manifest.json`은 skinId/버전/재질 파라미터/소스 SHA/생성 자산 상대경로·SHA/검증 범위를 기록한다.
PNG는 일반 sRGB base-color 이미지다. FBX는 neutral rig와 이 텍스처를 함께 내보낸 별도 파일이다.
실시간 파라미터 조정은 뷰어용이며 기본 PNG/FBX를 자동으로 다시 굽지는 않는다.

## 피부색 기준 0.1.2 유지

`skin_style.json`에 참고 이미지에서 추출한 RGB `[250,223,202]` (`#FADFCA`)와 가이드 파라미터를 저장한다.
살색 두 재질에만 프리뷰 밝기 보정 0.30을 적용해 이전의 어두운 갈색 표현을 교정한다.
가이드선에도 같은 색 보정을 적용하며 재질별 shader cache를 분리한다. 일반 PNG/FBX는 이 기준색의 albedo를 포함하며,
프리뷰의 밝기 보정은 뷰어 전용이다. 다른 프로그램의 조명에 따른 외관까지 동일하다고 주장하지 않는다.
