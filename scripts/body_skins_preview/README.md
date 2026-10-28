# 교체형 인체 스킨 검수 초안 0.1.2

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

## 표면에 붙는 방식

두 원본에는 UV가 없다. 복사본에 `StandinSkinUV_v1`을 만들고,
기본 자세의 해부학적 위치에 작성한 선을 4096² UV 거리 마스크로 굽는다.
중심선·눈높이·몸통 단면·사지 축은 기본 자세의 신체 기준을 따른다.
주황 몸통 중심선은 머리부터 골반 아래 표면 끝까지 이어지고, 회색 상체 세로선과 주황 하체 축선은 같은 3D 곡선을 공유하며 골반에서 색이 바뀐다.
회색 선을 다리까지 별도로 연장하지 않는다. 주황색은 `#ff8a35`,
기본 전체 폭 4.4mm/농도 100%로 얇은 회색 선과 분명히 구분한다.
주황 축선은 앞/뒤 표면에 각각 투영해 어깨·손에서 폭이 불어나지 않게 한다.
근육선은 명명한 쇄골, 대흉근, 삼각근, 복직근, 복사근, 대퇴, 종아리 등의
곡선을 앞/뒤 실제 삼각형 표면으로 ray projection한 뒤 **3D 거리**로 폭을 계산한다.
가려진 부위로 곡선이 크게 점프하면 연결하지 않는다. 화면 좌표 오버레이나 wireframe이 아니다.

R=주황 중심선, G=회색 구조선, B=주요 근육 경계다. 바탕색은 마스크와 분리된다.
근육선 초안의 낮은 채도 점토색 `#d3c3b1`은 음영과 짙은 갈색 선을 구분하기 위한 선택이다.
혈관·피부 결·의학용 세부 해부도는 포함하지 않는다. 작화용 주요 경계의 초안이며 작가 검수가 필요하다.

브라우저는 UV 거리 필드를 Standard PBR 재질에서 합성한다. 바탕색·선 굵기·농도를 바로 조절한다.
선 색·기본 폭·농도는 `manifest.py`, `viewer.js`, `generate_textures.py`의 파라미터에 명시된다.
미세 선의 거리 필드가 mipmap 평균에서 사라지지 않도록 선형 샘플링과 화면 미분 안티앨리어싱을 사용한다.

## 리그와 검수 범위

UV 생성 전후 정점 위치·폴리곤 인덱스·모든 가중치·rest bone 행렬의 해시가 동일해야 빌드가 계속된다.
원본의 shape key는 둘 다 없다. 이후 원본 데이터나 체형 비례를 바꾸는 스킨 로직은 없다.

브라우저 포즈 계산은 4가중치 제한으로 원본과 달라질 수 있다. 검수용 GLB 8개는
Blender가 **전체 원본 가중치**로 계산한 실제 표면을 저장한다. 모델/자세 안에서는 모든 스킨이
동일 geometry를 공유한다. 별도 rigged GLB는 모든 influence를 내보내고 `.blend`도 리그를 유지한다.
검수용 정적 GLB를 동적 리그 자산으로 오인하면 안 된다.

- 남/여 기본형 각각 기본, 팔 굽힘, 앉은 자세를 확인했다.
- 체형 변형은 `Spine1` local scale `[1.12,1,1.08]`의 진단 복사본 한 종류다.
  승인된 전체 body-template 또는 제품 shape-controller 검증을 뜻하지 않는다.
- 정면/45도/측면/후면 40장, 자세·변형 12장, 확대 8장, 비교 2장 = 실제 WebGL 렌더 62장.
- 브라우저 오류 0, 마우스 회전/확대 및 스킨 전환 시 카메라 유지 검사 통과.
- 10개 FBX의 Blender 재임포트: 정점/폴리곤/52개 뼈 보존, 가중치 최대 오차 0,
  정점 최대 오차 약 0.00032mm, bone 행렬 최대 오차 6.24e-6.
- UV overlap은 픽셀 래스터 진단도 남긴다. 모든 seam의 무결성·모든 체형/포즈를 인증한 결과는 아니다.
- **CSP 미검증.** FBX/PNG 파일 생성 및 Blender 왕복까지만 확인했다.
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

## 피부색 기준 0.1.2

`skin_style.json`에 참고 이미지에서 추출한 RGB `[250,223,202]` (`#FADFCA`)와 공유 접합 곡선을 저장한다.
살색 두 재질에만 프리뷰 밝기 보정 0.30을 적용해 이전의 어두운 갈색 표현을 교정한다.
가이드선에도 같은 색 보정을 적용하며 재질별 shader cache를 분리한다. 일반 PNG/FBX는 이 기준색의 albedo를 포함하며,
프리뷰의 밝기 보정은 뷰어 전용이다. 다른 프로그램의 조명에 따른 외관까지 동일하다고 주장하지 않는다.
