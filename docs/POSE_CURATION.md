# 포즈 수집·생성·검수

현재 검수 기준과 통합 실행 명령은 [검수 기준·자동화](POSE_REVIEW_STANDARD.md)를 따른다.
아래 날짜별 결과는 과거 작업 기록이며 최신 판정 기준을 덮어쓰지 않는다.

`pose_curation`은 운영 라이브러리의 로컬 스냅샷을 기준으로 신규 후보를 만들고 검수하는 도구다.
추론 서버와 분리되어 있어 VLM 키나 ONNX 모델 없이 실행한다. `build → hands → render → audit → serve(검수) → publish`가 기본 흐름이다.
`publish`는 로컬 검색 DB를 만드는 명령이며 AWS 업로드나 운영 배포를 하지 않는다.

## 준비

Python 3.12 환경에서:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-curation.txt
```

기존 S3 라이브러리를 `data/` 아래에 준비한다. 필요한 파일은 `poses.db`, `bvh/`, `thumbs/`다.
원본 BVH, DB, 캐릭터와 렌더 결과는 모두 Git에서 제외되는 `data/`에 둔다.

캐릭터 미리보기에는 운영 캐릭터 레지스트리의 `standin-master-v2.fbx`와 **Blender 5.2.0**이 필요하다.
`config/characters.example.json`의 캐릭터 SHA-256을 렌더 시작 시 검증한다.
Blender는 [공식 배포처](https://download.blender.org/release/Blender5.2/)의 포터블을 사용할 수 있다.
배포처의 `.sha256`과 다운로드한 아카이브를 대조한 뒤 압축을 푼다.

현재 작업 환경의 경로:

```text
data/tools/blender-5.2.0-windows-x64/blender.exe
data/curation/characters/standin-master-v2.fbx
```

## 실행

```powershell
# 1. 공식 카탈로그에서 필요한 BVH만 수집하고 1프레임 후보 생성
.venv/Scripts/python.exe -m pose_curation build --batch 100style-pilot-20261001

# 2. 양손에 손가락 30개 관절과 기본 손 모양 추가
.venv/Scripts/python.exe -m pose_curation hands `
  --batch-dir data/curation/batches/100style-pilot-20261001

# 3. 기존 라이브러리와 같은 캐릭터로 4방향 JPEG 생성
.venv/Scripts/python.exe -m pose_curation render `
  --batch-dir data/curation/batches/100style-pilot-20261001 `
  --blender data/tools/blender-5.2.0-windows-x64/blender.exe `
  --character data/curation/characters/standin-master-v2.fbx `
  --workers 2

# 4. 전체 기존/신규 포즈를 조회하고 검수
.venv/Scripts/python.exe -m pose_curation serve --port 8765
```

브라우저: <http://127.0.0.1:8765/>. 서버는 로컬 루프백에 바인딩한다.

포즈 상세의 **4방향 미리보기 → 전신 / 반신 / 흉상 / 두상**에서 신체 범위를
선택한다. 전신은 기존 썸네일을 사용하고, 부분 범위는 `converter/framing.py`와
동일한 메시 절단을 수행한 FBX를 다시 읽어 512px 정면·45도·측면·후면을 생성한다.
반신은 팔과 손을 포함한 상체, 흉상은 팔을 제외한 가슴·어깨·목·머리, 두상은
머리를 남긴다. 부분 미리보기에는 소품 도형을 넣지 않는다.
BVH 내려받기와 3D 골격은 전신을 유지한다. 부분 출력으로 전신 검수를 대체하지 않는다.

최초 선택 시 생성 상태를 표시하며 실패하면 재시도할 수 있다. Blender 작업은
별도 프로세스에서 한 번에 하나씩, 최대 8개 대기로 실행한다. 캐시는
`data/curation/framed-previews/`에 BVH·캐릭터·변환/렌더 코드·범위별로 저장된다.
네 이미지가 모두 검증된 경우만 제공하며 포즈가 바뀌면 이전 캐시를 사용하지 않는다.
서버 코드를 수정한 뒤에는 검수 서버를 재시작해야 한다.

### 자유 각도와 출력 파일

포즈 상세에서 범위를 선택한 뒤 **자유 각도·내보내기**를 연다. 좌우 시점
(-180~180°), 높낮이(-90~90°), 화면 기울기(-180~180°)를 슬라이더나 숫자로
입력하고 **이 각도로 미리보기·파일 생성**을 누른다. 실제 FBX를 다시 읽은
미리보기가 로드되면 같은 결과의 FBX/각도 JSON 링크가 활성화된다.
각도·범위·포즈가 바뀌면 이전 다운로드 링크는 즉시 사라진다.
러프 겹쳐 보기는 브라우저의 로컬 이미지로 비교하며 서버로 전송하지 않는다.
상체 대표 라이브러리와 기존 러프 자동 시점 비교는 `/scoped`에서 제공한다. 아래 설명을 참고한다.

`pose_curation/orientation.py`의 `front-baked-yup-v1`은 BVH Y-up 좌표에서
`Rz(-roll) @ Rx(pitch) @ Ry(-yaw)`를 사용한다. yaw +는 오른쪽 카메라 시점,
pitch +는 내려다보기, roll +는 화면 시계 방향이다. 루트 위치를 피벗으로
모델 전체만 회전한다. 현재 각도 출력은 FBX만 제공하며 BVH 다운로드는 원본 자세다.
Blender에는 Y-up→Z-up 기저변환으로 같은 회전을 전달한다. 변환기는 기존
원본을 리타게팅한 다음 회전을 적용하므로 재정렬 과정에서 방향이 사라지지 않는다.
부분 FBX의 뼈대는 전신을 유지하며 BVH도 항상 전신이다.

이 기능은 고정 정면·정사영에서 보이는 방향을 모델에 반영한다. 카메라·원근·
확대·화면 이동을 파일로 전달하지 않는다. CSP의 기존 레이어 카메라는 변경되지
않으므로 정면·평행 투영에서 먼저 비교해야 한다. 모델 자체가 바닥에 대해
기울 수 있다. CSP에서 확인된 범위는 아래 사용자 검증 기록을 따른다.

검수 API `POST /api/poses/{key}/oriented`는 `scope, yaw, pitch, roll, content_hash`를
받고 `GET`으로 상태를 조회한다. 준비된 버전의 `/oriented/{preview|fbx|settings}`만
다운로드하며 BVH 해시·범위·각도·코드 버전과 3개 출력 해시가 일치해야 한다.
Blender 작업은 기존 검수 대기열을 공유한다. 기본 캐릭터 FBX를 재사용하므로
새 각도마다 본체 리타게팅을 반복하지 않는다.

재현 및 CSP 전달용 파일 생성:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_pose_orientation.py tests/test_pose_review_framing.py tests/test_pose_curation.py -q
.venv/Scripts/python.exe scripts/verify_review_orientation.py
```

기본 출력 `data/dev/csp-fbx-orientation-20261002.zip`에 7가지 각도·범위 조합의
FBX/JPEG/JSON, 비교용 `index.html`, 한국어 `README.txt`를 포함한다.
원본 파일과 검수 결정은 변경하지 않는다.

**CSP 사용자 검증 — 2026-10-02**

- 사용자가 반신·흉상·전신 FBX를 CSP에 가져와 의도한 방향으로 잘 표시됨을
  보고했고, 각 범위의 화면 3장을 제공했다. 이 세 범위의 가져오기·방향·부분
  메시 표시를 사용자 확인으로 기록한다.
- 두상, 모든 시험 각도 조합, 편집 가능성, 재질·조명 일치까지 검증한 것으로
  확대하지 않는다. 정확한 CSP 버전과 사용한 카메라 설정은 확인되지 않았다.
- 이후 사용자가 BVH도 시험했고 조정한 각도가 반영되지 않는다고 화면과 함께 보고했다.
  사용자 요청에 따라 각도 출력은 FBX만 제공한다. BVH는 원본 전신 자세 다운로드로 유지한다.
  CSP 가져오기 과정의 정확한 원인과 버전별 차이는 확인하지 않았다.
- 자동 생성된 시험 ZIP의 `verification.json`은 생성 당시 Blender 검증 기록이다.
  후속 사용자 확인은 별도 `data/dev/csp-orientation-user-verification-20261002.json`에
  보관하며, 전체 CSP 검증 완료로 일괄 표시하지 않는다.

렌더링과 검수 서버는 함께 실행해도 된다. 완료된 포즈는 바로 반영하며, 목록을 보는 동안 진행 상황을 갱신한다.

### 반신·흉상 대표 라이브러리와 러프 시점 비교 (2026-10-02)

검수 서버 `/scoped`에서 반신 대표 256개, 흉상 대표 96개를 탐색하고 저장된
러프의 상체 관절에 맞는 포즈·시점을 검색한다. 메인 목록의 **대표 라이브러리**
필터도 같은 색인을 사용한다. 목록 카드는 원본 전신이며 상세 화면을 열면
선택한 범위의 실제 FBX 메시 절단 미리보기를 생성한다.

색인의 원본은 1,568개(기존 1,199 + 현재 QA 통과 신규 369)다. 제외 169개,
대기 146개를 넣지 않았으며 기존 자료를 새로 QA 통과한 것으로 표시하지 않는다.
BVH 파일을 복제·절단하지 않고 원본 키와 해시로 대표를 참조한다. 신규 자료에는
기존의 리비전별 자동·시각 검수 조건을 그대로 적용한다. 포즈·검수 기록이 바뀌면
오래된 대표 색인은 409로 거부하고 명시적으로 재생성해야 한다.

`scoped/geometry.py`는 골반의 rest 좌표계와 root 회전을 이용해 전역 방향을
제거한 후 몸통·어깨·머리 위치, 실제 머리 회전축, 반신일 때 팔 위치를 비교한다.
다리 자세와 손가락·소품 차이는 군집 피처에서 제외한다. 흉상에서는 팔도 제외한다.
결정적인 farthest-first 방식으로 실제 자료를 대표로 선택한다. 256/96개는 초기
개수 상한이며 모든 원본과 완전히 같은 형상을 보장하지 않는다. 현재 피처 RMS의
최대 대표 거리는 반신 0.299, 흉상 0.177로 목표 반경(0.13/0.10)을 충족하지
못했다. `manifest.json`에 배정 관계·거리·미충족 여부를 기록하고 원본 전체는 유지한다.

`scoped/matching.py`는 얼굴의 가상 Head 좌표와 숨은 골반·다리를 쓰지 않는다.
실제 RTMPose 어깨·팔 관절 중 점수 0.45 이상, 이미지 경계 안의 양 어깨와
최소 한쪽의 어깨·팔꿈치·손목 전체 관측을 요구한다. 작거나 어깨가 붕괴한 투영은 거부한다.
초기 0.3/양쪽 상완 허용 기준은 잘린 흉상 팔을 추측한 검출도 허용하는 문제가 있어 폐기했다. yaw/pitch 격자에서 탐색한 뒤 연속 최적화하고,
화면 roll·양의 스케일·이동은 관측 관절에 맞춘다. 공유 `rotation_matrix()`를 써서
검색·FBX 출력의 회전 부호와 순서를 일치시킨다. 관절 회전을 보정하지 않는다.

반신 대표 검색의 정규화 오차가 0.05보다 크면 1,568개 전체로 다시 검색한다.
상위 5개 후보와 추정 각도, 관절 겹침 비교를 제공한다. 선택 시 같은 원본 해시와
각도를 상세 화면에 넘기고, 사용자가 생성한 실제 FBX 미리보기를 확인한 뒤
**이 각도 FBX**를 내려받는다. 2D 스케일·이동·카메라·원근은 FBX에 전달하지 않는다.
BVH는 각도 출력을 제공하지 않으며 기존 **원본 BVH**만 유지한다.

현재는 사용자가 제공하거나 로컬 검수 사용을 승인한 439장과 기존 실제 추출 결과를
사용한다. 강화된 기준에서 비교 가능한 관측은 216건이다. 기존 기하 분류상
상체만 관측된 31건은 모두 강화 기준을 충족하지 못하므로 자동 후보를 제공하지 않는다.
화면은 기본적으로 팔이 충분히 관측된 전체 러프의 상체 비교 목록을 연다.
원본 경로·사용자/설치 ID를 UI에 노출하거나 외부로 전송하지 않는다. 이미지 해시가
추출 결과와 다르면 거부한다. 다인 러프의 관절 소유권은 다시 추론하지 않으므로
사용자가 화면에서 인물 선택·관절 위치를 확인해야 한다. 새 파일 즉석 추출은 미구현이다.

정량 검증은 강화 기준을 만족한 관측에서 순서상 균등하게 선택한 48건으로 수행한다.
평가 파일에는 같은 관측 마스크로 계산한 고정 4시점 전체 검색, 대표+연속 시점,
전체 연속 시점의 오차와 시간을 기록한다. 이전의 잘린 팔 허용 기준으로 구한 31건의
수치는 현행 평가로 사용하지 않는다. 이 측정은 동일 캐시의 기하 적합도이며, 3D 정답·
시각적 정확도·사용자 성공률 평가가 아니다. 원본 중복은 독립 표본으로 해석하지 않는다.
현행 48건의 오차 중앙값은 고정 4방향 0.1206, 대표+시점 0.0466, 전체 연속 시점 0.0464였다.
전체 재검색은 37건, 전체 연속 검색보다 0.02 이상 오차가 큰 사례는 2건이다. 대표 검색만으로 정확도가 보장되지는 않는다.
검출된 팔이 부족한 흉상·두상은 상체 관절만으로 자동 방향을 추정하지 않는다.
대표 탐색과 수동 각도 FBX 출력은 사용할 수 있다. 얼굴 기준점을 직접 지정하는
두상·흉상 작업실은 아래 별도 절차를 따른다. 운영 검색/refine 정책은 바꾸지 않았다.

### 두상·흉상 시점 작업실 (2026-10-02)

`/head`는 기존 BVH 포즈를 검색하지 않고 Standin Master V2의 기본 자세에서
두상 또는 흉상 메시를 만든다. 중립 목 자세를 기준으로 모델 전체를 회전하므로
기존 포즈의 목 회전을 중복 적용하지 않는다. 흉상의 어깨·표정은 기본 상태이며
러프의 어깨 자세를 추정한 결과로 표시하지 않는다. 메시 범위만 자르고 전신
뼈대는 유지한다. 출력은 FBX와 방향 설정 JSON이며 BVH 출력은 제공하지 않는다.

사용자는 저장된 러프에서 한 얼굴의 양쪽 눈 바깥꼬리, 미간, 코끝, 턱끝,
이마 중앙 위끝을 지정할 수 있다. `head/fitting.py`는 SHA로 고정한 MediaPipe
canonical face의 대응 6개 점을 공통 `orientation.rotation_matrix()`로 투영해
yaw/pitch를 맞추고 roll/양의 스케일/이동을 구한다. 얼굴 비율·원근·표정을
복원하지 않으며, 결과는 `source=user_landmarks`, `confidence=unvalidated`,
`requires_visual_review=true`인 참고 후보다. VLM이나 몸의 Head 관절로 얼굴
좌표를 만들지 않는다. 기준점·추천 기록 JSON에 입력 이미지 해시와 점을 남긴다.

최소 눈 사이 12px, 이마–턱 30px, 전체 점 분포 40px가 필요하다. 유한 좌표·
이미지 경계·눈 순서·눈축/얼굴축의 퇴화를 검사한다. 정규화 투영 오차 0.2 초과,
|yaw|>65°, |pitch|>55°는 추천하지 않는다. 이 수치는 거부용 초기 휴리스틱이며
확률이나 검증된 정확도가 아니다. 가림·옆얼굴·뒤통수는 직접 각도를 조절한다.
초록색 사용자 점과 주황색 기준 얼굴 투영, 실제 FBX의 재가져오기 미리보기,
얼굴 영역 겹침으로 검수한다. 겹침의 위치·크기·불투명도는 출력에 반영하지 않는다.

얼굴이 없는 컷은 **두상 없는 이미지 제외**로 목록에서 제외하고, **제외한 이미지**
목록에서 복구할 수 있다. `head-direction/query-reviews.json`의 이미지 해시별
기록이며 원본 파일·포즈 라이브러리·상체 검색에는 영향을 주지 않는다. 동일한
이미지의 중복 입력에도 제외가 적용된다. 제외된 이미지는 기준점 fit API에서도
거부한다. 사용자가 지적한 음식·손 컷 `rough_001`을 이 방식으로 제외했다.

**초기 전체 이미지 검출 실험 (아래 확대 검출 단계 이전)**

- 기존 승인된 로컬 이미지 439장(제공 러프·사용자 입력, 중복 포함)에
  MediaPipe Face Landmarker 0.10.21을 CPU/IMAGE 모드로 실행했다. 검출/존재
  임계값 0.7, 최대 8개 얼굴, 최대 변 1600px, 원본과 좌우 반전 두 번이다.
- 검출된 이미지는 2장, 얼굴은 2개였다. 시각 확인 결과 `rough_114`,
  `rough_115`에 붙은 **참고 인물 사진**이었다. 그 컷의 웹툰 얼굴은 검출하지
  못했다. 2건 중 1건이 기하·반전 검사까지 통과했지만, 목표 인물 소유권을
  확인하지 않은 수치 검사만으로 올바른 결과를 승인할 수 없음을 확인했다.
- 이 단계에서는 자동 얼굴 추천을 연결하지 않았다. 이후 그림용 영역 검출과
  사용자 대상 선택을 추가한 현재 동작은 다음 절의 확대 검출 방식을 따른다.
  이 결과를 439장에 대한 그림 얼굴 정확도나 모든 얼굴 모델의 성능으로
  일반화하지 않는다.
- 모델·랜드마크는 프로젝트 `data/curation/head-direction`에만 저장한다.
  사용자 이미지 업로드나 외부 추론 API 호출은 없다. 서버는 MediaPipe를 import하지
  않으며 기존 포즈 런타임의 NumPy를 변경하지 않았다.

근거: [공식 Face Landmarker 안내](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker),
[Python API](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python),
[Face Mesh V2 모델 카드](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf),
[canonical face 원본](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj).
모델 카드는 실사 카메라 자료를 설명하며 웹툰 러프 적합성을 보장하지 않는다.
모델과 canonical mesh는 Apache-2.0 출처다.

선택적 얼굴 검출 실험 환경 재현(서비스 실행에는 불필요):

```powershell
.venv/Scripts/python.exe -m venv data/tools/face-runtime
data/tools/face-runtime/Scripts/python.exe -m pip install -r requirements-head-direction.txt
New-Item -ItemType Directory -Force data/curation/head-direction/models | Out-Null
Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task' -OutFile data/curation/head-direction/models/face_landmarker.task
Invoke-WebRequest 'https://raw.githubusercontent.com/google-ai-edge/mediapipe/refs/heads/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj' -OutFile data/curation/head-direction/models/canonical_face_model.obj
data/tools/face-runtime/Scripts/python.exe -m pose_curation.head.extract
```

수동 기준점 추천에는 canonical mesh만 필요하다. task 모델 SHA256은
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`,
canonical mesh는 `8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618`이다.
바뀐 모델은 조용히 재사용하지 않고 거부한다. 기본 캐릭터·Blender는 기존 검수
설치와 같다. 별도 런타임 의존성은 `requirements-head-direction.txt`로 분리했다.

검증: 네 조합의 알려진 투영으로 각도 복구, 퇴화·범위 밖 점 거부, 좌우 반전 회전,
원본 해시 변경·같은 출처 요청·FBX만 허용·제외/복구 지속성 테스트를 수행한다.
실제 기본 두상 (0,0,0), (30,20,-10), 흉상 (-35,-15,12) FBX를 생성하고
Blender에서 재가져와 전체 뼈대 좌표를 검사했다. 이는 CSP 두상 가져오기나
수동 기준점 추천의 그림 정확도를 추가로 확인한 것과는 구별한다.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_pose_head.py tests/test_pose_orientation.py tests/test_pose_review_framing.py tests/test_pose_curation.py tests/test_pose_scoped.py -q
.venv/Scripts/python.exe scripts/verify_review_head.py
```

검증용 `data/dev/head-direction-20261002.zip`에는 위 3개 조합의 FBX/JPG/JSON,
뼈대 검증 결과와 CSP 비교 안내를 넣는다. 특정 러프를 재현한 파일로 표시하지 않는다.

### 얼굴 확대 검출·선택형 자동 각도 (2026-10-02 후속)

현재 `/head`의 기본 목록은 **자동 각도 후보 있는 러프**다. 원하는 얼굴의
**얼굴 각도 사용**을 누르면 실제 검출 기준점과 각도가 적용된다. 이후
**이 각도로 미리보기·파일 생성**으로 중립 두상/흉상 FBX를 만든다. 후보를
자동 선택하지 않으며, 모든 이미지가 처리되는 기능은 아니다. 후보가 없는
러프는 전체 검수 대상 목록에서 기준점 또는 각도를 직접 지정한다.
`/api/head`의 `automatic_enabled`는 현재 유효한 추천 캐시가 있을 때만 true다.

관측·기하 검사·화면을 분리했다.

- `head/regions.py`, `crop_experiment.py`: 기존 RTMPose 관측은 얼굴을 찾을
  **확대 영역**에만 사용한다. 몸 관절을 얼굴 기준점으로 대체하지 않는다.
- `head/anime_experiment.py`: 그림용 검출기의 얼굴 상자를 확대한 뒤, 그
  실제 이미지에서 MediaPipe 478개 점을 독립 검출한다. Anime 모델의 28개 점은
  감사 기록에만 보관하며, 다른 토폴로지의 canonical face에 억지로 대응시키지 않는다.
- `head/fitting.py`, `candidates.py`: 눈·코·이마·턱·볼 12개 관측점으로 공통
  회전 규약의 참고 각도를 구하고, 두 경로의 같은 얼굴을 IoU > 0.5로 합친다.
- `review/head_routes.py`, `static/head-candidates.js`: 추론 없이 로컬 후보만
  읽는다. 사용자가 대상 얼굴을 선택한 뒤에만 `/api/head/suggest`가 각도를
  반환한다. 이 요청만으로 렌더링·내보내기를 시작하지 않는다.

자동 후보의 거부 조건은 이미지 밖/최소 변 32px 미만 얼굴, 점 분포 40px 미만,
정규화 투영 오차 0.15 초과, |yaw| > 65°, |pitch| > 55°다. 좌우 반전 이미지에서
같은 얼굴이 유일하게 검출되고 반전 보정 후 회전 차이 ≤ 12°여야 한다.
여러 확대 영역의 유효한 추정끼리 차이가 15°를 넘으면 수동 조정으로 남긴다.
모두 초기 거부 휴리스틱이며 정확도·확률·3D 정답 검증을 의미하지 않는다.

438장 전체(얼굴 없는 음식 컷 1장 제외)를 처리한 결과:

| 단계 | 이미지 수 | 얼굴 관측/후보 수 |
|---|---:|---:|
| 몸 관측 확대 영역에서 얼굴 검출 | 51 | 63 |
| 그림 얼굴 상자 확대 후 얼굴 검출 | 28 | 29 |
| 중복 얼굴 통합 | 54 | 69 |
| 기하·반전 검사 및 참고 사진 제외 후 추천 | **32** | **33** |

그림용 검출기는 159장에서 169개 상자를 제안했지만, 독립 얼굴 기준점 검출까지
성공한 것은 위 표의 28장이다. 몸 관측 기반 제안은 267장 376영역이다.
초기 전체 이미지 검출의 2장/2얼굴과 같은 지표로 정확도를 비교하지 않는다.
제공 러프와 실제 사용자 입력에는 시각적으로 중복된 그림이 있으며 독립 평가
세트가 아니다. 정답 각도 라벨이 없으므로 정밀도·재현율·각도 정확도는 미측정이다.

추천 가능한 33개 얼굴의 확대 이미지를 시각적으로 확인했다. 모두 그림 얼굴이며,
별도로 발견한 참고 사진 **15개 원시 관측**은 `source-face-reviews.json`에서
관측 파일 SHA와 얼굴 인덱스에 묶어 제외했다. 사진이 있는 이미지 전체를
제외하지 않으므로 `rough_114/115`의 그림 얼굴은 계속 사용할 수 있다. 이 기록은
실행 시 사진을 자동 분류하는 모델이 아니다. 새 자료에서도 사용자가 대상
얼굴을 확인해야 한다. 작은 얼굴·심한 측면·가림·뒤통수는 수동 조정이 필요하다.

`candidates.json`은 입력 이미지, 원시 관측, 코드, 참고 사진 검수 기록의 SHA에
묶인다. 변경된 원시 관측·제외 이미지·오래된 코드의 후보 선택은 409로 거부한다.
같은 출처 요청, `X-Pose-Review: 1`, 명시적 `target_confirmed=true`가 필요하다.
추천 기록은 `source=detected_face_user_selected`, `confidence=unvalidated`,
`requires_visual_review=true`이며 라이브러리의 채택 판정을 대신하지 않는다.
화면에서 러프·각도를 바꾸면 이전 추천과 다운로드를 무효화한다.

실제 추천 5건(`rough_026/084/114/279/313`)은 두상 FBX 생성·재가져오기 후
52개 뼈대 좌표를 검사했다. 최대 위치 차이는 4.77e-7 Blender 단위 이하였다.
원본 얼굴과 출력 미리보기를 비교해 좌우 방향·기울기·내려다봄을 확인했다.
얼굴 모양·표정·원근의 일치나 CSP에서 이 5개 파일의 가져오기까지 보증하지 않는다.
관련 테스트 74개가 통과했다. CSP에서 시험할 수 있는
`data/dev/head-candidates-20261002.zip`에는 5개 FBX/JPG/각도 JSON과 안내만 담고,
원본 러프·사용자 입력 이미지는 넣지 않았다.
기록: `data/dev/head-candidate-verification/verification.json`, `comparison.jpg`,
`data/curation/head-direction/evaluation-v2.json`. UI에서는 사진 후보 비활성화,
선택한 그림의 각도 적용, 출력 파일 생성, 이전 다운로드 무효화를 확인한다.

**격리된 추론 환경과 재현**

서비스 `.venv`와 얼굴 추론용 Python 3.12 환경을 분리한다. 기본 MediaPipe 환경을
위 절대로 준비한 뒤 아래 CPU 패키지와 고정 모델을 추가한다. 모델은 공개 원본에서
명시적으로 내려받으며 입력 이미지 전송은 없다. 추론에서는 HF 오프라인 모드를
켜고 아래 로컬 파일만 읽는다. API 서버는 Torch/MediaPipe를 import하지 않는다.

```powershell
data/tools/face-runtime/Scripts/python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
data/tools/face-runtime/Scripts/python.exe -m pip install -r requirements-anime-head.txt
Invoke-WebRequest 'https://huggingface.co/hysts/anime-face-detector-yolov3/resolve/afdd4226a79ae8bb81f334dbcffd34f8cc000c38/model.safetensors' -OutFile data/curation/head-direction/models/anime-yolov3.safetensors
Invoke-WebRequest 'https://huggingface.co/hysts/anime-face-detector-hrnetv2/resolve/9b3435248b26aeb82e2a8578fe9d86d5d57158af/model.safetensors' -OutFile data/curation/head-direction/models/anime-hrnetv2.safetensors
data/tools/face-runtime/Scripts/python.exe -m pose_curation.head.crop_experiment
data/tools/face-runtime/Scripts/python.exe -m pose_curation.head.anime_experiment
.venv/Scripts/python.exe -m pose_curation.head.candidates
.venv/Scripts/python.exe scripts/verify_head_candidates.py
```

두 가중치 SHA는 `anime_experiment.WEIGHT_HASHES`에 고정되어 다르면 실행을 거부한다.
캐시는 이미지·모델·설정이 같을 때 재사용한다. 파일을 수정하는 수동 사진 검수 후에는
`head.candidates`를 다시 실행해야 한다. 위 렌더 검증 스크립트는 현재 샘플 5건의
추천 캐시와 기본 캐릭터가 준비된 환경에서 실행한다.

출처: [공식 anime-face-detector](https://github.com/hysts/anime-face-detector),
[배포 패키지 0.1.0](https://pypi.org/project/anime-face-detector/0.1.0/),
[YOLOv3 모델](https://huggingface.co/hysts/anime-face-detector-yolov3),
[HRNetV2 모델](https://huggingface.co/hysts/anime-face-detector-hrnetv2).
프로젝트·가중치는 MIT 표시를 따르며 vendored 구성 요소는
[THIRD_PARTY_LICENSES](https://github.com/hysts/anime-face-detector/blob/main/THIRD_PARTY_LICENSES.md)의
Apache-2.0 고지를 함께 따른다. 모델 카드의 중심 범위는 정면에 가까운 그림 얼굴이다.

```powershell
.venv/Scripts/python.exe -m pose_curation scoped-build
.venv/Scripts/python.exe -m pose_curation.scoped.evaluation
.venv/Scripts/python.exe -m pytest tests/test_pose_scoped.py tests/test_pose_orientation.py tests/test_pose_review_framing.py tests/test_pose_curation.py -q
```

`scoped/index.py`가 선택·색인·리비전 유효성, `geometry.py`가 군집 피처,
`matching.py`가 관측과 시점 탐색, `queries.py`가 로컬 입력 접근을 맡는다.
HTTP와 화면은 `review/scoped_routes.py`, `review/static/scoped.*`에 둔다.
생성물은 `data/curation/scoped-library/`, 평가 집계는
`data/dev/scoped-library/evaluation.json`에 저장한다.
관련 자동 검사 55개가 통과했다. 실제 추천 각도의 반신 FBX를 HTTP로 내려받아
생성 파일의 SHA와 비교했으며, 52개 뼈 재가져오기 오차는 약 5.96e-7이었다.
각도 BVH 경로의 422 거부와 원본 BVH 해시 보존도 확인했다. 브라우저 검수 화면은
`data/dev/scoped-library/rough-match-review.png`, `fbx-only-review.png`에 보관한다.

같은 배치에 대한 `build`와 `render`는 순서대로 실행한다.

`build --offline`은 검증된 다운로드 캐시만 사용한다. `build --limit 1`은 첫 클립의 후보만 만든다.
입력 설정이나 요청 클립이 바뀌면 새 배치 이름을 사용한다.
`render --limit 1`은 첫 포즈의 외형을 먼저 확인할 때 사용한다. 이후 `--limit` 없이 실행하면 나머지를 처리한다.
렌더 워커 수는 1~4이며 각 워커의 Blender 스레드는 2개다. 기본값 2는 로컬 작업 중의 부하를 줄이기 위한 값이다.

두 생성 단계 모두 재실행할 수 있다. 수집은 원본 해시와 BVH 해시, 렌더는 BVH·캐릭터·Blender·변환 코드·조명 설정 해시로 재사용 여부를 판단한다.
렌더 결과의 4개 이미지가 모두 존재하고 해시가 일치할 때만 검수 목록에 게시한다.
새 이미지의 해시를 URL에 포함해 이전 미리보기 캐시가 남지 않게 한다.
미완료 항목은 준비 상태를 표시하며, 골격은 상세 화면의 별도 탭에서 확인한다.

## 책임 분리

| 모듈 | 역할 |
| --- | --- |
| `sources/style100.py` | 공식 카탈로그·프레임 범위 파싱, 다운로드 및 해시 캐시 |
| `motion.py` | BVH 읽기, 고정 주기로 샘플링, 1프레임 추출 및 저장 전후 FK 검사 |
| `selection.py` | 유지 구간·움직임의 정점·동작 단계 후보 중 서로 다른 자세 선택 |
| `quality.py` | 기존 라이브러리에서 정규화된 3D 관절 거리로 유사 포즈 찾기 |
| `audit.py` | FK 관절 각도 검사, 시각 검수용 캐릭터 모음 이미지 생성 |
| `pipeline.py` | 위 단계를 조립하고 배치 manifest 및 후보 검색 DB 저장 |
| `candidates.py` | 현재 BVH 파일로 공통 검색 feature 및 후보 DB 생성 |
| `hands/bvh.py`, `hands/presets.py` | 몸 자세를 유지하면서 손가락 계층 및 회전 채널 추가 |
| `hands/batch.py` | 원본 보존, 손 보완 출처 기록, 재실행 및 후보 DB 갱신 |
| `rendering/profiles.py` | 100STYLE 관절과 변환기의 22개 표준 관절 연결 |
| `rendering/worker.py` | 독립 Blender 프로세스에서 운영 변환기 호출 및 결과 렌더링 |
| `rendering/scene.py`, `neutral.sl` | 인체 모델의 카메라, 4방향 구도, 회색 배경과 조명 |
| `rendering/fingers.py` | BVH 손가락 회전을 캐릭터의 로컬 관절 축으로 변환해 적용 |
| `rendering/batch.py` | 제한된 수의 워커 실행, 캐시 검증, 완성된 미리보기 게시 |
| `review/catalog.py` | 읽기 전용 기존 DB와 신규 배치 목록 통합 |
| `review/store.py` | 포즈 파일 해시에 묶인 검수 결정 및 변경 이력 저장 |
| `review/selection.py` | 기본 목록에서 제외할 항목과 로컬 DB에 포함할 항목의 공통 규칙 |
| `publication.py` | 기존 항목의 제외 적용, 채택한 신규 항목을 합쳐 별도 로컬 검색 DB 생성 |
| `review/app.py`, `review/static/` | 로컬 API와 검수 UI |

수집·선택·파일 생성은 웹 요청 안에서 실행하지 않는다. 검수 서버도 Blender나 추론 모델을 import하지 않는다.
100STYLE 프로파일은 렌더 워커 안에서만 등록하므로 운영 변환기의 고정 소스 및 정책 파일은 바뀌지 않는다.
추가된 `Chest3`의 변환은 계층 안에 유지하고, 목과 쇄골이 분기하는 `Chest4`를 상부 척추로 매핑한다.
안전 폴백이 선택될 수 있으므로 변환 중의 메모리 씬 대신 **최종 FBX를 다시 읽어서** 렌더링한다.

## 데이터와 검수

```text
data/curation/
  sources/100style/                  # 공식 카탈로그, 원본 및 다운로드 메타데이터
  characters/                       # 서비스 캐릭터
  batches/<batch>/
    manifest.json                   # 출처, 라이선스, 프레임, 해시, 선택 이유, 생성 상태
    bvh/                            # 원본 관절 계층을 보존한 1프레임 BVH
    hands/<fingerprint>/            # 손가락이 추가된 BVH; 원본 몸 BVH와 분리
    candidates.db                   # 운영 검색과 동일한 feature 생성 경로의 후보 DB
    character-thumbs/<fingerprint>/  # 정면/45도/측면/후면, 256px JPEG
    render-results/<fingerprint>/   # 포즈별 변환 보고서, 워커 로그와 재현용 job
    ATTRIBUTION.txt
  reviews.sqlite                    # 검수 결정 및 이력; 배치 생성 결과와 분리
  audit/                            # 관절 검사 JSON, 시각 검수 이미지와 결정 근거
  library/poses.db                   # 제외 적용 + 채택한 신규 포즈의 로컬 검색 DB
  library/manifest.json              # DB 해시, 기존/신규 개수와 제외 이력
```

검수 화면에서 전체/기존/신규/제외 목록, 이름·스타일, 배치, 검수 상태로 필터링한다.
상세 화면은 4방향 캐릭터, 회전 가능한 BVH 골격, 가장 유사한 기존 포즈, 출처와 프레임을 보여 준다.
채택·보류·제외·대기와 메모는 별도 SQLite에 저장한다. 파일 내용이 바뀌면 이전 검수 결과를 재사용하지 않는다.
`/api/reviews/export`에서 기록을 JSON으로 내보낼 수 있다. 채택은 검수 결과이며 운영 게시 단계는 별도다.

**제외 후 저장하면 기본 목록과 기존/신규 목록에서 즉시 빠진다.** 이전에 저장된 제외 결정도 같은 규칙을 적용한다.
원본 파일은 보관하며, `제외한 포즈`에서 대기·채택·보류로 다시 저장하면 기본 목록으로 복구된다.
`/api/summary.total`과 `groups`는 제외한 항목을 뺀 수이고, `excluded`는 별도 개수다.
`/api/poses?group=excluded` 또는 `status=rejected`로 제외 목록을 볼 수 있다. 직접 상세 링크와 BVH는 복구 검수를 위해 유지한다.

## 관절 검사와 로컬 라이브러리 반영

```powershell
# 기존 전체의 파일·관절 좌표를 검사하고, 큰 굽힘이 있는 후보의 정면/측면 모음 생성
.venv/Scripts/python.exe -m pose_curation audit --group existing `
  --flagged-only --output data/curation/audit/existing

# 신규 배치의 전체 포즈를 시각 검수
.venv/Scripts/python.exe -m pose_curation audit `
  --batch 100style-dynamic-20261001 --output data/curation/audit/dynamic

# 현재 검수 결정을 별도 로컬 검색 DB로 구체화
.venv/Scripts/python.exe -m pose_curation publish
```

`audit`의 굽힘 각도는 일직선일 때 0도다. 손목 65도 초과, 팔꿈치·무릎 155도 초과는 **검토 요청**으로만 표시한다.
손목은 손끝 대신 손바닥의 중지 시작점을 사용해서 주먹을 쥔 것과 손목 꺾임을 구분한다.
깊게 앉기나 발차기도 큰 각도를 가지므로 각도만으로 제외하지 않는다. 반대로 피부·메시 찌그러짐은 각도 검사로 잡을 수 없어 렌더 확인이 필요하다.
모음 이미지 번호와 실제 포즈 ID는 같은 폴더의 `audit.json`에 저장된다.

`publish`는 기존 포즈 중 `rejected`를 제외하고, 신규 포즈는 **현재 BVH 해시에 대한 `accepted`만** 포함한다.
신규 포즈의 캐릭터 이미지 4장과 해시도 확인한다. 원래 검색 projection을 바이트 그대로 복사하며,
현재 BVH의 절대 경로를 기록하므로 결과 DB는 이 컴퓨터의 로컬 사용용이다.
실패하면 마지막 정상 DB를 유지한다. 원본 `data/poses.db`와 `data/bvh/`는 수정하지 않는다.
검수 화면 변경은 즉시 반영되지만 내보낸 DB는 그 시점의 스냅샷이므로, 이후 결정이 바뀌면 `publish`를 다시 실행한다.
검색/BVH 검증에 사용할 때는 `DB_PATH`를 `data/curation/library/poses.db`의 절대 경로로 지정할 수 있다.
추론 서버의 기존 썸네일 API는 `DATA_DIR/thumbs`를 별도로 읽으므로, 이 DB만 바꾸는 것을 완전한 배포로 간주하지 않는다.
검수 서버는 각 배치의 이미지 경로를 읽어 기존/신규 미리보기를 함께 제공한다.

역동적 동작 수집 설정은 `config/pose_curation_dynamic.json`과 `config/pose_curation_dynamic_extremes.json`에 있다.
각각 12개 스타일 36클립, 팔다리를 크게 뻗는 4개 스타일 8클립을 사용한다.
스타일 이름만으로 채택하지 않고, 한 프레임에서 동작이 잘 읽히는지 렌더를 보고 결정한다.
펀치·발차기는 `fist`, 수도·큰 팔 동작은 `open`, 나머지는 `relaxed`를 사용했다.

2026-10-01 검수에서는 기존 1,248개에 대해 FK 검사를 하고, 반전본을 제외한 625개 정면 미리보기를 훑었다.
각도 경고 32개와 육안 의심 항목은 추가 방향으로 확인했다. 사용자 제외 9개와 추가 17개를 합쳐 기존 26개를 제외했다.
첨부된 예시는 `Draw A Great Sword 2_00000`으로 대조됐으며 반전본도 제외했다.
신규 동작 후보 150개를 4방향으로 검수해 94개를 채택하고, 정적이거나 동작 구분이 약한 56개는 제외했다.
이 결정은 `[AI 시각 검수 2026-10-01]` 메모로 구분하며, 완전한 해부학적 검증을 뜻하지 않는다.
원본/이미지/검수 근거는 `data/curation/audit/`, 사용자가 먼저 저장한 9개 결정의 스냅샷은 `user-review-audit.json`에 보관한다.

파일 검사는 통과한 항목만 후보가 되지만, 캐릭터 렌더 성공은 포즈 품질의 최종 승인을 뜻하지 않는다.
검수자는 실루엣, 관절 방향, 접지, 손·발 변형과 기존 라이브러리 대비 유용성을 확인한다.
유사도는 몸통 길이로 정규화한 12개 몸 관절의 RMS 거리다. 중복 참고 지표로 표시하며 자동 삭제하지 않는다.

## 손가락 보완

100STYLE 원본은 손목과 손 끝 End Site까지 있으며 손가락 관절은 없다.
`hands` 단계는 기존 손 길이에 맞춰 양손 각각 5개 손가락 × 3개 관절을 추가한다.
추가되는 회전 채널은 90개이며, 기존 몸의 회전·위치 채널과 손목까지의 FK 좌표를 검증해서 보존한다.
손 비율과 모양은 절차적으로 만든 기본값이며 원본 모션에서 복원한 것이 아니다.
manifest의 `hand_augmentation.captured_from_source=false`와 화면의 출처 표시로 구분한다.

기본값은 `relaxed`(힘을 뺀 손)이며, `open`(편 손), `fist`(주먹)를 양손에 따로 지정할 수 있다.
특정 포즈만 바꾸려면 정확한 ID를 `--pose`로 전달한다. 여러 포즈에는 `--pose`를 반복한다.

```powershell
.venv/Scripts/python.exe -m pose_curation hands `
  --batch-dir data/curation/batches/100style-pilot-20261001 `
  --pose style100_Angry_FR_f000886_945a1548 --left fist --right open
# 이어서 위 render 명령 실행: BVH가 바뀐 포즈만 다시 렌더링한다.
```

`body_bvh`와 `body_bvh_sha256`은 보관한 몸 원본을 가리킨다. 프리셋을 바꿀 때도 항상 이 원본부터 만든다.
같은 설정으로 재실행하면 관절을 중복 추가하지 않으며, 같은 배치에서 렌더링 중에는 BVH를 바꾸지 않는다.
검수 및 다운로드 API, 후보 DB는 손이 추가된 BVH를 가리키고, 검색 feature는 몸 원본과 같다.
BVH 파일 해시가 달라지므로 이전 파일에 대한 검수 기록은 새 파일의 승인으로 재사용하지 않는다.

운영 변환기는 22개 몸 관절을 담당한다. 검수 렌더의 별도 손 어댑터가 추가된 손가락 채널을
캐릭터 관절의 로컬 축으로 옮기고, 손 적용 전후 몸 관절 변환이 그대로인지 검사한다.
검수 화면의 **3D 골격 → 왼손 확대 / 오른손 확대**에서 실제 다운로드될 관절과 손 모양을 확인할 수 있다.
일부 스타일은 손에 든 물건이나 행동에 맞는 별도 손 모양이 필요하므로 프리셋을 조정하며 검수한다.

## 현재 소스 범위

첫 어댑터는 [Ian Mason 등의 100STYLE 공식 배포](https://www.ianxmason.com/100style/)를 사용한다.
라이선스는 CC BY 4.0이며 원본 링크, 저작자, 라이선스 링크, 원본 SHA, 프레임 번호와 변경 내용을 기록한다.
파일 전체의 샘플링 대신 공식 Frame Cuts의 내부 구간을 사용한다.
포즈를 저장할 때 루트의 X/Z 이동과 Y축 회전만 제거하고 높이·기울기·팔다리 회전 및 계층은 보존한다.
파일 저장 전후 FK 좌표 일치와 `Frames: 1`을 확인한다.

파일럿 설정은 20개 스타일 × 정지/걷기/달리기의 60개 클립이다.
스타일별 최대 3개 대표 자세를 선택하며, 정적인 클립은 서로 비슷한 자세를 추가로 만들지 않는다.
실행 결과는 신규 146개 포즈와 584개 검색 projection이다. 기존 S3 스냅샷은 1,248개 포즈다.
기존 DB의 `synthetic`/`n/a` 출처 표기는 원본 스냅샷의 값이다. 신규 출처 정보와 섞어 덮어쓰지 않는다.

## 테스트

```powershell
.venv/Scripts/python.exe -m pip install pytest httpx
.venv/Scripts/python.exe -m pytest tests/test_pose_curation.py tests/test_pose_hands.py -q
```

테스트는 직접 생성한 관절 계층을 사용하며 외부 BVH나 AWS, Blender가 없어도 실행한다.
출처 계약, 결정적인 프레임 선택, BVH 좌표 보존, 재실행과 손상 복구, 검수 저장, 경로 경계,
이미지 4장과 파일 해시가 일치해야 게시되는 렌더 계약을 확인한다.
손 테스트는 30개 관절·90개 채널, 세 가지 손 모양, 좌우 굽힘 방향, 몸 좌표·검색 feature 보존,
원본 보관 및 중복 추가 방지를 확인한다.
실제 Blender 및 서비스 캐릭터 검사는 위 `render --limit 1` 실행과 생성 이미지 확인으로 수행한다.

## 전투 포즈와 실제 러프 비교 (2026-10-01)

`/coverage`는 제공 러프와 익명 처리한 실제 입력, 모델 관절, 확장 전후 Top-1,
수동 조정에 사용할 기준 자세를 함께 표시한다. 원본 라이브러리 화면은 `/`다.
전체 439장 자동 기록과 사람이 확인한 대표 부족 사례 17종을 구분한다.
검수 화면은 `127.0.0.1`에만 바인딩한다. 비공개 원본과 평가 산출물은 git에서 제외된
`data/curation/coverage/<날짜>/` 아래에 보관하며 화면에는 설치 ID/S3 key/로컬 경로를 내보내지 않는다.

### 코드 경계

| 모듈 | 책임 |
|---|---|
| `sources/quaternius.py`, `quaternius_export.py`, `quaternius_rig.py` | 출처 해시 검증, Blender 추출, 두 버전 관절명 매핑 |
| `combat.py` | 클립별 대표 프레임 선택, 출처가 있는 후보 manifest |
| `authored.py` + `config/pose_curation_combat_*.json` | 사람이 편집할 3D 목표·굽힘 방향, 뼈 길이를 보존하는 2본 IK |
| `coverage/inputs.py`, `inference.py` | 이미지 목록/익명화, 실제 모델 추론과 재실행 캐시 |
| `coverage/evaluate.py`, `orientation.py` | 같은 입력 관절의 전후 비교, 몸 전체 방향만 바꾸는 구도 후보 |
| `coverage/report.py`, `review/coverage.py` | 검수 주석과 측정 결과 결합, 로컬 보고서 API |

운영 `src`, `api`, 변환기의 계약은 변경하지 않았다. 기존 `normalize_skeleton`,
`knn_geometric`, BVH FK와 DB writer를 재사용한다. 새 query 관절을 VLM으로 만들지 않는다.

### 출처부터 다시 만들기

[Quaternius Universal Animation Library](https://quaternius.com/packs/universalanimationlibrary.html)와
[두 번째 라이브러리](https://quaternius.com/packs/universalanimationlibrary2.html)의 CC0 배포를 사용한다.
저작자가 OpenGameArt에 올린 standard ZIP의 SHA256과 GLB SHA256을 설정에 고정했다.
소스가 바뀌면 해시 검증에서 실패하며, 자동으로 새 버전을 신뢰하지 않는다.
52개 관절 중 30개는 원본 손가락이다. 모든 추출 프레임은 원본과 FK 차이 0.02cm 미만을 확인한다.
직접 만든 자세는 원본 손가락 주먹 채널과 관절 길이를 재사용하고 제작 사실을 표시한다.
원본 애니메이션이 모션 캡처라는 보장은 없으며, 수집 프레임과 직접 제작은 별도 출처다.

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-curation.txt
.venv/Scripts/python.exe -m pose_curation.sources.quaternius `
  --config config/pose_curation_combat_sources.json `
  --blender data/tools/blender-5.2.0-windows-x64/blender.exe
# 검증된 로컬 소스만 사용하려면 --offline

.venv/Scripts/python.exe -m pose_curation.combat `
  --pool data/curation/sources/quaternius/export-1/pool.json `
  --pool data/curation/sources/quaternius/export-2/pool.json `
  --batch data/curation/batches/combat-quaternius-<새배치> --count 3

.venv/Scripts/python.exe -m pose_curation.authored `
  --config config/pose_curation_combat_authored.json `
  --batch data/curation/batches/combat-authored-<새배치>
# specific/guard 설정도 각자 새 배치로 생성한다.
# 이후 기존 render → 4방향 시각 검수 → publish 흐름을 사용한다.
```

기존 검수 배치를 덮어쓰지 않는다. Quaternius 프레임 번호는 glTF를 Blender의 24fps
장면에서 샘플링한 프레임이다. 원본 캡처 프레임 번호로 해석하지 않으며 초 단위 시간도 기록한다.
3D 제작 recipe의 `reference_roughs`는 제작 참고 대상이다. 자동 검색 성공 근거가 아니며,
실제 성공 여부는 같은 관절로 재검색한 보고서와 원본 이미지 대조를 통해 판단한다.

### 실제 입력 분석 재실행

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-curation-analysis.txt
$env:TORCH_HOME = 'C:/workspaces/Standin-server/data/curation/models/rtmlib'
.venv/Scripts/python.exe -m pose_curation.coverage.inference `
  --inputs data/curation/coverage/20261001/inputs.json `
  --output data/curation/coverage/20261001/extraction
.venv/Scripts/python.exe -m pose_curation.coverage.evaluate `
  --inputs data/curation/coverage/20261001/inputs.json `
  --extraction data/curation/coverage/20261001/extraction `
  --database data/curation/library/poses.db `
  --output data/curation/coverage/20261001/after.json
.venv/Scripts/python.exe -m pose_curation.coverage.report `
  --root data/curation/coverage/20261001
```

`before.db`는 작업 시작 시 1,316개 DB의 보관본이다. `before.json`과 `after.json`은
이미지/DB/모델 해시, 모델의 원래 좌표와 score, 검출별 Top-5를 기록한다.
보고서 생성은 모델·입력 해시·관절·score가 달라졌으면 실패한다.
`annotations.json`은 대표 사례의 의도한 검출 번호, 원인, 남은 문제와 기준 포즈 ID를 저장한다.
입력 이미지와 JSON에는 개인정보가 있을 수 있으므로 저장소나 공개 호스팅에 넣지 않는다.

모델은 실제 RTMPose Body current-X다. 운영의 HumanArt rescue·VLM 크롭·재랭킹 전체를 실행한
종단 간 평가가 아니다. 최소 관절 조건을 통과해도 관절 배정이 틀릴 수 있다.
거리가 줄어든 것만으로 사용 가능 판정을 하지 않는다. 같은 자세의 여러 이미지 버전도
독립적인 평가 사례로 세지 않는다.

구도 보완은 검수된 몸의 루트 회전만 변경하며 팔·다리·손가락 회전과 뼈 길이를 보존한다.
같은 입력 사례로 방향을 보정하는 **캘리브레이션**이므로 일반화 성능으로 보고하지 않는다.
단일 포즈의 공간 배치이며, 기울어진 자세를 서 있는 접지 동작으로 쓰지 않는다.

이번 결과는 후보 117개 중 93개 채택, 24개 제외다. 채택분은 원본 애니메이션 54개,
직접 제작 34개(반전 포함), 구도 변형 5개다. 최종 로컬 DB는 기존 1,222 + 신규 187 =
1,409개, projection 5,636개다. 이전에 제외한 기존 26개를 유지했다.
이번 93개의 손가락 30관절, BVH 1프레임, 372장 캐릭터 미리보기를 검수했다.
운영 S3 및 원본 `data/poses.db`에는 게시하지 않았다.

검증 명령은 아래와 같다. 실제 모델 추론은 별도의 분석 작업이며 단위 테스트에서는 다운로드하지 않는다.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_pose_curation.py tests/test_pose_hands.py tests/test_pose_coverage.py -q
```

## 직접 제작 자세 재검수 (2026-10-01 후속 수정)

초기 각도 검사만으로는 자연스러움을 보장하지 못했다. 사용자가 지적한
`combat_authored_flying_side_kick_mirror`를 포함해 직접 제작 36개와 해당 자세에서
파생된 구도 2개를 다시 렌더링·검수했다. 각도 검사에서 경고가 없어도 육안으로 어색하면 제외한다.

- 1차: 발의 월드 회전 고정을 제거하고, 부모 관절의 회전에서 자식 관절을 이어서 계산한다.
  공중 발은 정강이를 따라가며, 접지 발은 설정된 방향을 향하되 발목 보정량을 제한한다.
- 2차: 공중 발차기의 접은 무릎을 몸 아래로 모으고 뒤꿈치를 뒤로 접었다.
  서서 하는 발차기는 지지 다리와 몸통 중심, 가드, 시선을 함께 조정했다.
- 3차: 착지의 비대칭 중심과 팔 위치, 무릎차기 가드, 수평 돌진의 팔·다리·시선을 조정했다.
  양발 벌리기는 수정 후에도 과도한 실루엣이 남아 좌우 2개를 추가 제외했다.

직접 제작 32개와 그로부터 파생된 구도 1개는 현재 파일 해시로 다시 채택했다.
기존에 제외됐던 공중 웅크리기 2개와 구도 1개는 제외를 유지했다.
최신 로컬 DB는 **1,407개 = 기존 1,222 + 신규 185**, projection 5,628개다.
기존 26개 제외와 다른 배치의 검수 결정은 유지한다. 과거 1,409개 수치는 최초 확장 시점의 기록이다.

`kinematics.py`가 팔다리 회전 수송과 발의 제한된 보정을 담당한다.
공중/접지는 recipe의 `left_foot`/`right_foot` 설정으로 명시하며 이름 기반 예외를 넣지 않는다.
이 제한은 제작 가드이며 해부학적 검증이나 실제 지면 접촉을 보장하는 물리 시뮬레이션이 아니다.
캐릭터의 스킨 변형·교차·실루엣은 반드시 다시 렌더링해서 확인한다.

```powershell
.venv/Scripts/python.exe -m pose_curation.revisions `
  --batch data/curation/batches/combat-authored-20261001 `
  --config config/pose_curation_combat_authored.json `
  --revision next-review --recipe flying_side_kick
# --recipe 생략 시 해당 배치 전체. 이어서 render → 시각 검수 → publish.
```

수정 도구는 검수 중인 배치를 거부하고, 원본 BVH·이미지·manifest를 배치의
`revisions/<수정명>/before/`에 보관한다. 생성 실패 시 보이는 BVH와 manifest를 유지한다.
수정본은 썸네일과 이전 승인을 자동 승계하지 않으며, 새 해시로 다시 검수한다.
같은 BVH가 재생성됐을 때만 동일한 렌더 캐시를 재사용한다.
`rebase_compositions()`는 수정한 기준 자세로 구도 변형도 다시 만들고 원본 해시를 갱신한다.

전체 38개 수정본의 4방향 시트, 이전 이미지, 3차 검수 사유는
`data/curation/audit/authored-repair-20261001/`과 각 배치의 `revisions/`에 저장한다.
미리보기 상세의 ‘자세 재검수’에 구체적인 변경점을 표시한다.
손가락 30개와 단일 프레임은 유지하며, 서비스 변환기와 운영 S3는 수정하지 않았다.

## 팔 역관절·몸통 관통 재검수와 ACCAD 확장 (2026-10-01 최종)

현재 로컬 검색 DB는 **1,368개 = 기존 1,199 + 신규 169**, projection 5,472개다.
위 1,407개는 이 재검수 전 기록이다. 이전에 제외한 기존 26개를 유지하고 기존 23개,
이전 신규 28개를 추가 제외했다. ACCAD 후보 22개 중 12개를 새로 채택했다.
제외는 원본 삭제가 아니라 검수 DB의 상태 변경이며 제외 탭에서 확인할 수 있다.

사용자가 지적한 `deep_landing`의 팔꿈치 방향과 `side_kick_mirror`의 가슴 관통을
실제 Master V2 메시로 재현했다. 세 관절의 무부호 각도만으로는 역방향 굽힘을 구분할 수
없었다. `anatomical_arm_frames()`가 IK로 정한 어깨·팔꿈치·손목 위치를 유지하면서
상완의 축 회전을 굽힘 평면에 맞춘다. 착지·옆차기·공중 옆차기·머리 보호 자세의
목표 위치도 수정했다. 실패한 수정안을 보관하고 다섯 번째 수정안까지 렌더링을 반복했다.
직접 제작 32개와 파생 구도 1개를 4방향으로 다시 확인해 채택했다.

모션캡처의 작은 체형 차이로 생긴 관통은 `corrections.py`에서 상완을 8도씩 바깥으로
조정했다. 누적 16도까지 제한하고 손가락·팔꿈치 로컬 회전, 뼈 길이, 다른 관절의 위치를
보존한다. 이전 신규 35개 중 33개가 재검수를 통과했고 2개는 관통이 남아 제외했다.
ACCAD에도 같은 제한을 적용했으며 보정 실패 2개와 구별성이 낮은 준비·복귀 프레임
8개를 제외했다. 합성 손 모양과 원본 손 데이터는 화면과 manifest에서 구별한다.

### 검사 범위와 게시 조건

- 전체 기존·후보 1,661개의 BVH 팔/몸통 캡슐 교차를 선별 검사했다.
- 기존 1,248개의 소스 팔꿈치 굽힘 방향을 별도로 검사했다. 심한 역방향 2개는
  원본 4방향 이미지와 함께 확인해 제외했다.
- 이전 채택 신규 185개, 새 ACCAD 22개, 기존 캡슐 경고 22개는 실제 캐릭터 변환 후
  팔꿈치 평면과 팔·손/몸통 메시 교차를 검사했다. 수정본은 다시 검사했다.
- 새로 채택하는 모든 자세는 현재 BVH 해시의 검사 기록, 미해결 경고 0개,
  일치하는 4방향 이미지 해시, 현재 해시에 묶인 채택 결정을 요구한다.
  `publication.py`가 누락·오래된 검사·남은 경고를 거부하며 마지막 정상 DB를 보존한다.

메시 검사는 팔/손과 몸통, 굽힘 검사는 팔꿈치에 한정한다. 어깨 연결부 교차는 따로
기록한다. 다리끼리·팔끼리·의상 전체의 충돌이나 접지·동역학을 보장하지 않는다.
기존 1,248개 모두를 메시 검사한 것은 아니다. 자동 경고와 4방향 시각 검수를 함께 사용한다.

### 소스 조사 결과

| 소스 | 동작·조건 | 이번 처리 |
|---|---|---|
| [ACCAD / Ohio State](https://accad.osu.edu/research/motion-lab/mocap-system-and-data) | 원본 BVH 149클립, 무술 발차기·펀치·방어·회피. CC BY 3.0 | 11클립에서 22프레임 생성, 12개 채택. 원본 해시·프레임·저자·변경점 기록 |
| [Quaternius UAL](https://quaternius.com/packs/universalanimationlibrary.html), [UAL2](https://quaternius.com/packs/universalanimationlibrary2.html) | 기존 CC0 애니메이션, 점프·타격·무기 | 이전 추가분을 메시 재검사, 문제 프레임 제외 |
| [CMU](https://mocap.cs.cmu.edu/) | 무술·체조 등. 원본·변환 데이터 재판매 제한 문구 | 이번 신규 수집에 포함하지 않음 |
| [NUS/SFU](https://mocap.cs.sfu.ca/nusmocap.html) | 검도·우슈·파쿠르, 연구 목적 한정·상업 제품 금지 | 서비스 라이브러리에 추가하지 않음 |
| [KayKit Legacy](https://kaylousberg.itch.io/kaykit-animations) | CC0, 다리 없는 캐릭터용 애니메이션 | 인간 전신 동작의 보완 소스로 부적합하여 제외 |

ACCAD는 캡처 당시의 휴식 자세가 T-pose가 아니므로 `sources/rebase.py`에서 축을
변환한다. 별도로 다시 읽은 원본의 실제 관절 FK와 일치하는지 검증하며, 손가락으로
대체되는 가상 손끝 End Site는 비교에서 제외한다. 양손 30관절과 주먹은 합성한다.
이후 체형 보정을 적용한 포즈는 원본 관절 위치가 그대로라는 의미가 아니며
`anatomy_correction`에 보정량을 남긴다. 출처·변경 고지는 배치의 `ATTRIBUTION.txt`와
`sources/accad/provenance.json`에 저장한다.

```powershell
# 새 배치로 원본 프레임 추출 → render → 실제 이미지 검수 → 현재 해시 채택 → publish
.venv/Scripts/python.exe -m pose_curation.sources.accad `
  --source data/curation/sources/accad/bvh --batch data/curation/batches/<새-배치>
# 실제 변환 메시의 독립 재검사. --pose를 반복하면 지정 포즈만 검사한다.
.venv/Scripts/python.exe -m pose_curation.rendering.diagnostics `
  --output data/curation/audit/<새-검수> --workers 2 --render
# 기존 BVH 선별 검사 결과를 포함하려면 --triage <triage.json>을 명시한다.
.venv/Scripts/python.exe -m pytest tests/test_pose_curation.py tests/test_pose_hands.py tests/test_pose_coverage.py tests/test_pose_anatomy.py -q
```

최종 수정 전후 이미지·제외 사유·현재 해시는 `data/curation/audit/anatomy-20261001/`에
저장한다. 단위/통합 검사 32개와 코어 스모크 54개를 통과했다. 러프 439장의 검색 비교도
현재 1,368개로 다시 계산하며 모델 좌표는 바꾸지 않는다. 운영 S3와 원본 DB는 그대로다.

## 허리 접힘 보정 (2026-10-01 후속)

첨부된 돌려차기와 어퍼컷은 ACCAD 프레임이었다. `ToSpine`은 골반 뒤쪽 보조
관절인데 이를 요추로 취급해 약 100도의 고정 회전과 반대 방향 척추 회전이
허리 스킨에 적용됐다. `torso.py`는 골반·가슴 기준축을 복원하고 내부 척추 위치를
부드러운 곡선으로 재구성한다. 내부 척추 표지는 원본 캡처 좌표와 달라진다.
이 단계에서 엉덩이·사지·쇄골·목·머리 위치와 몸통 외 월드 회전은 보존한다.

Quaternius는 위치를 유지하면서 골반에서 가슴까지의 회전을 세 척추 관절에
분산한다. 팔다리와 손가락의 월드 회전도 유지한다. 직접 제작 recipe의 몸통
굽힘도 `distributed_spine_frames()`로 분산하며 제작 버전은 4다. 현행 직접 제작
32개의 몸통 pitch는 0이어서 파일을 재생성하지 않고 실제 캐릭터를 재검사했다.

채택분 **43개(ACCAD 12 + Quaternius 29 + 구도 변형 2)**를 수정하고 4방향으로
재검수했다. 수정이 필요 없던 전투 포즈 48개도 척추 및 팔/몸통 메시 검사에서
경고가 없었다. ACCAD 제외 10개는 허리가 수정돼도 기존 제외를 유지한다.
따라서 총수는 **1,368개 = 기존 1,199 + 신규 169**로 동일하다.

허리 변경 뒤 새로 생긴 팔/몸통 접촉은 어퍼컷 오른팔 바깥 8도·앞 4도,
회피 왼팔 바깥 4도로 해소했다. 이 단계는 선택한 팔 위치를 변경하므로
원본 FK 보존 주장에 포함하지 않는다. 변경량과 이전 단계는 `torso_correction`에
기록하며, 손가락과 팔꿈치의 로컬 회전은 유지한다. 파일 저장은 9자리 정밀도로
유지해 몸통 수정 중 손가락 회전이 반올림으로 달라지는 것도 방지한다.

실제 캐릭터 검사에는 휴식 자세의 곡률을 제거한 척추 분절 회전 검사를 추가했다.
35도는 시각 재검수를 요구하는 제작 가드이며 임상적 운동 범위나 동역학 보증이
아니다. 검사 통과만으로 채택하지 않고, 변경된 BVH 해시의 4방향 이미지와 검수
결정을 다시 요구한다. 상세 화면의 ‘허리 보정’·‘척추 검사’에서 확인할 수 있다.

```powershell
# Mixamo 무접두사 1프레임 후보의 집중 회전 재분산. 이후 render·시각 검수 필수.
.venv/Scripts/python.exe -m pose_curation.torso `
  --batch data/curation/batches/<배치> --revision <새-수정명> `
  --mode distribute --pose <포즈-ID>
.venv/Scripts/python.exe -m pytest tests/test_pose_curation.py tests/test_pose_hands.py tests/test_pose_coverage.py tests/test_pose_anatomy.py tests/test_pose_torso.py -q
```

수정 전 DB·검수 기록, 실패한 수정안, 최종 4방향 시트와 검증 기록은
`data/curation/audit/torso-20261001/`에 보관한다. 이번 단위/통합 테스트는 37개
통과했다. 실제 입력의 추출 관절을 유지한 채 439장 검색 비교를 새 DB로 갱신한다.
서비스 변환기·운영 S3·원본 `data/poses.db`는 변경하지 않았다.

## 웹툰 상황별 포즈 200개 (2026-10-02)

`config/pose_scenarios_20261002.json`에 일상·판타지·전투·로맨스를 각각 50개씩
명시한다. 각 항목에는 한국어 상황, 발/골반 자세, 손 목표, 팔꿈치 방향,
양손 모양, 시선, 소품 배치가 있다. 같은 파일을 다른 이름으로 복사하지 않으며
동일 BVH 해시가 발생하면 제작을 중단한다. 캐릭터는 기존 Master V2를 사용한다.

| 카테고리 | 상황 예시 |
|---|---|
| 일상 | 앉아서 머리 말리기, 스트레칭, 컵 들기, 상자 들어 올리기, 청소 |
| 판타지 | 양손 검, 창 들기, 방패 방어, 활 당기기, 지팡이, 무거운 보물상자 |
| 전투 | 검 찌르기, 창 겨누기, 무릎 꿇고 조준, 가드, 잡기, 방어 |
| 로맨스 | 컵을 두 손으로 감싸기, 꽃·선물 건네기, 손 잡기, 포옹 준비, 앉아서 대화 |

이 배치는 CC0 Quaternius 리그에서 **직접 제작한 자세**다. 새로 캡처한 모션이나
사용자 러프에서 추론한 관절이 아니다. 손가락 30관절은 원본 주먹 회전을 바탕으로
합성했으며, 손목 위치를 유지한 전완 축 회전으로 손바닥 방향을 조절한다.
원본·제작 레시피·이전 버전의 해시를 남긴다.

검·창·컵·상자 등은 잡는 위치를 보기 위한 **단순 소품 가이드**다. BVH에는
한 인물의 관절만 들어간다. 소품 메시와 좌석, 상대 인물은 내려받는 BVH에 포함되지
않으며 장면에서 별도로 배치해야 한다. 실제 물체의 무게·마찰·손가락 접촉을
동역학으로 검증한 결과가 아니다. 자세의 자연스러움은 실제 4방향 렌더로 확인한다.

### 제작과 재검수

```powershell
# 검토 기록을 덮어쓰지 않도록 새 배치 디렉터리를 사용한다.
.venv/Scripts/python.exe -m pose_curation.scenarios.build `
  --config config/pose_scenarios_20261002.json --batch data/curation/batches/<새-배치>
.venv/Scripts/python.exe -m pose_curation render `
  --batch-dir data/curation/batches/<새-배치> `
  --blender data/tools/blender-5.2.0-windows-x64/blender.exe --workers 4
.venv/Scripts/python.exe -m pose_curation qa --batch <새-배치>
# /scenarios에서 상황별 확인 → 상세 화면에서 5항목 검수·메모·채택 → publish
.venv/Scripts/python.exe -m pose_curation publish
```

수정은 `scenarios.revise.revise(config, batch, revision, pose_ids)`로 이전 BVH·이미지·
manifest를 백업하고 준비된 새 파일만 적용한다. 렌더 중 수정은 차단한다.
레시피를 수정한 뒤 해당 포즈를 다시 렌더하고 현재 해시로 재검수해야 한다.
전체 배치의 유사도 재계산만 필요하면 다음 명령을 사용한다.

```powershell
.venv/Scripts/python.exe -m pose_curation.scenarios.duplicates `
  --batch data/curation/batches/webtoon-scenarios-20261002
```

팔·다리 IK 목표 오차는 각각 4cm 이하이며 누락·NaN도 차단한다. 팔다리 관절 방향,
척추 분산 회전, 손가락 계층, 팔/몸통 메시 관통, 4방향 파일의 해시도 공통 QA로
검사한다. 소품 또는 좌석 설정이 바뀌면 이전 렌더를 재사용할 수 없다.

이번 검수에서는 팔/몸통 관통 8개를 손 위치와 팔꿈치 방향으로 보정하고, 반무릎
8개의 무릎·발 접지, 컵을 감싸는 양손 간격, 앉은 자세의 무릎 간격 등을 수정했다.
수정된 최종 4방향 이미지를 다시 확인한 뒤에만 채택한다. 기존 포즈와 유사한
자세도 손 모양·도달 위치·동작 단계 차이가 필요한 상황별 변형으로 유지한다.
200개 중 157개는 정규화된 몸 관절 유사도 경고가 있으므로, 200개의 서로 완전히
다른 전신 실루엣이 늘었다는 의미는 아니다. 검색기는 소품·카테고리 의미로 순위를
바꾸지 않는다. 카테고리는 검수 화면의 탐색용 분류다.

| 모듈 | 책임 |
|---|---|
| `scenarios/recipes.py` | 자세·동작 어휘와 명시적 목표 좌표 구성 |
| `scenarios/build.py`, `hands.py` | 단일 프레임 제작과 손·전완 회전 |
| `scenarios/validation.py`, `duplicates.py` | 도달 오차와 기존/배치 내 유사도 |
| `scenarios/revise.py` | 백업 및 선택한 포즈의 버전 교체 |
| `rendering/props.py` | 미리보기 전용 소품 가이드 |
| `review/scenarios.py` | 현재 검수 상태를 표시하는 200개 상황 목록 |

검수 이미지·보정 이력·QA 결과는 `data/curation/qa/scenarios-20261002/`와
배치의 `revisions/`에 보관한다. 목록은 로컬 서버 `/scenarios`, 카테고리별
미리보기는 `/?category=daily` 등에서 확인한다.

최종 로컬 검색 DB는 **1,568개 = 기존 1,199 + 이전 신규 169 + 이번 신규 200**,
투영은 6,272개다. 기존 제외는 유지했다. 신규 채택 전체 369개가 공통 QA의
현재 검수 유효 상태이며, 기존 경고 22개의 4방향 재검수와 경고 해소 근거도
저장했다. 이번 제작·검수·서버 관련 단위/통합 테스트 60개가 통과했다.
운영 S3 게시와 원본 DB 변경은 포함하지 않는다.

고정된 러프 439장의 추출 관절로 검색 비교를 갱신했다. 정량 조건에 해당하는
230명 중 31명의 최상위 검색 거리가 감소했고 199명은 같았다. 31명은 이번 포즈가
최상위에 검색됐다. 이 수치는 기하학적 거리이며 장면 의도나 사용자 성공률의
시각 평가가 아니다. `coverage-comparison.json`에 이전 1,368개와의 비교를 저장했다.
