# 손 라이브러리 썸네일

손 라이브러리와 결과 화면의 손 선택 팝업은 같은 `/hands/{preset}/thumbnail?side=left|right`
PNG를 사용한다. 79종 × 좌우 158장을 `converter/hand_thumbs/`에 포함한다.

기본 인체 `standin-master-v2`의 실제 손 메시와 FBX 출력에 사용하는 `finger_rotations`로
생성한다. 포즈 라이브러리의 `approved-reference-anatomy-detail-dark-v1` 스튜디오,
회색 인체, 어두운 배경을 사용한다. 손가락이 위를 향하도록 손바닥 사선 시점에서
손목과 손끝을 포함해 320 × 320으로 촬영한다. 깊게 굽힌 손은 엄지가 가려지지 않도록
엄지 쪽 사선 구도를 쓴다. 좌우는 인물 자신의 기준이다.

재생성:

```sh
blender -b --factory-startup --python-exit-code 1 \
  --python scripts/render_hand_mesh_thumbnails.py -- \
  --character /path/to/standin-master-v2.fbx
```

입력 캐릭터 SHA-256을 검증하고 모든 프리셋 렌더가 성공한 뒤 manifest를 쓴다.
manifest는 라이브러리 해시, 캐릭터 해시, 스타일, 각 PNG 해시를 기록한다.
샘플은 `--out /tmp/hand-preview --presets open relaxed fist`로 생성할 수 있다.
샘플 생성 시 카탈로그 manifest는 쓰지 않는다. 기존
`render_hand_thumbnails.mjs`는 관절 인형을 그리는 개발용 진단 도구이며
배포용 이미지 재생성에는 사용하지 않는다.

검증: `python -m pytest tests/test_hand_catalog.py`. 배포는 추론 서버 이미지에
새 PNG와 manifest를 함께 포함한다. 클라이언트의 목록, 칩, 선택 팝업은 기존
이미지 API를 그대로 사용한다.
