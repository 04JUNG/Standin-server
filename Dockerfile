# 도원 추론 서버 (Python/FastAPI).
#
# 기본은 mock 모드 — API 키·무거운 모델 없이 바로 뜬다(로컬/CI).
# 실모델은 VLM_PROVIDER/POSE_BACKEND env + requirements.txt의 주석 의존성 해제.
#
# ⚠ 포즈 라이브러리(data/)는 이미지에 넣지 않는다 — Mixamo/CMU 재배포 금지 조항 때문에
#   .gitignore·.dockerignore로 제외돼 있다. 배포 환경에서는 POSE_LIBRARY_URI로
#   번들을 받아 푼다(src/library_source.py). 로컬은 data/에 직접 둔다.
FROM python:3.12-slim

WORKDIR /app

# 코어 + API 의존성만(실모델 의존성은 requirements.txt에서 주석 처리됨)
COPY requirements.txt ./
# rtmlib가 GUI용 OpenCV 패키지를 의존성으로 다시 설치하므로, 화면이 없는 ECS에서는
# 두 패키지를 제거하고 headless 변형만 마지막에 재설치한다.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python opencv-contrib-python \
    && pip install --no-cache-dir --no-deps --force-reinstall "opencv-python-headless>=4.9"

COPY . .

# 루트로 돌리지 않는다. 라이브러리를 받아 풀 디렉터리를 미리 만들고 소유권을 준다.
RUN useradd --create-home --uid 10001 standin \
    && mkdir -p /app/data \
    && chown -R standin:standin /app
USER standin

# 배포 버전을 이미지에 굽는다. 로그의 `version` 필드가 이 값이 된다.
#
# 태스크 정의 env로 주지 않는 이유: env가 이미지 ENV를 덮어쓰는데, 어느 커밋이
# 배포되는지 아는 것은 앱을 빌드한 워크플로뿐이다(CDK는 모른다). 이미지에 구우면
# 나중에 cdk deploy가 돌아도 값이 사라지지 않고, 이미지와 버전이 함께 움직인다.
#
# ⚠ 이 뒤의 레이어는 값이 바뀔 때마다 캐시가 깨진다. 그래서 빌드 맨 끝에 둔다.
ARG DEPLOYMENT_VERSION=development
ENV APP_ENV=development \
    DATA_DIR=/app/data \
    DB_PATH=/app/data/poses.db \
    INDEX_PATH=/app/data/index.pkl \
    POSE_MODELS_ROOT=/app/data/pose-models \
    PYTHONUNBUFFERED=1 \
    DEPLOYMENT_VERSION=$DEPLOYMENT_VERSION

EXPOSE 8000

# 라이브러리가 비면 /healthz가 503을 준다 → 오케스트레이터가 태스크를 교체한다.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
