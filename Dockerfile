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

ENV APP_ENV=development \
    DATA_DIR=/app/data \
    DB_PATH=/app/data/poses.db \
    INDEX_PATH=/app/data/index.pkl \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# 라이브러리가 비면 /healthz가 503을 준다 → 오케스트레이터가 태스크를 교체한다.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
