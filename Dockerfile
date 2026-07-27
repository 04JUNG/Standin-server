# 도원 추론 서버 (Python/FastAPI).
# 기본 mock 모드로 실행 — API 키·무거운 모델 없이 바로 뜬다.
# 실모델을 쓰려면 VLM_PROVIDER/POSE_BACKEND env + requirements.txt의 주석 의존성 해제.
FROM python:3.12-slim

WORKDIR /app

# 코어 + API 의존성만(실모델 의존성은 requirements.txt에서 주석 처리됨)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 합성 포즈 라이브러리는 앱 기동 시 자동 생성(api/app.py lifespan → _ensure_db).
# DB_PATH env로 위치 조정 가능(기본 data/poses.db). 동기화 폴더 금지(SQLite 락).
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
