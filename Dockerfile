FROM python:3.12-slim

WORKDIR /app

# psycopg2-binary, boto3 등은 wheel로 설치되어 빌드 도구가 필요 없다.
COPY requirements.txt .
# 로컬 db 패키지(-e ./packages/db_schema)는 pip install 전에 소스가 있어야 설치된다.
COPY packages/db_schema/ packages/db_schema/

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# OCI는 8000, Render는 $PORT 로 뜬다. (shell 형식이라야 ${PORT} 가 확장된다)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
