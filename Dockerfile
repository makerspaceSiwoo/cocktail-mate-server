FROM python:3.12-slim

WORKDIR /app

# psycopg2-binary, boto3 등은 wheel로 설치되어 빌드 도구가 필요 없다.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
