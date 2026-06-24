FROM python:3.12-slim

WORKDIR /app

# cocktail-mate-db(private 레포) 설치용 git + ssh 클라이언트.
# 키는 이미지에 굽지 않고 BuildKit ssh mount로 빌드 시에만 전달한다
# (빌드: compose의 ssh: ["default"] — 호스트 ssh-agent에 GitHub 키 필요).
RUN apt-get update && apt-get install -y --no-install-recommends git openssh-client \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# psycopg2-binary, boto3 등은 wheel로 설치되어 빌드 도구가 필요 없다.
COPY requirements.txt .

RUN --mount=type=ssh pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
