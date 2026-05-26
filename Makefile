.PHONY: up down logs rebuild shell check

# Docker 컨테이너들 실행
up:
	docker compose up --build

# 백그라운드 실행
up-d:
	docker compose up -d --build

# 종료
down:
	docker compose down

# pytest 실행
test:
	docker compose exec api pytest
	
# 로그 보기
logs:
	docker compose logs -f

# api 컨테이너 접속
shell:
	docker compose exec api bash

# mysql 접속(비밀번호: "app_password")
db-shell:
	docker compose exec db mysql -u app_user -p app_db

# 컴파일 체크
check:
	docker compose exec api python -m compileall app

# 이미지 재빌드
rebuild:
	docker compose build --no-cache