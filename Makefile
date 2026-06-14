.PHONY: up up-d down logs rebuild shell check test prod-up prod-down hooks

# Docker 컨테이너 실행 (로컬: api — DB는 .env의 OCI 원격 DB 사용)
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

# 컴파일 체크
check:
	docker compose exec api python -m compileall app

# 이미지 재빌드
rebuild:
	docker compose build --no-cache

# --- 프로덕션(Oracle) ---
# 배포 실행
prod-up:
	docker compose -f docker-compose.prod.yml up -d --build

# 배포 종료
prod-down:
	docker compose -f docker-compose.prod.yml down

# --- Git hooks (최초 1회 실행) ---
# 브랜치명 검증(pre-commit) + main push 차단·build 체크(pre-push) 활성화
hooks:
	git config core.hooksPath .githooks
	chmod +x .githooks/*
	@echo "✅ git hooks 활성화됨 (.githooks)"
