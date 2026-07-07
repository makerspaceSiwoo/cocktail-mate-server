.PHONY: up up-d down logs rebuild shell check format format-check prod-up prod-down hooks ssh-check

# 빌드 전 GitHub SSH 인증 점검 (cocktail-mate-db private 레포 설치 전제)
ssh-check:
	@bash scripts/check-ssh.sh

# Docker 컨테이너 실행 (로컬: api — DB는 .env의 OCI 원격 DB 사용)
up: ssh-check
	docker compose up --build

# 백그라운드 실행
up-d: ssh-check
	docker compose up -d --build

# 종료
down:
	docker compose down

# 로그 보기
logs:
	docker compose logs -f

# api 컨테이너 접속
shell:
	docker compose exec api bash

# 컴파일 체크
check:
	docker compose exec api python -m compileall app

# 코드 포맷팅 (ruff — 로컬 venv/시스템 ruff 사용). 자동 정리.
format:
	ruff format . && ruff check --fix .

# 포맷/린트 검사만 (수정 없음) — pre-push 훅과 동일 기준
format-check:
	ruff format --check . && ruff check .

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
# 브랜치명 검증(pre-commit) + main push 차단·ruff 포맷/린트·build 체크(pre-push) 활성화
hooks:
	git config core.hooksPath .githooks
	chmod +x .githooks/*
	@echo "✅ git hooks 활성화됨 (.githooks)"
