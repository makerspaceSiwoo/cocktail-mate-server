.PHONY: up up-d require-env down logs rebuild shell check format format-check prod-up prod-down hooks ssh-check

# 빌드 전 GitHub SSH 인증 점검 (cocktail-mate-db private 레포 설치 전제)
ssh-check:
	@bash scripts/check-ssh.sh

# env 파일 자동 선택: .env.local 우선 → 없으면 .env (둘 다 없으면 빈 값 → 타깃에서 에러).
# 각 파일은 self-contained(모든 키). 개발자는 .env.local 만, 배포 서버는 .env 만 두면 된다.
ENV_FILE := $(shell if [ -f .env.local ]; then echo .env.local; elif [ -f .env ]; then echo .env; fi)

# 실행 전 env 파일 존재 확인 (둘 다 없으면 중단)
require-env:
	@test -n "$(ENV_FILE)" || { echo "❌ .env.local 도 .env 도 없습니다 — 하나를 만들어 주세요 (.env.example 참고)."; exit 1; }
	@echo "▶ env_file: $(ENV_FILE)"

# Docker 실행 (foreground)
up: ssh-check require-env
	ENV_FILE=$(ENV_FILE) docker compose up --build

# 백그라운드 실행
up-d: ssh-check require-env
	ENV_FILE=$(ENV_FILE) docker compose up -d --build

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
