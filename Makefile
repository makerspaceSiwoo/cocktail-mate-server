.PHONY: up up-d require-env down logs rebuild shell check test format format-check prod-up prod-down hooks ssh-check image-preflight image-generate image-logs image-status image-export-prompts image-batch-prepare image-batch-status image-batch-download image-batch-wait image-upload-batch embedding-install embedding-preflight embedding-run embedding-cluster-surface embedding-apply-db taste-query-train

PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

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
up: require-env
	ENV_FILE=$(ENV_FILE) docker compose up --build

# 백그라운드 실행
up-d: require-env
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
	docker compose exec api python -m compileall app scripts

# 단위 테스트 (실제 Gemini/NVIDIA/DB 호출은 mock 처리)
test:
	pytest -q

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

# --- 칵테일 이미지 생성기(프로덕션 서버) ---
image-preflight:
	docker compose -f docker-compose.prod.yml --profile image-generation run --rm image-generator python -m scripts.generate_cocktail_images preflight

image-generate:
	docker compose -f docker-compose.prod.yml --profile image-generation up -d --force-recreate image-generator

image-logs:
	docker compose -f docker-compose.prod.yml --profile image-generation logs -f image-generator

image-status:
	docker compose -f docker-compose.prod.yml --profile image-generation run --rm image-generator python -m scripts.generate_cocktail_images status

image-export-prompts:
	$(PYTHON) -m scripts.export_cocktail_image_prompts

image-batch-prepare:
	$(PYTHON) -m scripts.generate_cocktail_images_batch prepare

image-batch-status:
	$(PYTHON) -m scripts.generate_cocktail_images_batch status

image-batch-download:
	$(PYTHON) -m scripts.generate_cocktail_images_batch download

image-batch-wait:
	$(PYTHON) -m scripts.generate_cocktail_images_batch wait-download

image-upload-batch:
	$(PYTHON) -m scripts.upload_cocktail_images

# --- 로컬 맛 임베딩/축소/3D 실험(프로덕션 API 이미지와 분리) ---
embedding-install:
	$(PYTHON) -m pip install -r requirements-embedding.txt

embedding-preflight:
	$(PYTHON) -m scripts.build_cocktail_embeddings preflight

embedding-run:
	$(PYTHON) -m scripts.build_cocktail_embeddings run-all

embedding-cluster-surface:
	$(PYTHON) -m scripts.build_cocktail_embeddings experiment-cluster-surface

# 기본은 read-only DB 검증. 실제 반영은 직접 --commit을 붙여 실행한다.
embedding-apply-db:
	$(PYTHON) -m scripts.build_cocktail_embeddings apply-db

taste-query-train:
	$(PYTHON) -m scripts.train_taste_query_gnn train

# --- Git hooks (최초 1회 실행) ---
# 브랜치명 검증(pre-commit) + main push 차단·ruff 포맷/린트·build 체크(pre-push) 활성화
hooks:
	git config core.hooksPath .githooks
	chmod +x .githooks/*
	@echo "✅ git hooks 활성화됨 (.githooks)"
