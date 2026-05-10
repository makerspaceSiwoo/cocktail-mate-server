# cocktail-mate-server

Cocktail Mate 백엔드 API 서버. FastAPI 기반으로 작성되며, 추후 Supabase(Postgres + pgvector)와 연동하여 칵테일 추천/검색 기능을 제공합니다.

## 기술 스택

| 항목 | 내용 |
| --- | --- |
| 언어 | Python 3.11+ |
| 프레임워크 | FastAPI |
| 서버 | Uvicorn (standard) |
| 린터/포매터 | Ruff (lint + format) |
| Git Hooks | pre-commit |

## 폴더 구조

```
cocktail-mate-server/
├── app/                       # 애플리케이션 패키지
│   ├── __init__.py            # __version__ (APP_VERSION env var) 노출
│   ├── main.py                # FastAPI 앱 엔트리포인트
│   └── api/
│       └── routers/
│           └── health.py      # /healthz 라우터
├── pyproject.toml             # 빌드/의존성/Ruff 설정
├── requirements.txt           # 런타임 의존성 (참고용)
├── requirements-dev.txt       # 개발 의존성 (참고용)
├── .pre-commit-config.yaml    # Ruff + 브랜치명 검증 훅
├── .vscode/
│   └── settings.json          # 저장 시 Ruff 자동 적용
├── .python-version            # 3.11 고정
└── .env.example               # 환경 변수 예시
```

## 시작하기

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install --hook-type pre-commit --hook-type pre-push
uvicorn app.main:app --reload --port 8000
```

`pip install -e ".[dev]"`은 [pyproject.toml](pyproject.toml)의 런타임 의존성과 `dev` extra(개발 의존성)를 함께 설치합니다.

## 주요 명령

| 명령 | 설명 |
| --- | --- |
| `uvicorn app.main:app --reload` | 개발 서버 실행 (http://localhost:8000) |
| `ruff check .` | 린트 검사 |
| `ruff format .` | 코드 포매팅 |
| `pre-commit run --all-files` | 전체 파일에 대해 pre-commit 훅 실행 |

## 환경 변수

[.env.example](.env.example) 참고.

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `APP_VERSION` | `0.1.0` | `/healthz` 응답에 노출되는 버전 문자열 |

> Supabase 관련 변수(`SUPABASE_URL`, `SUPABASE_ANON_KEY` 등)는 추후 작업에서 추가됩니다.

## 컨벤션

- **브랜치명 패턴**: `^(main|develop|(feat|fix|hotfix|chore|docs|refactor|test)\/[a-z0-9._-]+)$` (front 레포와 동일)
  - 예: `feat/health-endpoint`, `fix/typo`, `chore/ci-update`
  - `pre-push` 단계에서 [.pre-commit-config.yaml](.pre-commit-config.yaml)의 `validate-branch-name` 훅이 검사
- **커밋 메시지 컨벤션 검증 없음** — 메시지는 자유 형식
- **Ruff 일원화**: lint와 format을 모두 Ruff가 처리. VS Code 사용 시 [.vscode/settings.json](.vscode/settings.json) 적용으로 저장 시 자동 fix + import 정렬이 동작 (`charliermarsh.ruff` 확장 설치 필요)
- **버전 단일 진실 소스**: [pyproject.toml](pyproject.toml)의 `version`이 단일 소스. `app/__init__.py`는 `APP_VERSION` 환경 변수를 읽어 노출하며, 미설정 시 `0.1.0`을 기본값으로 사용

## API 엔드포인트 (현재)

| Method | Path | 응답 |
| --- | --- | --- |
| GET | `/healthz` | `{"status": "ok", "version": "<APP_VERSION>"}` |
