# cocktail-mate-server

Cocktail Mate 백엔드. 현재는 FastAPI에 `/healthz` 한 개만 있는 최소 스캐폴드.

## 기술 스택

Python 3.11+ · FastAPI · Uvicorn · Ruff · pre-commit

## 폴더 구조

```
cocktail-mate-server/
├── app/
│   ├── __init__.py
│   └── main.py                  # FastAPI 앱 + /healthz
├── pyproject.toml               # 빌드/의존성/Ruff 설정
├── .pre-commit-config.yaml      # Ruff + 브랜치명 검증
├── .vscode/settings.json
└── .python-version
```

## 시작하기

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install --hook-type pre-commit --hook-type pre-push
uvicorn app.main:app --reload --port 8000
```

확인: `curl http://localhost:8000/healthz` → `{"status":"ok"}`

## 주요 명령

| 명령 | 설명 |
| --- | --- |
| `uvicorn app.main:app --reload` | 개발 서버 |
| `ruff check . && ruff format .` | lint + format |
| `pre-commit run --all-files` | 전체 훅 실행 |

## 컨벤션

- **브랜치명 패턴**: `^(main|develop|(feat|fix|hotfix|chore|docs|refactor|test)\/[a-z0-9._-]+)$` (front 레포와 동일). `pre-push` 단계에서 검증.
- **커밋 메시지 컨벤션 검증 없음** (자유 형식)
- **Ruff 일원화** — lint + format. VS Code 사용 시 [.vscode/settings.json](.vscode/settings.json)으로 저장 시 자동 적용 (`charliermarsh.ruff` 확장 설치 필요)
- **버전 단일 진실 소스**: [pyproject.toml](pyproject.toml)의 `version` 필드

## 추후 추가 예정

CORS · Supabase 연결 · 인증 · 임베딩 파이프라인 · 검색 라우터.
