# 개발 온보딩

SSH 터널, VPN, 로컬 DB 전부 불필요. 두 단계면 끝난다.

## 1. `.env` 받기

팀 채널(비공개 — Notion/Discord 등)에서 `.env` 파일을 받아 레포 루트에 저장한다.

> ⚠️ 이 레포는 **public**이다. `.env`는 절대 커밋 금지 (`.gitignore`에 등록돼 있음).
> 공유도 반드시 비공개 채널로만.

## 2. 실행

```bash
docker compose up -d --build
curl localhost:8000/health   # {"db":"ok","vector":"ok","storage":"ok"} 이면 끝
```

api 컨테이너가 `.env`의 OCI 개발 DB(공인 IP)에 바로 연결된다.

### 빌드 전제: 인접 DB 저장소

로컬 Docker 빌드는 아직 커밋하거나 태그하지 않은 ORM 변경도 함께 검증할 수 있도록
인접한 `cocktail-mate-db` 소스를 직접 설치한다. 두 저장소가 다음 구조로 체크아웃돼
있어야 한다.

```text
cocktail-mate/
├── cocktail-mate-server/
└── cocktail-mate-db/
```

`make up`에는 GitHub SSH agent가 필요하지 않다. 배포 이미지만 `requirements.txt`에
고정된 DB 패키지 Git 태그를 사용한다.

## DB 직접 보기 (DBeaver)

새 연결 → PostgreSQL:

| 항목 | 값 |
|---|---|
| Host | `<OCI_PUBLIC_IP>` (.env의 DATABASE_URL과 동일) |
| Port | `5432` |
| Database | `cocktail_mate` |
| 계정 | `cm_dml` (비밀번호는 팀 채널) |

`cm_dml`은 DML 전용이라 실수로 테이블을 만들거나 지울 수 없다 (DDL 차단).
스키마 변경이 필요하면 [cocktail-mate-db](https://github.com/makerspaceSiwoo/cocktail-mate-db) 레포에서 Alembic으로.

## 자주 묻는 것

- **모델(ORM)은 어디에?** → `cocktail-mate-db` 패키지. `from cocktail_mate_db.models import Cocktail` 또는 기존처럼 `from app.core.database import Base`.
- **로컬에서 모델이 바뀌면?** → 인접 DB 저장소를 수정하고 `make up`으로 함께 빌드한다.
- **배포 모델이 바뀌면?** → DB 저장소의 새 태그를 배포하고 `requirements.txt`의 태그를 올린다.
- **이미지 스토리지는?** → 팀 `.env`가 Oracle Object Storage를 가리킨다. 오프라인 작업은 `--profile local-storage`로 MinIO 기동.
