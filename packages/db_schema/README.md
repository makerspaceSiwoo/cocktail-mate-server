# cocktail-mate-db

cocktail-mate의 **canonical DB 레포** (private): SQLAlchemy 모델 + Alembic 마이그레이션 + OCI 인프라(Terraform) + DB 운영.

api-server는 이 패키지를 모노레포 로컬 패키지로 설치해 모델을 공유한다:

```
# cocktail-mate-server/requirements.txt (모노레포 로컬 editable 설치)
-e ./packages/db_schema[migrations]
```

> **이전 방식** (`git+ssh://git@github.com/makerspaceSiwoo/cocktail-mate-db.git@v0.1.0`) 은 모노레포 통합(Task 2) 이후 더 이상 사용하지 않는다.

Alembic 마이그레이션은 `ALEMBIC_DATABASE_URL` (cm_admin 계정, DDL 권한) 환경변수를 사용한다.
런타임 API 컨테이너에는 이 변수가 존재하지 않으며, 마이그레이션 환경(`MIGRATION_ENV_FILE`)에서만 주입된다.

## 구성

| 경로 | 내용 |
|---|---|
| `src/cocktail_mate_db/` | SQLAlchemy 모델 (canonical) — `Base`, `User`, `Cocktail`, `Ingredient`, `CocktailIngredient`, `Like` |
| `alembic/` | 마이그레이션. DDL은 **전부 Alembic 경유** (수동 SQL 금지) |
| `../../docker-compose.db.yml` (레포 루트) | OCI에서 postgres(pgvector) 단독 구동 |
| `infra/oracle/terraform/` | OCI 네트워크/Object Storage. `tfstate`/`tfvars`는 로컬 보관(커밋 금지) |
| `scripts/apply_roles.py` | DB 계정/권한 생성 (`cm_app`/`cm_dml`) — 비번은 `.env`에서 읽음 |
| `src/cocktail_mate_db/config.py` | 전역 설정 `settings` (`.env` → `process.env`처럼 접근) |
| `scripts/bootstrap-admin.sql` | 기존 슈퍼유저(app_user) → `cm_admin` 전환 절차 |
| `scripts/legacy/` | Alembic 도입 전 init SQL (참고용 보관) |

## DB 계정

| 계정 | 용도 | DDL |
|---|---|---|
| `cm_admin` | Alembic 마이그레이션 전용 (객체 소유자) | ✅ |
| `cm_app` | FastAPI 런타임 (api-server `.env`) | ❌ |
| `cm_dml` | 사람용 — DBeaver 등에서 데이터 조회/수정 | ❌ |

- **모든 계정 비밀번호는 20자+ 랜덤** (`openssl rand -base64 24`). 5432가 인터넷에 공개되어 있어 비밀번호가 유일한 방어선이다.
- Alembic은 **반드시 `cm_admin`으로** 실행 — 새 테이블에 대한 cm_app/cm_dml 자동 권한 부여(default privileges)가 cm_admin 생성 객체에만 적용된다.

## 온보딩 (모델 수정/마이그레이션 담당자)

1. 이 레포 클론 (private — GitHub 접근 권한 필요)
2. `.env` 파일을 팀 채널에서 받아 레포 루트에 저장 (`.env.example` 참고)
3. `make install` (= `pip install -e ".[migrations]"`)

DBeaver로 데이터만 볼 거라면 이 레포는 필요 없다 — Host `<OCI_PUBLIC_IP>`, Port `5432`, DB `cocktail_mate`, 계정 `cm_dml`로 바로 접속 (DDL이 막혀 있어 실수로 스키마를 못 건드림).

## 스키마 변경 (DDL) 절차

```bash
# 1) src/cocktail_mate_db/models/ 에서 모델 수정
# 2) 마이그레이션 자동 생성 → 생성된 파일 반드시 리뷰
make migrate-new m="add users.bio"
# 3) 원격 개발 DB에 적용 (cm_admin)
make migrate-up
# 4) 커밋 + 태그 업 (api-server requirements가 태그 고정이므로)
git tag v0.x.y && git push --tags
```

- pgvector: `script.py.mako`가 `import pgvector.sqlalchemy`를 포함하므로 autogenerate가 Vector 컬럼을 렌더링할 수 있다. 새 DB를 처음 만들 때만 baseline의 `CREATE EXTENSION vector`가 실행된다.
- 임베딩 차원은 `cocktail_mate_db.EMBEDDING_DIM`(64) 하나로 관리.

## DB 운영 (OCI)

```bash
# 구동 (기존 cocktail-mate-server_postgres_data 볼륨을 external로 인계)
make db-up
```

5432는 보안 리스트 + compose 전체 바인딩으로 공개되어 있다 (개발 단계 트레이드오프 —
실사용자 데이터가 들어오면 `127.0.0.1:5432` 바인딩 + 보안 리스트에서 5432 제거로 되돌린다).

## Terraform

```bash
cd infra/oracle/terraform
terraform init && terraform plan
```

- `terraform.tfvars`/`terraform.tfstate`는 로컬에만 존재 (커밋 금지, 담당자 간 수동 공유)
- **compute 인스턴스는 콘솔에서 수동 생성되어 state에 없다** — Terraform은 네트워크/Object Storage만 관리
