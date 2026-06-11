# Oracle Cloud 배포 가이드 (api-server)

`cocktail-mate-server`(FastAPI + Caddy)를 Oracle Cloud Always Free 인스턴스에 배포하는 절차입니다.

> **인프라(Terraform)·DB(PostgreSQL/pgvector)는 이 레포 범위가 아닙니다** —
> [cocktail-mate-db](https://github.com/makerspaceSiwoo/cocktail-mate-db) (private) 레포가
> VCN/보안리스트/Object Storage 프로비저닝과 DB 구동·스키마(Alembic)를 담당합니다.
> 인스턴스 생성/네트워크/버킷/DB 기동은 그 레포의 README를 먼저 따라오세요.

> 요약 흐름: (db 레포에서 인프라·DB 준비) → 서버에 클론 → Deploy Key 등록 → `.env` 작성 → `docker compose` 배포 → 검증

---

## 0. 전제

- OCI 인스턴스가 떠 있고 SSH 접속 가능 (`ssh ubuntu@<instance_public_ip>`)
- 같은 인스턴스에서 DB가 구동 중 (db 레포의 `docker-compose.db.yml`, 5432)
- Object Storage 버킷 + S3 호환 키 발급 완료 (db 레포 terraform output)

## 1. Deploy Key (private 패키지 설치용, 1회)

api 이미지는 빌드 중 `cocktail-mate-db`(private)를 pip로 설치하므로, 서버에 **read-only Deploy Key**가 필요합니다:

```bash
# 서버에서 키 생성
ssh-keygen -t ed25519 -f ~/.ssh/cm_db_deploy -N "" -C "cocktail-mate-db deploy"
cat ~/.ssh/cm_db_deploy.pub
# → GitHub cocktail-mate-db 레포 Settings → Deploy keys → Add (read-only)

# 빌드 전 ssh-agent에 등록 (재로그인 시마다)
eval "$(ssh-agent -s)" && ssh-add ~/.ssh/cm_db_deploy
```

> PAT 방식보다 단순하고, 키가 이미지 레이어에 남지 않습니다(BuildKit ssh mount).

## 2. 서버에 배포

```bash
git clone https://github.com/makerspaceSiwoo/cocktail-mate-server.git
cd cocktail-mate-server
cp .env.example .env
nano .env
```

`.env` — DB는 같은 인스턴스의 5432로 (자기 공인 IP는 hairpin 불가 → `host.docker.internal`):

```dotenv
APP_ENV=production
CORS_ORIGINS=https://cocktail-mate-front.vercel.app
DATABASE_URL=postgresql+psycopg2://cm_app:<비밀번호>@host.docker.internal:5432/cocktail_mate

STORAGE_ENDPOINT=<s3_endpoint>
STORAGE_REGION=ap-chuncheon-1
STORAGE_ACCESS_KEY=<s3_access_key>
STORAGE_SECRET_KEY=<s3_secret_key>
STORAGE_BUCKET=cocktail-images
STORAGE_PUBLIC_BASE_URL=<s3_public_base_url>
```

기동:

```bash
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml logs -f api
```

## 3. 검증

```bash
curl https://<도메인>/health
# 기대: {"db":"ok","vector":"ok","storage":"ok"}
```

- `db: fail` → DB 컨테이너 구동 여부(`sudo docker ps`), `.env`의 비밀번호/`host.docker.internal` 확인
- `storage: fail` → `.env`의 `STORAGE_*` 값/리전, 버킷 이름 확인
- 빌드가 `Permission denied (publickey)`로 실패 → ssh-agent에 Deploy Key 등록 여부 확인(`ssh-add -l`)

## 4. 운영

- **자동 복구**: 컨테이너는 `restart: always` — 인스턴스 재부팅 시 자동 기동
- **업데이트 배포**:
  ```bash
  ssh-add ~/.ssh/cm_db_deploy   # 등록 안 돼 있으면
  git pull && docker compose -f docker-compose.prod.yml up -d --build
  ```
- **모델 패키지 버전 업**: `requirements.txt`의 `cocktail-mate-db @ ...@vX.Y.Z` 태그 변경 후 재빌드
- **인프라 삭제/변경**: db 레포의 terraform에서 (이 레포에는 인프라 코드 없음)
