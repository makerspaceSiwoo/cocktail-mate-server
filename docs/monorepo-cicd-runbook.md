# Monorepo CI/CD Runbook — Category B (Live Infrastructure)

> **Scope:** Steps that require live credentials, cloud console access, or real secret values.
> None of these steps were executed during the in-repo migration (Tasks 1–9). They must be run
> once — in the order shown — before the first production deploy.
>
> **Secret hygiene:** No secret *values* appear in this file. Only names and placeholders.

---

## A. GitHub Environments & Secrets

### A1. Environment setup

Create two environments in **Settings → Environments**:

| Environment | Required reviewers | Purpose |
|---|---|---|
| `production` | yes — at least 1 (owner) | OCI deploy + Terraform apply gate |
| `develop` | optional | Render develop deploy |

### A2. Secrets — `production` environment

| Secret name | Content |
|---|---|
| `APP_ENV_FILE` | Full dotenv content for the **API runtime** container (`cm_app` role). Every key in `.env.example` must be present. `DATABASE_URL` must use the `cm_app` role (no DDL). |
| `MIGRATION_ENV_FILE` | Dotenv containing `ALEMBIC_DATABASE_URL` = cm_admin 접속 URL (형식은 `packages/db_schema/.env.example` 참고) 및 migration env에 필요한 기타 키. The `cm_admin` credential stays here — never in `APP_ENV_FILE`. |
| `OCI_HOST` | Public IP of the OCI compute instance |
| `OCI_USER` | SSH login user on the OCI instance (e.g. `ubuntu`) |
| `OCI_SSH_KEY` | Private key whose public counterpart is in `~/.ssh/authorized_keys` on the instance |

### A3. Secrets — `develop` environment

| Secret name | Content |
|---|---|
| `APP_ENV_FILE` | Full dotenv for the Render service (cm_app role, develop DB). PUT replaces ALL env vars — every key the app needs must be present. |
| `ALEMBIC_DATABASE_URL` | `cm_admin` URL for the **develop** PostgreSQL DB (pgvector enabled, runner-reachable). |
| `RENDER_API_KEY` | Render API key with write access to the service |
| `RENDER_SERVICE_ID` | Service ID from the Render dashboard URL |
| `RENDER_DEPLOY_HOOK_URL` | Deploy hook URL from Render service settings (Settings → Deploy Hook) |
| `RENDER_HEALTH_URL` | Publicly reachable health endpoint, e.g. `https://<service>.onrender.com/health` |

### A4. Repo-level secrets (not environment-scoped)

The Terraform **plan** job runs without the `production` environment gate (PRs must trigger it), so these two secrets must live at **repo level** (Settings → Secrets and variables → Actions → Repository secrets):

| Secret name | Content |
|---|---|
| `TERRAFORM_TFVARS_JSON` | Rendered JSON of `terraform.tfvars` (see `terraform/oci/terraform.tfvars.example`) |
| `TERRAFORM_BACKEND_CONFIG` | Rendered `backend.hcl` content (see `terraform/oci/backend.hcl.example`) |

### A5. Delete obsolete secret

Delete `CMDB_DEPLOY_KEY` (the former deploy key for the now-merged cocktail-mate-db private repo).
It is no longer referenced anywhere in the codebase after the monorepo integration.

---

## B. Branch Protection & Actions Settings

### B1. Required status checks

Enable the following on **both** `main` and `develop`:

- **Require status checks to pass before merging** → add the required status checks: `lint`, `schema`, `docker-build` from `.github/workflows/ci.yml`.
  This is the enforcement mechanism for "deploy only after CI passes" — `deploy.yml` is push-triggered
  and will run immediately after merge, so CI must gate the merge itself.

### B2. Code owner review

- Enable **"Require review from Code Owners"** (Settings → Branches → branch rule).
  `.github/CODEOWNERS` assigns `@makerspaceSiwoo` to `/packages/db_schema/`, `/terraform/`, and
  `/.github/workflows/`. Any PR touching these paths will require the owner's explicit approval.

### B3. Fork/outside-collaborator PRs (public repo)

- Settings → Actions → General → **"Require approval for all outside collaborators and fork PRs"**.
  
  Rationale: `terraform.yml` runs `terraform plan` on every PR, executing runner-side code. A malicious
  fork PR could embed a `local-exec` provisioner to exfiltrate secrets. This setting prevents unknown
  contributors from triggering plan without maintainer approval.

---

## C. Terraform Remote State & First Apply

Complete these steps **in order** before triggering the `apply` workflow dispatch.

### C1. Create the OCI Object Storage bucket for remote state

The bucket is NOT managed by this Terraform config (bootstrapping problem). Create it manually in
the OCI Console or with a separate one-off CLI call. Note the bucket name and namespace — they go
into `TERRAFORM_BACKEND_CONFIG`.

### C2. Render the backend config and init

```bash
# Fill in backend.hcl from the example; this file is gitignored (public repo)
cp terraform/oci/backend.hcl.example terraform/oci/backend.hcl
# Edit backend.hcl with real bucket name, namespace, region, user OCID, key fingerprint, key path

cd terraform/oci
# 1) 레거시 로컬 state를 작업 디렉터리로 복사 (terraform init 이 이 파일을 원격 backend로 이관한다)
cp ../../cocktail-mate-db/infra/oracle/terraform/terraform.tfstate terraform/oci/terraform.tfstate
cd terraform/oci
# 2) 원격 backend로 init + state 이관 (프롬프트에 yes)
terraform init -backend-config=<렌더된 backend.hcl> -migrate-state
```

> **CAUTION:** 이관 완료 및 `terraform import`(C3) 후, `terraform plan`을 실행해 `0 to destroy / 0 to replace` 임을 반드시 확인한 뒤에만 apply를 진행하십시오 (C4 게이트 참고).

> **Note:** After this, local `terraform init` (without `-backend-config`) will fail. Either supply
> `-backend-config=backend.hcl` or use `-backend=false` for offline validation only.

### C3. Import the live compute instance

The OCI compute instance was created manually and is **not in Terraform state**. Without import,
`terraform plan` will show "1 to add" and attempt to create a second instance on apply.

```bash
# Replace <INSTANCE_OCID> with the real OCID from the OCI Console
terraform import oci_core_instance.server <INSTANCE_OCID>
```

### C4. Verify plan before first apply

```bash
terraform plan -var-file="$RUNNER_TEMP/terraform.tfvars.json" -out=tfplan
```

**Confirm the output shows `0 to destroy` and `0 to replace` before proceeding.**
Any destroy/replace on `oci_core_instance.server` means the import did not succeed — stop and diagnose.

### C5. Set required reviewers for the `production` environment

In Settings → Environments → `production`, add at least one required reviewer. The `apply` job in
`terraform.yml` is gated on this environment — no `apply` runs without explicit approval.

### C6. Trigger apply via workflow_dispatch

Go to Actions → `Terraform` → Run workflow → `action: apply`. The job will wait for environment
approval before running `terraform apply -auto-approve -no-color tfplan`.

---

## D. main → OCI Cutover

### Pre-conditions

- Section A secrets set for `production`.
- `CMDB_DEPLOY_KEY` deleted (Section A5).
- OCI instance is running; PostgreSQL (`cm_admin` accessible at `host.docker.internal:5432`) is up.
- Terraform state is correct (Section C).

### Deploy flow (automatic on `main` push)

1. **CI** (`ci.yml`): ruff lint, compile check, Alembic schema check — must pass before merge.
2. **Build** (`deploy.yml` `build` job): build `linux/arm64` image, push to `ghcr.io/makerspacesiwoo/cocktail-mate-server:latest` and `:<sha>`.
3. **Deploy** (`deploy.yml` `deploy` job, `environment: production`):
   a. SSH to OCI → pull new image.
   b. Upload runtime `.env` from `APP_ENV_FILE` secret.
   c. Write `MIGRATION_ENV_FILE` to `$RUNNER_TEMP` temp file (trap-cleaned on exit).
   d. Run one-off migration container: `alembic upgrade head` — failure stops the job (API is not restarted).
   e. `docker compose up -d` → poll `/health` for up to 3 minutes.
   f. On health failure: roll back to previous image (`docker compose up -d` with previous tag).

### Rollback

The previous image tag (`:<prev_sha>`) is preserved in GHCR. Set the `API_IMAGE` compose variable to
the previous tag and re-run `docker compose up -d` on the instance.

---

## E. develop → Render Setup

### E1. Create the Render service

Create an **IMAGE-BACKED** service in Render:
- Image URL: `ghcr.io/makerspacesiwoo/cocktail-mate-server`
- Auto-deploy: **OFF** (the deploy workflow controls deploys via hook).
- Note the Service ID from the dashboard URL (it appears as `srv-...`).

### E2. GHCR pull credentials

The GHCR image may be private. In Render service settings → Registry Credentials, add:
- Registry: `ghcr.io`
- Username: GitHub username (e.g. `makerspaceSiwoo`)
- Password: a GitHub PAT with `read:packages` scope.

### E3. Verify imgURL base path

The deploy hook uses `?imgURL=ghcr.io/makerspacesiwoo/cocktail-mate-server:<sha>`.
The Render service's configured image URL **must** be `ghcr.io/makerspacesiwoo/cocktail-mate-server`
(without tag) — the hook param only changes the tag/digest. Confirm in Render Settings before first deploy.

### E4. develop PostgreSQL DB

- Enable **pgvector** extension on the develop DB.
- Ensure the DB is **reachable from GitHub-hosted runners** (public endpoint or runner IP allowlist).
- Provide the `cm_admin` connection URL as `ALEMBIC_DATABASE_URL` in the `develop` GitHub Environment.

### E5. APP_ENV_FILE completeness

The Render env-vars `PUT /v1/services/{id}/env-vars` endpoint **replaces ALL env vars atomically**.
Any key not in `APP_ENV_FILE` will be removed after deploy. Verify the file contains every required
runtime key — including any Render-injected vars (e.g. `PORT`) you want to preserve.

### E6. Verify before first deploy

1. Check the `RENDER_DEPLOY_HOOK_URL` format: if it already contains `?`, the workflow appends `&imgURL=...`; otherwise `?imgURL=...`. The `if echo ... | grep -q '?'` check in the workflow handles both cases.
2. Optionally call the Render env-vars API manually with a test payload to confirm auth and endpoint.
3. Source: Render API docs — `PUT https://api.render.com/v1/services/{serviceId}/env-vars` with `Authorization: Bearer <RENDER_API_KEY>` and JSON body `[{"key": "...", "value": "..."}]`.

### Deploy flow (automatic on `develop` push)

1. **CI** passes.
2. **build-develop** job: build `linux/amd64` image → push `ghcr.io/makerspacesiwoo/cocktail-mate-server:develop` and `:<sha>`.
3. **deploy-develop** job (`environment: develop`):
   a. Checkout + install `packages/db_schema[migrations]`.
   b. Run `alembic upgrade head` against `ALEMBIC_DATABASE_URL` (develop) — failure stops job.
   c. Sync Render env vars from `APP_ENV_FILE` (PUT replaces all).
   d. Trigger deploy hook with `imgURL=...:$sha`.
   e. Health check: 12 × 15s retry against `RENDER_HEALTH_URL`.

---

## F. Deploy Scenario Tests

Run these three scenarios before treating the pipeline as production-ready.

### F1. Schema-only change

Modify a model in `packages/db_schema/`, generate and commit a migration. Push to `develop` (or `main`).
Expected: migration runs and succeeds before the Render (or OCI) deploy starts. API continues to serve
with the new schema.

### F2. Server-only change

Modify application code only (no migration). Push to branch.
Expected: `alembic upgrade head` is a no-op (already at head); deploy proceeds normally.

### F3. Schema + server change

Combine a new migration with a new API endpoint that depends on it. Push.
Expected: migration runs first, API deploys second — the new endpoint is never live on the old schema.

---

## G. Pre-Import / Ongoing Security

### G1. gitleaks scan (before finalizing merge)

```bash
# Run on full history of both repos before the branch merges to main
gitleaks detect --source . -v
# If scanning the former cocktail-mate-db repo history
gitleaks detect --source ../cocktail-mate-db -v
```

Expected: 0 findings. The gitleaks CI job (`.github/workflows/gitleaks.yml`) runs on every push — this
manual scan is an extra gate before the merge that closes out the monorepo migration.

### G2. Ongoing gitleaks CI

The `gitleaks.yml` workflow scans every push. Do not remove or disable it. If a finding appears, rotate
the exposed credential immediately, then add a `.gitleaksignore` entry only for confirmed false positives.

---

## H. Recommended Follow-ups (Non-blocking Deferred Minors)

These items were noted during code review but are not blocking the first deploy.

| Item | Detail |
|---|---|
| CI double-run on PR branches | `on: [pull_request, push]` triggers ci.yml twice for PR branches. Scope `push` to `branches: [main, develop]` to eliminate the duplicate. |
| Pin `ruff` version | `pip install ruff` in ci.yml installs the latest. Pin to the verified version (e.g. `ruff==0.x.y`) for reproducible lints. |
| SHA-pin GitHub Actions | Actions are currently pinned to major version tags (e.g. `actions/checkout@v4`). Pin to commit SHAs for supply-chain safety (advisory — not a blocker for a private/controlled workflow). |
| Split concurrency groups | `concurrency.group: oci-deploy` covers all jobs in deploy.yml. Concurrent `develop` pushes queue behind `main` deploys. Split into separate groups (`oci-deploy` / `render-deploy`) when needed. |
| Guard workflow_dispatch on non-target branches | Triggering the deploy workflow_dispatch from a non-`main`/`develop` branch is currently a silent no-op. Add an explicit guard or error message for clarity. |
| Prefer `python-dotenv` for Render env conversion | The dotenv→JSON conversion in deploy-develop uses a shell `printf` loop. Edge cases with embedded quotes in values may cause parse errors. Replace with `python-dotenv` load for robustness; document that `APP_ENV_FILE` values must not contain embedded shell quotes. |
| `echo "$GHCR_TOKEN" \| docker login` | GitHub Actions automatically masks secret values in logs. This pattern is safe/accepted but can be replaced with `--password-stdin < <(printf '%s' "$GHCR_TOKEN")` if stricter pipe hygiene is desired. |
| CODEOWNERS branch protection | `.github/CODEOWNERS` is present but has no enforcement until "Require review from Code Owners" is enabled in branch protection (Section B). Enable this for `main` as a priority. |

---

## I. DB compose 운영권 OCI 이전 (무중단 인계 절차)

> **언제 실행하는가:** `cocktail-mate-db` 클론이 DB 스택을 구동 중인 OCI 인스턴스에서,
> 운영 주체를 `cocktail-mate-server`로 전환할 때 **수동으로 1회** 실행한다.
>
> **왜 데이터 손실이 없는가:** `docker-compose.db.yml`의 볼륨이 `external: true`,
> `name: cocktail-mate-server_postgres_data`로 선언되어 있다.
> external 볼륨은 `docker compose down`으로도 삭제되지 않으며, 이름이 고정되어 있어
> 실행 위치가 바뀌어도 항상 동일한 볼륨에 붙는다.

### 사전 확인 (데이터 보존 근거)

```bash
# 볼륨이 현재 존재하는지 확인
docker volume ls | grep postgres
# → cocktail-mate-server_postgres_data 가 목록에 있어야 함

# 컨테이너가 이 볼륨을 실제로 쓰는지 확인
docker inspect postgres-db --format '{{range .Mounts}}{{.Name}}{{end}}'
# → cocktail-mate-server_postgres_data 출력되면 OK
```

### DB 전용 env 준비 (API `.env`와 완전히 별개)

OCI 호스트에 `~/cocktail-mate-server/.env.db` 를 생성한다:

```
POSTGRES_DB=cocktail_mate
POSTGRES_USER=cm_admin
POSTGRES_PASSWORD=<cm_admin 비번>
```

> ⚠️ **절대 섞지 말 것:** deploy.yml은 API용 `.env`(cm_app 계정)를 자동으로 덮어쓴다.
> DB 전용 env는 반드시 `.env.db`처럼 별도 파일을 쓰고, 기동 시 `--env-file .env.db`를 명시해야 한다.
> compose가 자동 로드하는 `.env`(API env)와 혼동하면 DB에 잘못된 계정/비밀번호가 전달될 수 있다.
>
> (기존 볼륨이 이미 초기화되어 있다면 POSTGRES_* 환경변수는 초기화에 사용되지 않고 무시된다.
> 새 볼륨일 때만 초기화에 쓰인다.)

### 옛 실행 주체 정지

```bash
cd ~/cocktail-mate-db
docker compose -f docker/docker-compose.db.yml down
# external 볼륨은 down으로 삭제되지 않음 — 데이터 유지 확인됨
```

### server 레포로 기동

```bash
cd ~/cocktail-mate-server
git checkout main && git pull
docker compose --env-file .env.db -f docker-compose.db.yml up -d
```

### 검증

```bash
# 컨테이너 및 DB 응답 확인
docker exec postgres-db pg_isready -U cm_admin

# API 헬스체크 (db: ok 확인)
curl http://localhost:8000/health   # OCI 내부; 또는 공인 IP/health

# 핵심 테이블 row 수가 이전과 동일한지 확인 (예)
docker exec postgres-db psql -U cm_admin -d cocktail_mate -c "SELECT COUNT(*) FROM cocktails;"
```

### (선택) 정리

검증이 완료되면 `~/cocktail-mate-db` 클론은 더 이상 DB 구동에 필요하지 않다 (삭제 가능).

> **참고 — postgres는 수동 운영 전용:**
> postgres 컨테이너는 stateful하므로 CI/deploy.yml이 관리하지 않는다.
> 위 절차는 **수동 1회 이전**이며, 이후에도 DB는 수동으로 운영한다.
> deploy.yml은 API(FastAPI) + Caddy 컨테이너만 pull/up한다.

---

*Runbook synthesized from Tasks 6–8b, 10. Last updated: 2026-07-17.*
