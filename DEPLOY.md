# Oracle Cloud Always Free 배포 가이드

`cocktail-mate-server`를 **완전 무료**(Oracle Cloud Always Free)로 배포하는 절차입니다.
스택: **FastAPI + PostgreSQL(pgvector) + Oracle Object Storage(이미지)**, **Docker Compose**로 구동,
인스턴스/네트워크/버킷은 **Terraform**으로 자동 프로비저닝합니다.

> 요약 흐름: 계정 생성 → API 키 발급 → `terraform apply`(인스턴스·버킷 생성) → 출력값으로 `.env` 작성 → `docker compose` 배포 → 검증

---

## 0. 사전 준비물 (로컬 PC)

- [Terraform](https://developer.hashicorp.com/terraform/install) `>= 1.5` (또는 OpenTofu)
- SSH 키 쌍: 없으면 `ssh-keygen -t ed25519 -f ~/.ssh/cocktail_mate` 로 생성
  → 공개키는 `~/.ssh/cocktail_mate.pub`, 개인키는 `~/.ssh/cocktail_mate`
- Git

---

## 1. Oracle Cloud 계정 생성 (수동, 1회)

1. https://www.oracle.com/kr/cloud/free/ 접속 → **무료로 시작하기**.
2. 이메일·국가 입력 후 가입. **결제 카드 인증**이 필요합니다(해외결제 가능 카드). 인증용 소액 결제 후 환불되며,
   **Always Free 리소스만 쓰면 과금되지 않습니다.** (업그레이드하지 않는 한 유료 전환 안 됨)
3. **홈 리전(Home Region)** 선택 시 주의: 가입 후 변경 불가. 한국이면 `대한민국 중부(춘천) / ap-chuncheon-1` 권장.
   (이 가이드 기본값이 `ap-chuncheon-1`)

> ARM(A1) 용량은 인기가 많아 `Out of capacity` 가 날 수 있습니다 → 3단계 `apply` 실패 시 시간을 두고 재시도하거나
> 다른 가용성 도메인/리전을 고려하세요.

---

## 2. API 키 발급 (Terraform 인증값)

OCI 콘솔 우측 상단 프로필 → **내 프로필** → 좌측 **API 키** → **API 키 추가**:

1. **개인 키 생성 후 다운로드** → `~/.oci/oci_api_key.pem` 로 저장 (`chmod 600`).
2. **추가** 누르면 **구성 파일 미리보기**가 뜹니다. 여기서 다음 값을 복사:
   - `user` → `user_ocid`
   - `fingerprint` → `fingerprint`
   - `tenancy` → `tenancy_ocid`
   - `region` → `region`
3. `compartment_ocid` 는 보통 **테넌시 루트 OCID**(= `tenancy_ocid`)를 그대로 사용해도 됩니다.

---

## 3. Terraform 으로 인프라 생성

```bash
cd infra/oracle/terraform
cp terraform.tfvars.example terraform.tfvars
```

`terraform.tfvars` 를 편집해 2단계 값 + SSH 공개키를 채웁니다:

```hcl
tenancy_ocid     = "ocid1.tenancy.oc1..xxxx"
user_ocid        = "ocid1.user.oc1..xxxx"
fingerprint      = "aa:bb:..."
private_key_path = "~/.oci/oci_api_key.pem"
region           = "ap-chuncheon-1"
compartment_ocid = "ocid1.tenancy.oc1..xxxx"
ssh_public_key   = "ssh-ed25519 AAAA... user@host"   # cat ~/.ssh/cocktail_mate.pub
```

실행:

```bash
terraform init
terraform plan      # 생성될 리소스 확인
terraform apply     # yes 입력
```

생성되는 것: VCN/서브넷/게이트웨이/보안리스트, **ARM A1 인스턴스**(Ubuntu, Docker 자동 설치),
**Object Storage 버킷**(`cocktail-images`), **S3 호환 Access/Secret Key**.

> ⚠️ **보안**: `terraform.tfvars`, `*.tfstate`, `*.pem` 에는 비밀값이 들어 있어 `.gitignore`로 제외되어 있습니다.
> state에는 스토리지 Secret이 포함되니 파일을 안전하게 보관하세요(원격 백엔드 사용 권장).

---

## 4. 출력값 확인

```bash
terraform output                 # 일반값
terraform output -raw s3_access_key
terraform output -raw s3_secret_key
```

다음 값을 메모합니다:

| output | 용도(.env) |
|---|---|
| `instance_public_ip` | SSH/접속 IP |
| `s3_endpoint` | `STORAGE_ENDPOINT` |
| `s3_public_base_url` | `STORAGE_PUBLIC_BASE_URL` |
| `object_storage_namespace` | 참고 |
| `bucket_name` | `STORAGE_BUCKET` |
| `s3_access_key` | `STORAGE_ACCESS_KEY` |
| `s3_secret_key` | `STORAGE_SECRET_KEY` |

---

## 5. 서버에 배포

```bash
ssh ubuntu@<instance_public_ip>     # terraform output ssh_command 참고
#   (개인키 지정 시) ssh -i ~/.ssh/cocktail_mate ubuntu@<ip>
```

> Docker는 cloud-init이 자동 설치합니다. `docker ps` 가 동작하지 않으면 1~2분 후(설치 완료 대기) 재시도하거나
> 재로그인하세요(도커 그룹 적용).

서버에서:

```bash
git clone https://github.com/makerspaceSiwoo/cocktail-mate-server.git
cd cocktail-mate-server
git checkout other/cm-32-setup-server     # 머지 전이라면

cp .env.example .env
nano .env                                  # 아래 값 채우기
```

`.env` 편집 — DB 비밀번호는 임의 강한 값으로, 스토리지는 4단계 output 값으로:

```dotenv
APP_ENV=production
POSTGRES_DB=cocktail_mate
POSTGRES_USER=app_user
POSTGRES_PASSWORD=<강한_비밀번호>
DATABASE_URL=postgresql+psycopg2://app_user:<강한_비밀번호>@db:5432/cocktail_mate

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
# 또는: make prod-up
docker compose -f docker-compose.prod.yml logs -f api
```

---

## 6. 검증

```bash
curl http://<instance_public_ip>/health
# 기대: {"db":"ok","vector":"ok","storage":"ok"}
```

- Swagger 문서: `http://<instance_public_ip>/docs`
- 기존 mock 엔드포인트 동작 확인: `/list`, `/search?keyword=진`, `/explore/1`, `/drink-of-the-day` 등

`storage` 가 `fail` 이면 `.env`의 `STORAGE_*` 값/리전, 버킷 이름을 다시 확인하세요.

---

## 7. 운영 / 정리

- **자동 복구**: 컨테이너는 `restart: always` 라 인스턴스 재부팅 시 자동 기동됩니다.
- **업데이트 배포**:
  ```bash
  git pull && docker compose -f docker-compose.prod.yml up -d --build
  ```
- **전체 삭제(과금 방지 확실히)**: 로컬에서 `cd infra/oracle/terraform && terraform destroy`.
- **비용**: 위 리소스(A1 4 OCPU/24GB 한도 내, Object Storage 20GB 한도 내)는 **Always Free**입니다.
  단, 한도를 초과하거나 계정을 유료 업그레이드하면 과금될 수 있으니 콘솔의 **Cost/Usage**를 가끔 확인하세요.

---

## 8. (선택) 도메인 / HTTPS

- 도메인이 있다면 A 레코드를 `instance_public_ip` 로 지정.
- HTTPS는 `api` 앞에 nginx + certbot(Let's Encrypt) 리버스 프록시를 두거나, Caddy 컨테이너로 자동 인증서를 받는 방식을 권장합니다.
  (이 저장소 범위 밖 — 필요 시 별도 구성)
