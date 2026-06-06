# AWS Free Tier 배포 가이드

`cocktail-mate-server`를 AWS Free Tier(EC2 1대 + S3)에 Docker로 배포합니다.
DB는 RDS 대신 **EC2 안에서 Docker로 PostgreSQL+pgvector**를 함께 띄워 완전 무료를 유지합니다.

스택: EC2 `t2.micro`(Ubuntu, Docker, swap 2GB) + S3(이미지) + `docker-compose.prod.yml`

> ⚠️ **Free Tier 12개월 주의**: EC2 750시간/월 무료는 **신규 계정 가입 후 12개월** 한정입니다.
> 12개월이 지난 계정이면 `t2.micro`도 과금(월 ~$8-10)됩니다. 아래 0단계에서 먼저 확인하세요.

---

## AWS 콘솔에서 미리 준비할 것

### 0. Free Tier 자격 확인 (제일 먼저)
- 콘솔 → **Billing and Cost Management** → **Free Tier** 페이지
- "EC2 750 Hrs" 항목이 보이고 사용량 추적이 되면 = 12개월 이내(무료 대상) ✅
- 안 보이거나 계정이 12개월 초과면 → 과금됨. **Budget 알림**을 꼭 설정하세요(아래 참고).

### 1. 결제 수단 등록
- AWS 계정 생성 시 카드 필요 (이미 있으면 통과).

### 2. Terraform용 IAM 사용자 + 액세스 키 발급 ⭐
Terraform이 AWS를 제어하려면 자격증명이 필요합니다. **루트 대신 IAM 사용자**를 권장합니다:
1. 콘솔 → **IAM** → **Users** → **Create user** (예: `terraform-admin`)
2. 권한: **Attach policies directly** → **AdministratorAccess** 체크
   (EC2/VPC/S3/IAM 리소스를 모두 만들기 때문. 개인 프로젝트 기준 권장)
3. 생성 후 사용자 → **Security credentials** → **Create access key** → "CLI" 선택
4. **Access key ID / Secret access key** 를 안전히 보관 (Secret은 이때만 보임 🔴)

### 3. AWS CLI 설치 + 자격증명 등록 (로컬 PC)
```bash
brew install awscli          # 없으면 설치
aws configure
#  AWS Access Key ID     : (2번에서 발급한 키)
#  AWS Secret Access Key : (2번 시크릿)
#  Default region name   : ap-northeast-2
#  Default output format  : json
aws sts get-caller-identity  # 정상 출력되면 인증 OK
```

### 4. (선택) 비용 알림
- **Billing → Budgets → Create budget** → 월 $1, 실제 지출 $0.01 초과 시 `seeyouspace0@gmail.com` 알림.

> SSH 키는 이미 있는 `~/.ssh/id_ed25519.pub`를 재사용합니다. 따로 만들 필요 없습니다.

---

## 배포 절차

### 1. 변수 작성
```bash
cd infra/aws/terraform
cp terraform.tfvars.example terraform.tfvars
```
`terraform.tfvars` 편집:
```hcl
region         = "ap-northeast-2"
ssh_public_key = "ssh-ed25519 AAAA... seeyouspace0@gmail.com"   # cat ~/.ssh/id_ed25519.pub
# ssh_allowed_cidr = "<내IP>/32"   # curl ifconfig.me 로 확인 후 입력(권장)
```

### 2. 생성
```bash
terraform init
terraform plan
terraform apply
```
생성물: VPC/서브넷/IGW/보안그룹, EC2(t2.micro, Docker+swap 자동 설치), S3 버킷, 앱용 IAM 키.

### 3. 출력값 확인
```bash
terraform output
terraform output -raw s3_access_key
terraform output -raw s3_secret_key
```

| output | .env 항목 |
|---|---|
| `instance_public_ip` | 접속 IP |
| `s3_endpoint` | `STORAGE_ENDPOINT` |
| `s3_public_base_url` | `STORAGE_PUBLIC_BASE_URL` |
| `s3_bucket` | `STORAGE_BUCKET` |
| `s3_access_key` | `STORAGE_ACCESS_KEY` |
| `s3_secret_key` | `STORAGE_SECRET_KEY` |

### 4. 서버 배포
```bash
ssh ubuntu@<instance_public_ip>
#  Docker는 cloud-init이 자동 설치(1~2분). docker ps 안 되면 잠시 후 재로그인.

git clone https://github.com/makerspaceSiwoo/cocktail-mate-server.git
cd cocktail-mate-server
git checkout other/cm-32-setup-server

cp .env.example .env
nano .env
```
`.env` 값 (DB 비번은 임의 강한 값, 스토리지는 3번 output):
```dotenv
APP_ENV=production
POSTGRES_DB=cocktail_mate
POSTGRES_USER=app_user
POSTGRES_PASSWORD=<강한_비밀번호>
DATABASE_URL=postgresql+psycopg2://app_user:<강한_비밀번호>@db:5432/cocktail_mate

STORAGE_ENDPOINT=https://s3.ap-northeast-2.amazonaws.com
STORAGE_REGION=ap-northeast-2
STORAGE_ACCESS_KEY=<s3_access_key>
STORAGE_SECRET_KEY=<s3_secret_key>
STORAGE_BUCKET=<s3_bucket>
STORAGE_PUBLIC_BASE_URL=<s3_public_base_url>
```
기동:
```bash
docker compose -f docker-compose.prod.yml up -d --build
```

### 5. 검증
```bash
curl http://<instance_public_ip>/health    # {"db":"ok","vector":"ok","storage":"ok"}
```
- Swagger: `http://<instance_public_ip>/docs`

### 6. 정리 / 비용
- 전체 삭제: `terraform destroy`
- EC2는 12개월/750시간 한도 내 무료, S3 5GB 무료. 한도 초과/12개월 경과 시 과금 → Budget 알림 권장.

---

## 참고: AWS S3 vs 로컬 MinIO
앱 코드(`app/core/storage.py`)는 S3 호환 API라 **AWS S3에서도 코드 수정 없이** 동작합니다.
로컬 개발은 그대로 MinIO(`docker-compose.yml`), 프로덕션은 AWS S3(이 `.env` 값)만 바꾸면 됩니다.
