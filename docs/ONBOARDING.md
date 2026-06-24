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

### 빌드 전제: GitHub SSH 키 (최초 1회)

ORM 모델 패키지(`cocktail-mate-db`, **private 레포**)를 빌드 중 git+ssh로 설치한다.
`make up`은 빌드 전에 `make ssh-check`로 GitHub 인증을 먼저 확인하므로,
아래 3가지가 갖춰져야 빌드가 통과한다. (공용 키는 없다 — 각자 본인 GitHub 키를 쓴다.)

**1) `cocktail-mate-db` 레포 접근권한**
팀 관리자에게 본인 GitHub 계정을 해당 private 레포 collaborator로 초대해 달라고 요청.

**2) GitHub에 SSH 키 등록**
키가 없으면 생성 후 https://github.com/settings/keys 에 공개키 등록:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"   # 이미 키가 있으면 생략
cat ~/.ssh/id_ed25519.pub                            # 이 값을 GitHub에 등록
```

**3) 키를 ssh-agent에 로드 (재부팅 후에도 유지 — 1회면 끝)**
`make up`(`ssh: ["default"]`)은 호스트 ssh-agent의 키를 빌드에 자동으로 빌려 쓴다.
키 파일을 따로 지정할 필요 없이, **agent에 키를 한 번 넣어두면 빌드가 알아서 가져간다.**
관건은 agent가 재부팅 후에도 키를 들고 있게 만드는 것(=「매번」설치 실패의 흔한 원인).

**macOS** — keychain에 영구 등록:

```bash
ssh-add -l                                       # 키가 보이면 OK. 비어 있으면:
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
```

재부팅 후 자동 로드되게 `~/.ssh/config`에 한 번 추가:

```
Host github.com
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519
```

**Windows** — PowerShell(관리자)에서 ssh-agent 서비스를 자동 시작으로 켜고 키 등록.
Windows ssh-agent는 등록한 키를 자격증명 저장소에 보관하므로 **말 그대로 평생 1회**다:

```powershell
Set-Service ssh-agent -StartupType Automatic
Start-Service ssh-agent
ssh-add $env:USERPROFILE\.ssh\id_ed25519
```

> Docker Desktop이 Windows ssh-agent(named pipe)를 빌드에 연결한다. 위 설정 후
> `make up`이 자동으로 키를 사용한다.

**확인 (공통):** `ssh -T git@github.com` → `Hi <username>! You've successfully authenticated`
가 떠야 정상. 안 뜨면 위 1~3 중 빠진 게 있다.

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
- **모델이 바뀌면?** → db 레포에서 태그가 올라간다. `requirements.txt`의 태그를 올리고 `docker compose build`.
- **이미지 스토리지는?** → 팀 `.env`가 Oracle Object Storage를 가리킨다. 오프라인 작업은 `--profile local-storage`로 MinIO 기동.
