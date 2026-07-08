# cocktail-mate-server
다음 사항들을 토대로 작업해 주세요

## 1. 작업 순서
1) Notion Backend Todo 테이블에 작업 추가 or Todo 테이블의 작업 선택

2) 해당 작업에 해당하는 issue 생성하고 Notion Table에 이슈번호 추가
    issue 명은 [작업범주][내용]
    ex) [Feat] 유저정보 CRUD 구현

3) 해당 issue에 대응되는 branch 생성 후 Notion Table에 branch 명 추가 

   branch 명은 [issue 작업범주]/#[issue num]-~~ (영어로 작성)
   ex) issue number 1이 유저정보 crud 구현이라면 해당 이슈 branch 명은
        feat/#1-userInfo-crud

4) 해당 branch에서 작업 후 로컬에서 동작 확인
확인 방법은 
    3.로컬 환경 세팅 및 실행 방법
에 나와있습니다. 

5) 로컬 확인 후 PR

6) copilot & 팀원 리뷰 완료 후 부모 branch에 merge(main에는 XX)

## 2. 작업 범주에
| 커밋 유형 | 의미 |
|---|---|
| Feat | 새로운 기능 추가 |
| Fix | 버그 수정 |
| Docs | 문서 수정 |
| Style | 코드 formatting, 세미콜원 누락, 코드 자체의 변경이 없는 경우 |
| Refactor | 코드 리팩토링 |
| Test | 테스트 코드, 리팩토링 테스트 코드 추가 |
| Chore | 패키지 매니저 수정, 그 외 기타 수정 (ex. `.gitignore`) |
| Design | CSS 등 사용자 UI 디자인 변경 |
| Comment | 필요한 주석 추가 및 변경 |
| Rename | 파일 또는 폴더 명을 수정하거나 옮기는 작업만인 경우 |
| Remove | 파일을 삭제하는 작업만 수행한 경우 |
| !BREAKING CHANGE | 커다란 API 변경의 경우 |
| !HOTFIX | 급하게 치명적인 버그를 고쳐야 하는 경우 |

## 3. 로컬 환경 세팅 및 실행 방법
0) 요구사항
Docker를 설치해주시기 바랍니다.
아래 명령어들을 실행하려면 Docker가 실행중 이어야합니다. 

1) 환경 세팅
모두 동일한 환경에서 개발을 진행하기 위해 Dockerfile로 환경을 구성했습니다.

`.env` 파일을 팀 채널에서 받아 레포 루트에 두면 끝입니다 — 자세한 절차는
[docs/ONBOARDING.md](docs/ONBOARDING.md) 참고. (로컬 DB 없음 — OCI 원격 개발 DB에 직접 접속)
DB 스키마/운영, ORM 모델, 인프라(Terraform)는 별도 private 레포
[cocktail-mate-db](https://github.com/makerspaceSiwoo/cocktail-mate-db)가 담당합니다.

기타 자세한 사항은 Makefile을 참고하시면 됩니다.

2) 환경 세팅 명령어 흐름
make up: Docker 컨테이너 실행
make shell: 서버 컨테이너 접속
DB 직접 조회: DBeaver로 OCI 개발 DB 접속 (docs/ONBOARDING.md 참고)

3) 로컬 확인
make check: 컴파일이 되는지 체크
Swagger: http://localhost:8000/docs 접속

## 4. Git Hooks (협업 규칙, 최초 1회 설정)
레포 클론 후 한 번 실행하세요:

    make hooks

- **pre-commit**: 브랜치명 규칙 검증
- **pre-push**: main 직접 push 차단 + 브랜치명 검증 + build(컴파일) 체크
- 허용 브랜치명: `main | develop | (feat|fix|hotfix|chore|docs|refactor|test|other)/소문자-슬러그`
