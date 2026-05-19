# cocktail-mate-server
다음 사항들을 토대로 작업해 주세요

## 1. 작업 순서
1) Notion Backend Todo 테이블에 작업 추가 or Todo 테이블의 작업 선택

2) 해당 작업에 해당하는 issue 생성하고 Notion Table에 이슈번호 추가
    issue 명은 [작업범주][내용]
    ex) [Feat] 유저정보 CRUD 구현

3) 해당 issue에 대응되는 branch 생성 후 Notion Table에 branch 명 추가 
    (branch는 최소한 dev branch에서 분기할것!
    작업전 pull 필수! 
    main은 실행 가능한 상태유지해야 할 수 있기 때문)

   branch 명은 [issue 작업범주]/#[issue num]-~~ (영어로 작성)
   ex) issue number 1이 유저정보 crud 구현이라면 해당 이슈 branch 명은
        feat/#1-userInfo-crud

4) 해당 branch에서 작업 후 로컬 테스트

5) 로컬 테스트 통과시 PR

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