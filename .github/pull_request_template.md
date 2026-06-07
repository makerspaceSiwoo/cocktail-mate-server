<!--
제목 형식: [작업범주][내용]   예) [Feat] 유저정보 CRUD 구현
작업범주는 README 참고.
-->

## 작업 내용
<!-- 무엇을, 왜 했는지 요약 -->

## 관련 이슈
closes #

## 변경 사항 (계층별)
<!-- MVC 구조 기준. 해당하는 계층만 작성하세요. -->
- **Router (Controller)**:
- **Service** (비즈니스 로직):
- **Repository** (데이터 접근):
- **Schema** (Pydantic 요청/응답):
- **Model** (SQLAlchemy):
- **Core / Infra** (config·database·storage·docker·terraform):

## API 변경
- [ ] 엔드포인트 추가/변경 없음
<!-- 있으면 아래에 작성
- `METHOD /path` — 설명
-->

## 테스트
- [ ] `make up` 으로 로컬 스택 기동 확인
- [ ] `/health` 가 `{db, vector, storage}` 모두 ok
- [ ] 관련 엔드포인트 동작 확인 (`/docs`)
- [ ] 예외 처리 확인

## 고려 사항 / 참고
-
