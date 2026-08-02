# 칵테일 상세 맛 CSV 생성

`scripts.generate_cocktail_taste_descriptions`는 칵테일 DB를 읽고 Gemini로
정규화된 맛 프로필을 생성한 뒤 UTF-8 CSV에 저장한다. DB는 읽기만 하며
`cocktails.description`을 포함한 어떤 칼럼도 변경하지 않는다.

## 출력 설계

기본 출력 경로는 `taste-data/cocktail-taste-descriptions.csv`이고 열은 정확히
다음 5개다.

| 열 | 내용 |
| --- | --- |
| `cocktail_id` | `cocktails.id` |
| `cocktail_name_ko` | 한글 이름 |
| `cocktail_name_en` | 영문 이름, 없으면 빈 값 |
| `recipe` | JSON 문자열 배열로 보존한 레시피 |
| `embedding_text` | ANN 벡터 생성에 직접 사용하는 긍정형 자연어 맛 설명 |

Gemini 응답은 서로 목적이 다른 두 계층으로 나뉜다.

- `facets`: 단맛·산미·쓴맛·바디·탄산·크리미함·온도·알코올감·여운 등을
  `NONE`, `LOW`, `MEDIUM`, `HIGH` 계열 enum으로 평가한다. 향후 DB의 하드 필터와
  품질 검증에 사용하며 임베딩하지 않는다.
- `embedding_text`: 실제로 감지되는 지배적인 향, 풍미, 질감, 목넘김과 여운만
  2~4개의 자연스러운 한국어 문장으로 표현한다. CSV에는 이 값만 저장한다.

`embedding_text`에서는 `단맛 없음`, `탄산 없음`, `산미가 낮음`처럼 부재한
감각의 이름을 다시 노출하는 표현을 금지한다. 낮은 단맛은 드라이함·깔끔함,
낮은 산미는 부드럽고 둥근 목넘김, 가벼운 바디는 라이트하고 산뜻한 질감처럼
긍정형 감각어로 바꾼다. 중요하지 않은 축은 아예 언급하지 않는다.

색, 투명도, 층, 잔, 가니시, 역사, 유래 같은 비미각 정보와 숫자·ABV·단계명도
임베딩 문장에서 금지한다. 과일·허브·향신료·플로럴·오크·로스팅·유제품 계열
향은 구조화된 공통 어휘로 먼저 선택한 뒤 자연어 문장에 반영한다.

예시:

```text
레몬의 선명한 시트러스 산미와 주니퍼·허브 향이 생생하게 어우러지고, 은은한 단맛이 전체 풍미의 균형을 잡는다. 차갑고 산뜻한 질감과 매끄러운 목넘김 뒤로 깨끗한 스피릿 온기와 새콤하고 드라이한 여운이 또렷하게 이어진다.
```

## 입력 근거

Gemini에는 이름과 레시피뿐 아니라 다음 DB 값도 함께 전달한다.

- 완성 칵테일 ABV와 `base_tag`
- 모든 재료의 한글·영문 이름
- 재료별 용량과 단위
- 재료 카테고리, 설명, ABV

레시피와 계량값을 이름에 관한 일반 지식보다 우선하도록 프롬프트에 명시했다.
셰이크만 했다는 이유로 크리미함을 만들거나, 먹지 않는 가니시의 맛을 본문에
넣지 않도록 하는 규칙도 포함한다.

## 실행

`.env.local`(우선) 또는 `.env`에 아래 값을 설정한다. 이미지 생성기와 같은
Gemini 환경변수를 재사용한다.

```dotenv
GEMINI_API_KEY=<GOOGLE_AI_STUDIO_API_KEY>
GEMINI_TEXT_MODEL=gemini-3.5-flash-lite
GEMINI_MIN_REQUEST_INTERVAL_SECONDS=32
GEMINI_MAX_TRANSIENT_RETRIES=3
GEMINI_MAX_VALIDATION_RETRIES=4
```

모델 접근, DB 데이터, 기존 CSV 형식을 먼저 검사한다.

```bash
.venv/bin/python -m scripts.generate_cocktail_taste_descriptions preflight
```

일부 칵테일로 출력 형식을 확인한다.

```bash
.venv/bin/python -m scripts.generate_cocktail_taste_descriptions run --limit 3
```

전체를 생성한다.

```bash
.venv/bin/python -m scripts.generate_cocktail_taste_descriptions run
```

특정 ID만 처리할 수도 있다.

```bash
.venv/bin/python -m scripts.generate_cocktail_taste_descriptions run \
  --cocktail-id 12 \
  --cocktail-id 38
```

완료된 각 행마다 CSV 전체를 원자적으로 교체하므로 중단해도 완료 데이터는
남는다. 다시 실행하면 이름과 레시피가 동일하고 맛 설명이 존재하는 행은
건너뛴다. 레시피가 바뀐 행은 자동 재생성하며, 모델이나 프롬프트 버전 변경 후
전체를 다시 만들 때는 `--force`를 사용한다.

```bash
.venv/bin/python -m scripts.generate_cocktail_taste_descriptions run --force
```

`--dry-run`은 Gemini 호출과 CSV 쓰기 없이 처리 대상만 로그로 보여준다.

## 임베딩 파이프라인 연결 시 주의

이 CSV의 `embedding_text`만 로컬 text embedding 모델에 전달한다. 512차원은
출력 벡터 차원이며 모델의 최대 입력 토큰 수와는 별개이므로, 실제 모델의 토큰
제한과 pooling/정규화 방식을 별도로 고정해야 한다. cosine ANN에 저장하기 전에는
벡터를 L2 정규화하고, 문장→벡터 생성에 사용한 모델 버전과 프롬프트 버전을
함께 관리하는 것이 좋다.

향후 DB에 반영할 때는 기존 `description` 칼럼을 덮어쓰지 말고 별도
`embedding_text`, `taste_embedding vector(512)`와 정량 facet 칼럼 또는 전용
테이블을 추가하는 마이그레이션을 사용한다. 자연어 임베딩과 facet 필터는 서로
대체하는 값이 아니라 함께 사용한다.
