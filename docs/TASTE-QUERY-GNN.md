# 맛 표현 조합 기반 칵테일 추천

## 목표

사용자가 고른 맛 표현 ID 목록을 문장으로 합성하지 않고 32차원 가상 칵테일 임베딩으로 변환한다. 이 벡터로 `cocktails.embedding`의 pgvector ANN 검색을 수행해 가장 가까운 칵테일 5개를 반환한다.

온라인 요청에서는 Gemini나 다른 외부 API를 호출하지 않는다. PyTorch는 오프라인 학습에만 사용하며, 서버 추론은 저장된 파라미터와 NumPy만 사용한다.

## 데이터와 그래프

학습 그래프의 노드는 다음 세 종류다.

- 칵테일 602개
- 실제 문장에서 관측해 보편화한 맛 표현 43개
- 실제 레시피에서 사용된 재료 350개

칵테일-맛 간선은 기존 맛 표현 문장에서 제어 어휘를 추출해 만들고, 칵테일-재료 간선은 레시피의 재료와 용량으로 만든다. 용량은 단위를 ml 기준으로 환산할 수 있는 경우 환산한 뒤 `log1p` 가중치를 적용한다.

칵테일 노드의 정답은 로컬 텍스트 임베딩을 이웃 보존 방식으로 축소해 만든 기존 32차원 임베딩이다. 학습 모델은 맛 노드와 재료 노드의 메시지를 각각 모아 공유 출력 헤드를 거치는 이종 그래프 디코더다. 손실에는 다음 항목을 함께 사용한다.

- 정답 32차원 벡터와의 코사인 손실
- 정답 공간의 가까운 이웃 분포를 보존하는 distillation 손실
- 재료만으로 같은 칵테일 공간을 복원하는 보조 손실
- 맛 메시지와 재료 메시지의 잠재 표현 정렬 손실

맛 표현 일부를 무작위로 제거하는 descriptor dropout을 사용해 학습 문장 전체가 아니라 사용자가 고른 부분 집합도 처리하도록 했다.

## 학습 결과

고정 seed `20260801`, 3회 restart에서 검증 recall@5를 우선하고 정답 코사인을 보조 기준으로 모델을 선택했다.

| 평가 | 결과 |
| --- | ---: |
| held-out 칵테일 120개의 맛 조합 recall@5 | 10.33% |
| held-out 정답 임베딩 평균 cosine | 0.2831 |
| 전체 데이터 재학습 후 맛 조합 recall@5 | 14.32% |
| 단일 맛 표현 top-5 macro precision | 45.58% |
| 같은 표현의 평균 카탈로그 출현율 | 10.36% |
| 단일 표현 precision lift | 4.40배 |

43개 표현은 모두 실제 문장 5개 이상에서 관측된다. 각 카테고리에서는 문장에 가장 먼저 등장한 대표 표현 하나만 학습 간선으로 사용한다. 현재 데이터가 602개뿐이고 개별 과일·향을 넓은 범주로 합쳤으므로 임의의 복잡한 조합을 정확히 복원하는 생성 모델로 보기는 어렵다. 관측된 보편 맛 범주에서 가까운 칵테일을 찾는 검색용 디코더로 사용한다.

| 카테고리 | 선택지 수 | 예시 |
| --- | ---: | --- |
| fruit | 7 | 시트러스류, 베리류, 핵과류, 열대과일류 |
| aroma | 10 | 허브·보태니컬, 향신료, 꽃향, 오크·스모키 |
| mouthfeel | 6 | 청량함, 매끄러움, 크리미함, 탄산감 |
| finish | 9 | 깔끔함, 달콤함, 쌉쌀함, 긴 여운 |
| body | 4 | 가벼움, 리치함, 묵직함, 밀도감 |
| temperature | 3 | 매우 차가움, 차가움, 시원함 |
| alcohol | 4 | 부드러움, 깨끗한 스피릿감, 열감, 강렬함 |

원래 생성 스키마에는 `흙`이 있었지만 실제 602개 임베딩 문장에는 한 번도 등장하지 않았다. `알코올 향` 또는 `스피릿 향`이라는 직접 표현도 등장하지 않았다. 따라서 두 표현은 선택지에 넣지 않았고, 실제 등장한 `깨끗한 스피릿감`, `알코올 열감`, `강렬한 알코올감`을 사용한다.

학습 산출물은 다음 파일에 저장된다.

- `app/taste_query/artifacts/taste-query-gnn.npz`: NumPy 추론용 모델
- `app/taste_query/artifacts/taste-query-gnn.metrics.json`: 학습 및 평가 지표

## DB 구조

Alembic revision은 `6fd54a9c81e2`이다.

- `taste_descriptors`: 코드, 한국어 표시명, 카테고리, 노출 순서, 활성 상태

`taste_descriptor_conflicts`와 `cocktail_taste_descriptors` 테이블은 제거했다. 상충 쌍이나 칵테일별 학습 간선을 DB에 중복 저장하지 않고, 모든 카테고리에 동일하게 최대 한 개만 선택할 수 있다. 칵테일-맛 학습 간선은 로컬 CSV에서 매번 구성한다.

## API

### 선택 가능한 맛 표현 조회

```http
GET /taste-descriptors
```

응답의 `items`에는 현재 모델이 지원하는 표현만 포함된다. 프론트엔드는 `category`와 `maxSelectionsPerCategory`를 이용해 같은 카테고리의 다른 표현을 비활성화한다.

```json
{
  "items": [
    {
      "id": 1,
      "code": "fruit.citrus",
      "labelKo": "시트러스류",
      "category": "fruit"
    }
  ],
  "maxSelectionsPerCategory": 1
}
```

### 맛 표현 조합으로 추천

```http
POST /flavor/recommend
Content-Type: application/json

{
  "descriptorIds": [1, 8, 18, 33, 38, 40]
}
```

`descriptorIds`는 0개 이상 7개 이하이고 중복될 수 없다. 같은 카테고리에서는 하나만 선택할 수 있다. 한 개 이상이면 서버가 ID를 canonical code로 바꾼 뒤 모델로 단위 길이의 32차원 벡터를 만들고, `cocktails.embedding`의 cosine ANN 상위 5개를 반환한다. 빈 배열이면 비교할 맛이 없으므로 칵테일 5개를 무작위로 반환하며 `similarity`는 `0.0`이다.

```json
[
  {
    "id": 214,
    "name": "콥스 리바이버",
    "description": "진과 릴레 블랑, 트리플 섹, 레몬즙을 압생트 향과 함께 섞은 상큼한 클래식 칵테일이다.",
    "imageUrl": "https://api.cocktail-mate.com/media/cocktails/214-0ae87f30281e.webp",
    "similarity": 0.4561
  },
  {
    "id": 269,
    "name": "프로즌 민트 다이키리",
    "description": "라이트 럼과 라임즙에 민트를 넣어 얼음과 갈아낸 다이키리다.",
    "imageUrl": "https://api.cocktail-mate.com/media/cocktails/269-3f346cac8ad6.webp",
    "similarity": 0.4553
  },
  {
    "id": 494,
    "name": "럼 사워",
    "description": "라이트 럼에 레몬즙과 설탕을 더한 사워 칵테일이다.",
    "imageUrl": "https://api.cocktail-mate.com/media/cocktails/494-9eaa5f1dd9b4.webp",
    "similarity": 0.4447
  },
  {
    "id": 318,
    "name": "헤밍웨이 스페셜",
    "description": "럼에 자몽 주스, 마라스키노 리큐르, 라임즙을 더한 다이키리 변형이다.",
    "imageUrl": "https://api.cocktail-mate.com/media/cocktails/318-9f7e6f5697ab.webp",
    "similarity": 0.4377
  },
  {
    "id": 30,
    "name": "아담 선라이즈",
    "description": "보드카에 레몬에이드와 물, 설탕을 섞어 만든 가벼운 롱드링크이다.",
    "imageUrl": "https://api.cocktail-mate.com/media/cocktails/30-65f948868149.webp",
    "similarity": 0.4331
  }
]
```

유효하지 않은 ID, 중복 ID, 같은 카테고리의 복수 선택은 `422` 응답을 반환한다. 모델 파일이 없거나 읽을 수 없으면 `503`을 반환한다.

## 재학습

서버의 `.env.local`에 DB 연결 정보가 준비된 상태에서 실행한다.

```bash
make embedding-install
set -a
source .env.local
set +a
make taste-query-train
```

맛 표현과 칵테일의 학습 관계는 `taste-data/cocktail-taste-descriptions.csv`에서 구성하며 DB로 동기화하지 않는다.
