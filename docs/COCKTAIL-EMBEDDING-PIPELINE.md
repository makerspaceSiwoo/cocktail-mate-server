# 칵테일 로컬 임베딩·이웃보존 축소 파이프라인

`scripts.build_cocktail_embeddings`는 생성 완료된
`taste-data/cocktail-taste-descriptions.csv`의 `embedding_text`만 읽어 다음 산출물을
만든다.

1. 로컬 text-embedding 모델의 L2 정규화된 512차원 벡터
2. 코사인 이웃 분포를 직접 학습한 512→32차원 축소 모델과 pgvector용 벡터
3. 이웃보존 3D 구 내부 좌표와 구 표면 좌표, 두 방식의 품질 비교 결과
4. 강한 이웃만 보존하는 클러스터 우선 구 표면 좌표

이 스크립트는 Gemini 모듈을 import하지 않고 API 키도 읽지 않는다. 네트워크는
`download-model`에서 Hugging Face의 고정된 모델 파일을 한 번 내려받을 때만 쓴다.
그 뒤 `embed`, `train-32`, `experiment-3d`, `experiment-cluster-surface`,
`apply-db`는 전부 로컬에서 실행된다.

## 단계별 학습 여부

전체 흐름에서 사전 학습 모델의 단순 추론과 현재 칵테일 카탈로그를 대상으로 한
학습을 구분해야 한다.

| 단계 | 이번 작업에서 학습했는가 | 실제 처리 |
| --- | --- | --- |
| 레시피·재료 → 맛 표현 | 아니요 | Gemini의 기존 지식과 DB 입력을 사용한 생성 추론 |
| 맛 표현 → 512D | 아니요 | 고정된 로컬 SentenceTransformer로 encode만 수행 |
| 512D → 32D | 예 | 512D top-k 이웃 분포를 보존하도록 `602 × 32` 좌표를 직접 학습 |
| 32D → 맛 군집 | 예, 비지도 fit | KMeans가 현재 카탈로그의 코사인 구조로 군집을 계산 |
| 군집·32D → 3D 구면 | 예, 비지도 최적화 | 로컬 이웃과 군집 관계를 보존하도록 3D 좌표를 계산 |
| 대표 칵테일 → 맛 프로토타입 | 아니요 | 대표 임베딩을 평균하고 cosine 유사도만 비교하는 분석 단계 |

데이터의 생성 방향은 다음과 같다.

```text
레시피·재료
  ↓ Gemini 추론
맛 표현
  ↓ 고정된 로컬 text encoder
512D 의미 임베딩
  ↓ 현재 카탈로그의 이웃 관계를 teacher로 사용한 학습
32D 추천 좌표
  ↓ 비지도 군집화·이웃보존 최적화
클러스터와 3D 시각 좌표
```

따라서 맛 표현이 임베딩에서 역산되거나 정답 임베딩으로부터 학습된 것은 아니다.
맛 표현이 먼저 만들어지고, 그 문장을 512차원으로 임베딩한 뒤 512차원 이웃
관계를 teacher representation으로 사용해 32차원 좌표를 학습했다. 512차원
벡터도 사람이 부여한 절대적인 정답이 아니라 고정된 사전 학습 모델이 만든 의미
표현이다.

현재 512→32 단계는 임의의 새 벡터를 변환하는 일반적인
`projector(vector_512) -> vector_32` 함수를 학습하지 않는다. 현재 602개
칵테일의 좌표표를 transductive 방식으로 직접 최적화한다. 따라서 새로운 사용자
맛 문장은 같은 로컬 모델로 512차원 query를 만들 수 있지만, 이를 현재 32차원
추천 공간에 넣으려면 이웃 가중 보간 함수를 별도로 검증하거나 전체 카탈로그와
함께 재학습해야 한다.

## 고정한 512차원 모델

- 모델: `sentence-transformers/distiluse-base-multilingual-cased-v2`
- revision: `bfe45d0732ca50787611c0fe107ba278c7f3f889`
- 출력: 512차원 dense vector
- 언어: 한국어(`ko`)를 포함한 다국어 50개
- 라이선스: Apache-2.0

공식 [Hugging Face 모델 카드](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)는
512차원 문장/문단 임베딩과 semantic search 용도를 명시한다.
모델의 최대 문맥은 128 token이므로, 2~4문장인 맛 표현을 문장별로 임베딩한 뒤
평균하고 다시 L2 정규화한다. 이렇게 하면 뒤쪽 여운 문장이 통째로 잘리는 문제를
피하면서 최종 차원은 512로 유지된다.

## 설치와 전체 실행

무거운 ML 패키지는 FastAPI 프로덕션 이미지에 포함하지 않는다. 로컬 venv에만
별도 설치한다.

```bash
.venv/bin/pip install -r requirements-embedding.txt
.venv/bin/python -m scripts.build_cocktail_embeddings preflight
.venv/bin/python -m scripts.build_cocktail_embeddings run-all
```

모델과 산출물 기본 위치는 각각 `embedding-models/`, `embedding-artifacts/`이며
둘 다 gitignore 대상이다. 단계별 실행도 가능하다.

```bash
.venv/bin/python -m scripts.build_cocktail_embeddings download-model
.venv/bin/python -m scripts.build_cocktail_embeddings embed
.venv/bin/python -m scripts.build_cocktail_embeddings train-32
.venv/bin/python -m scripts.build_cocktail_embeddings experiment-3d
.venv/bin/python -m scripts.build_cocktail_embeddings experiment-cluster-surface
.venv/bin/python -m scripts.experiment_embedding_3d_top3
```

모든 산출물에는 입력 파일 SHA-256, 모델 revision, seed, 하이퍼파라미터와 측정
결과를 기록한다. NPZ는 `allow_pickle=False`로 다시 읽을 수 있는 숫자/문자열만
저장하고, 학습 가중치는 safetensors로 저장한다.

`experiment_embedding_3d_top3`는 기존 512D와 학습 32D를 다시 만들지 않는다.
32D 좌표를 고정한 채 UMAP·t-SNE 3D 후보를 CPU에서 비교하고, 3D 최근접 3개가
32D ANN 추천 5개를 하나라도 포함하는 비율로 후보를 선택한다. DB 연결이나
쓰기 작업은 수행하지 않는다.

기본 출력은 gitignore 대상인 `embedding-artifacts/top3-3d-experiment/`에
저장된다.

- `best-ball.npz/csv`: 구 내부 최고 hit 좌표
- `best-surface.npz/csv`: t-SNE 구면 투영 후보 중 최고 hit 좌표
- `metrics.json`: 전체 파라미터 그리드와 기존 클러스터 구면 비교 결과

구면 투영은 제한된 2 자유도의 손실을 정량화하기 위한 후보이며, 자동으로 DB에
반영되지 않는다. DB 적용 전에는 보고서의 클러스터 구면 기준선과 함께 검토한다.

## 512→32 학습 방식

단순 앞 32차원 절단이나 PCA를 최종 모델로 쓰지 않는다. 512차원에서 각 칵테일의
코사인 top-15 이웃 분포를 teacher로 만들고, 카탈로그의 `602 × 32` 좌표 자체를
학습 파라미터로 두어 같은 이웃 확률 분포를 갖도록 cross entropy로 학습한다.
여러 고정 seed restart 중 `recall@5`, `recall@10`, `recall@15` 순으로 가장 좋은
결과를 선택한다.

- `recall@5`, `recall@10`, `recall@15`를 512차원 이웃 기준으로 측정한다.
- 같은 데이터의 PCA 32차원 결과를 baseline으로 함께 기록한다.
- 이 축소는 현재 카탈로그의 관계 보존을 우선하는 transductive 모델이다. 신규
  칵테일이 추가되면 전체 카탈로그를 함께 재학습한다.

추천 API의 source of truth는 최종 L2 정규화 32차원 벡터다. DB 입력용 CSV는
`embedding-artifacts/embeddings-32.csv`, 재현 가능한 원본은
`embedding-artifacts/embeddings-32.npz`다.

## 3D 구 내부와 표면 비교

두 방식 모두 먼저 일반 유클리드 축소를 한 뒤 각 점을 표면으로 정규화하는 방식을
쓰지 않는다.

### `ball`: 구 내부

L2 정규화 32차원 벡터에 `UMAP(metric="cosine", n_components=3)`을 직접
적용한다. 결과 전체에 단 한 번의 평행이동과 동일 배율 스케일만 적용해 단위 구
내부에 맞춘다. 점별 반경 정규화는 하지 않으므로 UMAP의 3 자유도와 유클리드
이웃 순서는 그대로 유지된다.

### `surface`: 구 표면

각 좌표를 처음부터 단위 구면 `S²` 위의 학습 파라미터로 두고, 32차원 코사인
top-10 이웃 분포와 구면 좌표의 이웃 분포가 같아지도록 직접 최적화한다. 여러
고정 seed restart 중 `recall@5`, `recall@10`, `recall@15` 순으로 가장 좋은
결과를 고른다. 따라서 유클리드 3D 결과를 나중에 표면으로 투영하는 단계가 없다.

각 모드에 대해 다음을 기록한다.

- 32차원 추천 공간 기준 `recall@5/10/15`와 cosine-distance stress
- 원본 512차원 기준 `recall@5/10/15`와 stress
- 512차원 PCA 상위 3성분 설명 분산 비율
- 좌표 반경의 최소/평균/최대값

자동 선택은 실제 추천 공간인 32차원 `recall@5`를 우선하고, 동률이면
`recall@10`, stress 순으로 결정한다. 결과는
`embedding-artifacts/embedding-3d-metrics.json`에 남고 선택 좌표는
`embedding-3d-selected.*`로 복사된다.

## 강한 이웃만 보존하는 클러스터 구면

구 내부 탐색의 가독성을 개선하기 위한 별도 실험이다. 전체 602개 점 사이의 먼
거리는 복원하지 않는다.

기존 `ball` 방식은 3차원의 자유도를 모두 사용하므로 이웃 보존률은 상대적으로
높지만, 구 내부의 점이 앞뒤로 겹치고 다른 점에 가려져 사용자가 특정 칵테일이나
맛의 큰 계열을 찾기 어렵다. 반대로 기존 `surface` 방식은 모든 점을 표면에서 볼
수 있지만, 제한된 구면 자유도로 32차원 전체 거리 관계를 동시에 맞추려 하면서
추천 UX에 중요한 가까운 이웃까지 손실했다.

실제 추천 API는 모든 칵테일 쌍의 정확한 거리를 사용하지 않는다. pgvector ANN
검색에서 중요한 것은 각 칵테일과 코사인 유사도가 높은 top-k 이웃이다. 따라서
3D 시각화의 목표를 전체 거리 복원에서 다음과 같이 변경한다.

- 매우 비슷한 칵테일은 같은 구면 영역에 배치한다.
- 같은 맛 계열 안에서 top-1/3/5 이웃을 최대한 가깝게 유지한다.
- 관련성이 낮은 군집 사이의 정확한 거리는 의도적으로 복원하지 않는다.
- 추천 순위는 계속 32차원 벡터로 계산하고 3D는 탐색에만 사용한다.

즉 클러스터 구면은 모든 거리를 정확히 표현하는 지도가 아니라, 비슷한 칵테일을
쉽게 발견하기 위한 구면 탐색 지도다.

1. L2 정규화 32D 코사인 공간에서 K-means `k=7` 군집을 만든다.
2. 각 군집의 32D centroid cosine 유사도와, 원본 top-5 이웃이 군집 경계를
   넘나드는 대칭 edge association을 계산한다.
3. 원본 top-1/3/5/10 이웃이 같은 군집에 남는 비율을 기록한다.
4. 각 군집 내부에서만 `UMAP(metric="cosine", n_neighbors=3, min_dist=0.15)`을
   수행한다. 다른 군집의 먼 점은 이 목적함수에 들어가지 않는다.
5. centroid 관계 25%와 경계 top-5 관계 75%를 합친 top-3 군집 관계가 구면의
   중심 각도 순서로 나타나도록 7개 중심을 직접 학습한다.
6. 중심 벡터 평균의 크기를 줄이는 balance loss를 함께 적용하되 가중치는 `0.5`로
   제한한다. 관계가 있는 군집 중심은 더 가까워질 수 있고 전체가 한 반구로
   완전히 무너지는 것만 방지한다.
7. 각 로컬 배치 반경의 95 percentile을 cap 경계로 사용한다. 소수의 UMAP
   outlier 하나가 나머지 점 전체를 중심 근처에 압축하는 현상을 막으면서 대부분의
   로컬 거리에는 동일한 선형 배율을 적용한다.
8. 최근접 중심 각도의 95% 반경을 갖는 adaptive cap에 exponential map으로
   옮긴다. 반경 상한은 `0.95 rad`이며 실제 cap은 `45.8°~54.4°`다. 가까운
   군집의 cap은 일부 겹칠 수 있고 최종 모든 좌표의 반경은 정확히 1이다.

산출물은 다음과 같다.

- `embedding-3d-cluster-surface.npz/csv`: 3D 구면 좌표
- `embedding-3d-cluster-surface-metrics.json`: 후보 군집과 이웃 보존 지표
- `embedding-3d-cluster-assignments.json`: 칵테일별 군집 ID

이 실험의 주 지표는 global stress가 아니라 `recall@1/3/5`와 군집 내부 recall이다.
먼 군집 사이의 거리는 의도적으로 의미를 부여하지 않는다.

클러스터 구면에는 다음 한계가 있다.

- 서로 다른 cap 사이의 3D 거리는 군집 관계의 근사치이며 개별 칵테일의 정확한
  32차원 거리로 해석할 수 없다.
- 군집 경계의 일부 가까운 이웃은 서로 다른 cap으로 나뉠 수 있다.
- 전체 32차원을 재학습하면 군집 배정과 군집 ID가 달라질 수 있다.
- 군집 ID는 영구적인 맛 카테고리가 아니며 시각화·실험 메타데이터로만 사용한다.

추천의 source of truth는 항상 `cocktails.embedding vector(32)`이다.
`embedding_3d vector(3)`과 실험용 군집 ID는 추천 결과를 계산하지 않으며,
사용자가 결과를 탐색하기 쉽게 배치하는 역할만 담당한다.

## DB 반영

기본 실행은 DB를 변경하지 않는다. 다음 명령은 DB ID와 차원만 검증하고 rollback
한다.

```bash
.venv/bin/python -m scripts.build_cocktail_embeddings apply-db
```

측정 결과를 검토한 뒤에만 `--commit`으로 `cocktails.embedding`과
`cocktails.embedding_3d`, `embedding_updated_at`을 한 트랜잭션에서 갱신한다.

```bash
.venv/bin/python -m scripts.build_cocktail_embeddings apply-db --commit
```

구 내부 모드를 선택했다면 프론트 렌더러에서 좌표를 다시 단위 표면으로
정규화하면 안 된다. 표면 모드는 모든 좌표의 반경이 1이므로 기존 지구본 UX에
바로 사용할 수 있다.

클러스터 구면을 반영하려면 결과 검토 후 명시적으로 새 좌표를 지정한다.

```bash
.venv/bin/python -m scripts.build_cocktail_embeddings apply-db \
  --embedding-3d embedding-artifacts/embedding-3d-cluster-surface.npz \
  --commit
```
